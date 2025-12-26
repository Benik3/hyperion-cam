#include "gpuNv12Transform.h"

#ifdef HAVE_GPU_TRANSFORM

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
static inline uint64_t monoNs()
{
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
										std::chrono::steady_clock::now().time_since_epoch())
										.count());
}

static bool envEnabled(const char *name)
{
	const char *v = std::getenv(name);
	if (!v)
		return false;
	return (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
}

static std::string eglErrStr(EGLint err)
{
	switch (err)
	{
	case EGL_SUCCESS:
		return "EGL_SUCCESS";
	case EGL_NOT_INITIALIZED:
		return "EGL_NOT_INITIALIZED";
	case EGL_BAD_ACCESS:
		return "EGL_BAD_ACCESS";
	case EGL_BAD_ALLOC:
		return "EGL_BAD_ALLOC";
	case EGL_BAD_ATTRIBUTE:
		return "EGL_BAD_ATTRIBUTE";
	case EGL_BAD_CONTEXT:
		return "EGL_BAD_CONTEXT";
	case EGL_BAD_CONFIG:
		return "EGL_BAD_CONFIG";
	case EGL_BAD_CURRENT_SURFACE:
		return "EGL_BAD_CURRENT_SURFACE";
	case EGL_BAD_DISPLAY:
		return "EGL_BAD_DISPLAY";
	case EGL_BAD_SURFACE:
		return "EGL_BAD_SURFACE";
	case EGL_BAD_MATCH:
		return "EGL_BAD_MATCH";
	case EGL_BAD_PARAMETER:
		return "EGL_BAD_PARAMETER";
	case EGL_BAD_NATIVE_PIXMAP:
		return "EGL_BAD_NATIVE_PIXMAP";
	case EGL_BAD_NATIVE_WINDOW:
		return "EGL_BAD_NATIVE_WINDOW";
	case EGL_CONTEXT_LOST:
		return "EGL_CONTEXT_LOST";
	default:
		return "EGL_ERROR_" + std::to_string(err);
	}
}

static EGLDisplay getEglDisplayHeadless()
{
	// First try the default display.
	EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (dpy == EGL_NO_DISPLAY)
		return EGL_NO_DISPLAY;

	EGLint maj = 0, min = 0;
	if (eglInitialize(dpy, &maj, &min))
		return dpy;

	// Default display failed to initialize (common on headless without X/Wayland).
	// Try Mesa surfaceless platform if available.
	PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay =
		reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
	if (!getPlatformDisplay)
		return EGL_NO_DISPLAY;

#ifdef EGL_PLATFORM_SURFACELESS_MESA
	EGLDisplay surf = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
	if (surf != EGL_NO_DISPLAY)
	{
		if (eglInitialize(surf, &maj, &min))
			return surf;
	}
#endif

	return EGL_NO_DISPLAY;
}

static GLuint compileShader(GLenum type, const char *src)
{
	GLuint sh = glCreateShader(type);
	glShaderSource(sh, 1, &src, nullptr);
	glCompileShader(sh);
	GLint ok = 0;
	glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
	if (!ok)
	{
		char log[2048];
		GLsizei len = 0;
		glGetShaderInfoLog(sh, sizeof(log), &len, log);
		std::string msg = "Shader compile failed: ";
		msg.append(log, log + len);
		glDeleteShader(sh);
		throw std::runtime_error(msg);
	}
	return sh;
}

static GLuint linkProgram(GLuint vs, GLuint fs)
{
	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);
	GLint ok = 0;
	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok)
	{
		char log[2048];
		GLsizei len = 0;
		glGetProgramInfoLog(prog, sizeof(log), &len, log);
		std::string msg = "Program link failed: ";
		msg.append(log, log + len);
		glDeleteProgram(prog);
		throw std::runtime_error(msg);
	}
	return prog;
}

static void checkGl(const char *what)
{
	GLenum e = glGetError();
	if (e != GL_NO_ERROR)
	{
		throw std::runtime_error(std::string("GL error after ") + what + ": " + std::to_string(static_cast<int>(e)));
	}
}

static std::array<float, 9> mat3FromCv64(const cv::Mat &m)
{
	std::array<float, 9> out{};
	// GLSL expects column-major layout when transpose is GL_FALSE.
	// OpenCV stores matrices conceptually in row-major accessors.
	out[0] = static_cast<float>(m.at<double>(0, 0));
	out[1] = static_cast<float>(m.at<double>(1, 0));
	out[2] = static_cast<float>(m.at<double>(2, 0));
	out[3] = static_cast<float>(m.at<double>(0, 1));
	out[4] = static_cast<float>(m.at<double>(1, 1));
	out[5] = static_cast<float>(m.at<double>(2, 1));
	out[6] = static_cast<float>(m.at<double>(0, 2));
	out[7] = static_cast<float>(m.at<double>(1, 2));
	out[8] = static_cast<float>(m.at<double>(2, 2));
	return out;
}

} // namespace

struct GpuNv12Transform::Impl
{
	EGLDisplay dpy{EGL_NO_DISPLAY};
	EGLContext ctx{EGL_NO_CONTEXT};
	EGLSurface surf{EGL_NO_SURFACE};

	GLuint vao{0};
	GLuint vbo{0};

	GLuint progY{0};
	GLint uMatY{-1};
	GLint uTexY{-1};
	GLuint texY{0};
	GLuint texYOut{0};
	GLuint fboY{0};

	GLuint progUV{0};
	GLint uMatUV{-1};
	GLint uTexUV{-1};
	GLuint texUV{0};
	GLuint texUVOut{0};
	GLuint fboUV{0};

	// Optional pipelined readback via PBOs (reduces blocking on glReadPixels).
	bool usePbo{false};
	GLuint pboY[2]{0, 0};
	GLuint pboUV[2]{0, 0};
	GLsync fenceY[2]{0, 0};
	GLsync fenceUV[2]{0, 0};
	int pboIndex{0};
	std::size_t yBytes{0};
	std::size_t uvBytes{0};
	uint8_t *lastOutY{nullptr};
	uint8_t *lastOutUV{nullptr};
	bool haveLast{false};

	int srcW{0}, srcH{0};
	int dstW{0}, dstH{0};

	std::array<float, 9> invY{};
	std::array<float, 9> invUV{};

	bool profile{false};
	bool profileSync{false};
	uint64_t profFrames{0};
	uint64_t accUploadNs{0};
	uint64_t accUploadYNs{0};
	uint64_t accUploadUVNs{0};
	uint64_t accDrawYNs{0};
	uint64_t accReadYNs{0};
	uint64_t accReadKickYNs{0};
	uint64_t accDrawUVNs{0};
	uint64_t accReadUVNs{0};
	uint64_t accReadKickUVNs{0};
	uint64_t accConsumeNs{0};
	uint64_t accReuseCopyNs{0};
	uint64_t accTotalNs{0};
	uint64_t accXformTotalNs{0};
	uint64_t lastPrintNs{0};

	Impl(int srcWidth, int srcHeight, const cv::Point2f *srcQuad, int dstWidth, int dstHeight)
		: srcW(srcWidth), srcH(srcHeight), dstW(dstWidth), dstH(dstHeight)
	{
		profile = envEnabled("HYPERCAM_GL_PROFILE");
		profileSync = envEnabled("HYPERCAM_GL_PROFILE_SYNC");
		usePbo = envEnabled("HYPERCAM_NV12_PBO");
		lastPrintNs = monoNs();

		// EGL init (headless pbuffer). We try default display first, then fall back
		// to Mesa surfaceless if the default display can't initialize.
		dpy = getEglDisplayHeadless();
		if (dpy == EGL_NO_DISPLAY)
		{
			EGLint err = eglGetError();
			throw std::runtime_error("EGL display init failed (" + eglErrStr(err) + "). "
							"On headless systems you may need KMS/DRM enabled or set EGL_PLATFORM=surfaceless.");
		}

		static const EGLint cfgAttribs[] = {
			EGL_SURFACE_TYPE,
			EGL_PBUFFER_BIT,
			EGL_RENDERABLE_TYPE,
			EGL_OPENGL_ES3_BIT,
			EGL_RED_SIZE,
			8,
			EGL_GREEN_SIZE,
			8,
			EGL_BLUE_SIZE,
			8,
			EGL_ALPHA_SIZE,
			8,
			EGL_NONE};
		EGLConfig cfg;
		EGLint n = 0;
		if (!eglChooseConfig(dpy, cfgAttribs, &cfg, 1, &n) || n != 1)
			throw std::runtime_error("eglChooseConfig failed");

		static const EGLint pbufAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
		surf = eglCreatePbufferSurface(dpy, cfg, pbufAttribs);
		if (surf == EGL_NO_SURFACE)
			throw std::runtime_error("eglCreatePbufferSurface failed");

		static const EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
		if (!eglBindAPI(EGL_OPENGL_ES_API))
			throw std::runtime_error("eglBindAPI failed");
		ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttribs);
		if (ctx == EGL_NO_CONTEXT)
			throw std::runtime_error("eglCreateContext failed");
		if (!eglMakeCurrent(dpy, surf, surf, ctx))
			throw std::runtime_error("eglMakeCurrent failed");

		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// Compute inverse homographies for Y and UV output->input sampling
		cv::Point2f dstQuadY[4] = {
			cv::Point2f(0.f, 0.f),
			cv::Point2f(static_cast<float>(dstW), 0.f),
			cv::Point2f(0.f, static_cast<float>(dstH)),
			cv::Point2f(static_cast<float>(dstW), static_cast<float>(dstH))};
		cv::Mat H = cv::getPerspectiveTransform(srcQuad, dstQuadY);
		cv::Mat inv;
		cv::invert(H, inv);
		invY = mat3FromCv64(inv);

		cv::Point2f srcQuadUV[4];
		for (int i = 0; i < 4; i++)
			srcQuadUV[i] = cv::Point2f(srcQuad[i].x / 2.f, srcQuad[i].y / 2.f);
		cv::Point2f dstQuadUV[4] = {
			cv::Point2f(0.f, 0.f),
			cv::Point2f(static_cast<float>(dstW) / 2.f, 0.f),
			cv::Point2f(0.f, static_cast<float>(dstH) / 2.f),
			cv::Point2f(static_cast<float>(dstW) / 2.f, static_cast<float>(dstH) / 2.f)};
		cv::Mat Huv = cv::getPerspectiveTransform(srcQuadUV, dstQuadUV);
		cv::Mat invuv;
		cv::invert(Huv, invuv);
		invUV = mat3FromCv64(invuv);

		// Fullscreen triangle
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		const float verts[] = {
			-1.f,
			-1.f,
			3.f,
			-1.f,
			-1.f,
			3.f,
		};
		glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);

		const char *vsSrc = R"(
#version 300 es
layout(location=0) in vec2 aPos;
out vec2 vPos;
void main(){
  vPos = aPos;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

		const char *fsYSrc = R"(
#version 300 es
precision highp float;
uniform mat3 uInv;
uniform sampler2D uTex;
in vec2 vPos;
out vec4 outColor;
void main(){
  // vPos is in clip space [-1..1]; convert to pixel coords in destination
  vec2 frag = gl_FragCoord.xy - vec2(0.5, 0.5);
  vec3 src = uInv * vec3(frag, 1.0);
  vec2 s = src.xy / src.z;
  // Sample from GL_RED texture; normalized coords
  vec2 uv = (s + vec2(0.5, 0.5)) / vec2(textureSize(uTex, 0));
  float y = texture(uTex, uv).r;
  outColor = vec4(y, 0.0, 0.0, 1.0);
}
)";

		const char *fsUVSrc = R"(
#version 300 es
precision highp float;
uniform mat3 uInv;
uniform sampler2D uTex;
in vec2 vPos;
out vec4 outColor;
void main(){
  vec2 frag = gl_FragCoord.xy - vec2(0.5, 0.5);
  vec3 src = uInv * vec3(frag, 1.0);
  vec2 s = src.xy / src.z;
  vec2 uv = (s + vec2(0.5, 0.5)) / vec2(textureSize(uTex, 0));
  vec2 c = texture(uTex, uv).rg;
  outColor = vec4(c.r, c.g, 0.0, 1.0);
}
)";

		GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
		GLuint fsY = compileShader(GL_FRAGMENT_SHADER, fsYSrc);
		GLuint fsUV = compileShader(GL_FRAGMENT_SHADER, fsUVSrc);

		progY = linkProgram(vs, fsY);
		uMatY = glGetUniformLocation(progY, "uInv");
		uTexY = glGetUniformLocation(progY, "uTex");
		progUV = linkProgram(vs, fsUV);
		uMatUV = glGetUniformLocation(progUV, "uInv");
		uTexUV = glGetUniformLocation(progUV, "uTex");

		glDeleteShader(vs);
		glDeleteShader(fsY);
		glDeleteShader(fsUV);

		// Input textures
		glGenTextures(1, &texY);
		glBindTexture(GL_TEXTURE_2D, texY);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, srcW, srcH, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

		glGenTextures(1, &texUV);
		glBindTexture(GL_TEXTURE_2D, texUV);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, srcW / 2, srcH / 2, 0, GL_RG, GL_UNSIGNED_BYTE, nullptr);

		// Output textures + FBOs
		glGenTextures(1, &texYOut);
		glBindTexture(GL_TEXTURE_2D, texYOut);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, dstW, dstH, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

		glGenFramebuffers(1, &fboY);
		glBindFramebuffer(GL_FRAMEBUFFER, fboY);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texYOut, 0);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			throw std::runtime_error("FBO Y not complete");

		glGenTextures(1, &texUVOut);
		glBindTexture(GL_TEXTURE_2D, texUVOut);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, dstW / 2, dstH / 2, 0, GL_RG, GL_UNSIGNED_BYTE, nullptr);

		glGenFramebuffers(1, &fboUV);
		glBindFramebuffer(GL_FRAMEBUFFER, fboUV);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texUVOut, 0);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			throw std::runtime_error("FBO UV not complete");

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		yBytes = static_cast<std::size_t>(dstW) * static_cast<std::size_t>(dstH);
		uvBytes = (static_cast<std::size_t>(dstW) * static_cast<std::size_t>(dstH)) / 2u;

		if (usePbo)
		{
			glGenBuffers(2, pboY);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, pboY[0]);
			glBufferData(GL_PIXEL_PACK_BUFFER, yBytes, nullptr, GL_STREAM_READ);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, pboY[1]);
			glBufferData(GL_PIXEL_PACK_BUFFER, yBytes, nullptr, GL_STREAM_READ);
			glGenBuffers(2, pboUV);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, pboUV[0]);
			glBufferData(GL_PIXEL_PACK_BUFFER, uvBytes, nullptr, GL_STREAM_READ);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, pboUV[1]);
			glBufferData(GL_PIXEL_PACK_BUFFER, uvBytes, nullptr, GL_STREAM_READ);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		}

		checkGl("init");
	}

	~Impl()
	{
		if (dpy != EGL_NO_DISPLAY)
		{
			eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
			if (ctx != EGL_NO_CONTEXT)
				eglDestroyContext(dpy, ctx);
			if (surf != EGL_NO_SURFACE)
				eglDestroySurface(dpy, surf);
			eglTerminate(dpy);
		}

		if (fboY)
			glDeleteFramebuffers(1, &fboY);
		if (fboUV)
			glDeleteFramebuffers(1, &fboUV);
		for (int i = 0; i < 2; i++)
		{
			if (fenceY[i])
			{
				glDeleteSync(fenceY[i]);
				fenceY[i] = 0;
			}
			if (fenceUV[i])
			{
				glDeleteSync(fenceUV[i]);
				fenceUV[i] = 0;
			}
		}
		if (pboY[0] || pboY[1])
			glDeleteBuffers(2, pboY);
		if (pboUV[0] || pboUV[1])
			glDeleteBuffers(2, pboUV);
		if (texY)
			glDeleteTextures(1, &texY);
		if (texUV)
			glDeleteTextures(1, &texUV);
		if (texYOut)
			glDeleteTextures(1, &texYOut);
		if (texUVOut)
			glDeleteTextures(1, &texUVOut);
		if (progY)
			glDeleteProgram(progY);
		if (progUV)
			glDeleteProgram(progUV);
		if (vbo)
			glDeleteBuffers(1, &vbo);
		if (vao)
			glDeleteVertexArrays(1, &vao);
	}

	static bool fenceReady(GLsync s)
	{
		if (!s)
			return true;
		EGLint r = glClientWaitSync(s, 0, 0);
		return (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED);
	}

	bool tryConsume(int idx, uint8_t *outY, uint8_t *outUV)
	{
		if (!fenceY[idx] || !fenceUV[idx])
			return false;
		if (!fenceReady(fenceY[idx]) || !fenceReady(fenceUV[idx]))
			return false;

		uint64_t t0 = 0;
		if (profile)
			t0 = monoNs();

		glDeleteSync(fenceY[idx]);
		fenceY[idx] = 0;
		glDeleteSync(fenceUV[idx]);
		fenceUV[idx] = 0;

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pboY[idx]);
		void *py = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, yBytes, GL_MAP_READ_BIT);
		if (py)
		{
			std::memcpy(outY, py, yBytes);
			glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
		}

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pboUV[idx]);
		void *puv = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, uvBytes, GL_MAP_READ_BIT);
		if (puv)
		{
			std::memcpy(outUV, puv, uvBytes);
			glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
		}
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		lastOutY = outY;
		lastOutUV = outUV;
		haveLast = true;
		if (profile)
			accConsumeNs += (monoNs() - t0);
		return true;
	}

	void uploadNV12(const uint8_t *yPlane, const uint8_t *uvPlane, int strideY, int strideUV)
	{
		uint64_t t0 = 0;
		if (profile)
			t0 = monoNs();

		uint64_t ty0 = 0;
		if (profile)
			ty0 = monoNs();
		glBindTexture(GL_TEXTURE_2D, texY);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, strideY);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, srcW, srcH, GL_RED, GL_UNSIGNED_BYTE, yPlane);
		if (profile)
			accUploadYNs += (monoNs() - ty0);

		uint64_t tuv0 = 0;
		if (profile)
			tuv0 = monoNs();
		glBindTexture(GL_TEXTURE_2D, texUV);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, strideUV / 2); // two bytes per pixel in UV texture (RG)
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, srcW / 2, srcH / 2, GL_RG, GL_UNSIGNED_BYTE, uvPlane);
		if (profile)
			accUploadUVNs += (monoNs() - tuv0);

		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

		if (profileSync)
			glFinish();
		if (profile)
			accUploadNs += (monoNs() - t0);
	}

	void renderY(uint8_t *outY)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fboY);
		glViewport(0, 0, dstW, dstH);
		glUseProgram(progY);
		glUniformMatrix3fv(uMatY, 1, GL_FALSE, invY.data());
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texY);
		glUniform1i(uTexY, 0);
		glBindVertexArray(vao);
		uint64_t tDraw0 = 0;
		if (profile)
			tDraw0 = monoNs();
		glDrawArrays(GL_TRIANGLES, 0, 3);
		if (profileSync)
			glFinish();
		if (profile)
			accDrawYNs += (monoNs() - tDraw0);
		uint64_t tRead0 = 0;
		if (profile)
			tRead0 = monoNs();
		glReadPixels(0, 0, dstW, dstH, GL_RED, GL_UNSIGNED_BYTE, outY);
		if (profileSync)
			glFinish();
		if (profile)
			accReadYNs += (monoNs() - tRead0);
	}

	void renderUV(uint8_t *outUV)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fboUV);
		glViewport(0, 0, dstW / 2, dstH / 2);
		glUseProgram(progUV);
		glUniformMatrix3fv(uMatUV, 1, GL_FALSE, invUV.data());
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texUV);
		glUniform1i(uTexUV, 0);
		glBindVertexArray(vao);
		uint64_t tDraw0 = 0;
		if (profile)
			tDraw0 = monoNs();
		glDrawArrays(GL_TRIANGLES, 0, 3);
		if (profileSync)
			glFinish();
		if (profile)
			accDrawUVNs += (monoNs() - tDraw0);
		uint64_t tRead0 = 0;
		if (profile)
			tRead0 = monoNs();
		glReadPixels(0, 0, dstW / 2, dstH / 2, GL_RG, GL_UNSIGNED_BYTE, outUV);
		if (profileSync)
			glFinish();
		if (profile)
			accReadUVNs += (monoNs() - tRead0);
	}

	void transformNv12(const uint8_t *yPlane, const uint8_t *uvPlane, int strideY, int strideUV, uint8_t *outY, uint8_t *outUV)
	{
		uint64_t tXform0 = 0;
		uint64_t tTotal0 = 0;
		if (profile)
		{
			tXform0 = monoNs();
			tTotal0 = tXform0;
		}

		uploadNV12(yPlane, uvPlane, strideY, strideUV);

		if (!usePbo)
		{
			renderY(outY);
			renderUV(outUV);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
		else
		{
			// Pipeline readback using paired PBOs for Y and UV.
			const int cur = pboIndex;
			const int prev = (pboIndex + 1) & 1;
			pboIndex = prev;

			// Y draw + read kick
			glBindFramebuffer(GL_FRAMEBUFFER, fboY);
			glViewport(0, 0, dstW, dstH);
			glUseProgram(progY);
			glUniformMatrix3fv(uMatY, 1, GL_FALSE, invY.data());
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texY);
			glUniform1i(uTexY, 0);
			glBindVertexArray(vao);
			uint64_t tDraw0 = 0;
			if (profile)
				tDraw0 = monoNs();
			glDrawArrays(GL_TRIANGLES, 0, 3);
			if (profileSync)
				glFinish();
			if (profile)
				accDrawYNs += (monoNs() - tDraw0);

			glBindBuffer(GL_PIXEL_PACK_BUFFER, pboY[cur]);
			uint64_t tKick0 = 0;
			if (profile)
				tKick0 = monoNs();
			glReadPixels(0, 0, dstW, dstH, GL_RED, GL_UNSIGNED_BYTE, 0);
			if (profileSync)
				glFinish();
			if (profile)
				accReadKickYNs += (monoNs() - tKick0);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
			if (fenceY[cur])
			{
				glDeleteSync(fenceY[cur]);
				fenceY[cur] = 0;
			}
			fenceY[cur] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

			// UV draw + read kick
			glBindFramebuffer(GL_FRAMEBUFFER, fboUV);
			glViewport(0, 0, dstW / 2, dstH / 2);
			glUseProgram(progUV);
			glUniformMatrix3fv(uMatUV, 1, GL_FALSE, invUV.data());
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texUV);
			glUniform1i(uTexUV, 0);
			glBindVertexArray(vao);
			if (profile)
				tDraw0 = monoNs();
			glDrawArrays(GL_TRIANGLES, 0, 3);
			if (profileSync)
				glFinish();
			if (profile)
				accDrawUVNs += (monoNs() - tDraw0);

			glBindBuffer(GL_PIXEL_PACK_BUFFER, pboUV[cur]);
			if (profile)
				tKick0 = monoNs();
			glReadPixels(0, 0, dstW / 2, dstH / 2, GL_RG, GL_UNSIGNED_BYTE, 0);
			if (profileSync)
				glFinish();
			if (profile)
				accReadKickUVNs += (monoNs() - tKick0);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
			if (fenceUV[cur])
			{
				glDeleteSync(fenceUV[cur]);
				fenceUV[cur] = 0;
			}
			fenceUV[cur] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

			glFlush();
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			// Consume previous if ready (writes directly into the current output buffers)
			bool consumed = tryConsume(prev, outY, outUV);

			// Ensure first frame returns valid data
			if (!haveLast)
			{
				glFinish();
				consumed = tryConsume(cur, outY, outUV);
			}

			// If no new frame was consumed, reuse the last completed output.
			if (!consumed && haveLast && lastOutY && lastOutUV)
			{
				uint64_t tReuse0 = 0;
				if (profile)
					tReuse0 = monoNs();
				std::memcpy(outY, lastOutY, yBytes);
				std::memcpy(outUV, lastOutUV, uvBytes);
				if (profile)
					accReuseCopyNs += (monoNs() - tReuse0);
			}
		}

		if (!profile)
			return;

		accXformTotalNs += (monoNs() - tXform0);
		accTotalNs += (monoNs() - tTotal0);
		profFrames++;
		uint64_t now = monoNs();
		if (now - lastPrintNs > 1000000000ull)
		{
			double denom = profFrames ? static_cast<double>(profFrames) : 1.0;
			auto toMs = [&](uint64_t ns) { return (static_cast<double>(ns) / denom) / 1e6; };
				std::cerr << "[gpuNv12] avg ms: total(render)=" << toMs(accTotalNs)
						  << " xform(upload+render)=" << toMs(accXformTotalNs)
						  << " upload=" << toMs(accUploadNs)
						  << " uploadY=" << toMs(accUploadYNs)
						  << " uploadUV=" << toMs(accUploadUVNs)
						  << " yDraw=" << toMs(accDrawYNs);
				if (usePbo)
					std::cerr << " yReadKick=" << toMs(accReadKickYNs);
				else
					std::cerr << " yRead=" << toMs(accReadYNs);
				std::cerr << " uvDraw=" << toMs(accDrawUVNs);
				if (usePbo)
					std::cerr << " uvReadKick=" << toMs(accReadKickUVNs);
				else
					std::cerr << " uvRead=" << toMs(accReadUVNs);
				if (usePbo)
					std::cerr << " consume(map+cpy)=" << toMs(accConsumeNs) << " reuseCopy=" << toMs(accReuseCopyNs);
				std::cerr << (profileSync ? " (sync)" : "")
						  << std::endl;
			profFrames = 0;
				accUploadNs = accUploadYNs = accUploadUVNs = 0;
				accDrawYNs = accReadYNs = accReadKickYNs = 0;
				accDrawUVNs = accReadUVNs = accReadKickUVNs = 0;
				accConsumeNs = 0;
				accReuseCopyNs = 0;
			accTotalNs = accXformTotalNs = 0;
			lastPrintNs = now;
		}
	}
};

GpuNv12Transform::GpuNv12Transform(int srcWidth, int srcHeight, const cv::Point2f *srcQuad, int dstWidth, int dstHeight)
{
	_impl = new Impl(srcWidth, srcHeight, srcQuad, dstWidth, dstHeight);
}

GpuNv12Transform::~GpuNv12Transform()
{
	delete _impl;
	_impl = nullptr;
}

void GpuNv12Transform::transform(const uint8_t *yPlane, const uint8_t *uvPlane, int strideY, int strideUV, uint8_t *outY, uint8_t *outUV)
{
	_impl->transformNv12(yPlane, uvPlane, strideY, strideUV, outY, outUV);
}

#else

GpuNv12Transform::GpuNv12Transform(int, int, const cv::Point2f *, int, int)
{
	throw std::runtime_error("GPU transform support not compiled in (missing EGL/GLES3 dev libs)");
}

GpuNv12Transform::~GpuNv12Transform() {}

void GpuNv12Transform::transform(const uint8_t *, const uint8_t *, int, int, uint8_t *, uint8_t *) {}

#endif
