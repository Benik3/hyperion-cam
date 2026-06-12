#include "gpuRgbTransform.h"

#ifdef HAVE_GPU_TRANSFORM

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <array>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <string>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace
{
	static constexpr uint32_t fourcc_code(char a, char b, char c, char d)
	{
		return (static_cast<uint32_t>(a)) |
			   (static_cast<uint32_t>(b) << 8) |
			   (static_cast<uint32_t>(c) << 16) |
			   (static_cast<uint32_t>(d) << 24);
	}

	static constexpr uint32_t DRM_FORMAT_XRGB8888 = fourcc_code('X', 'R', '2', '4');
	static constexpr uint32_t DRM_FORMAT_XBGR8888 = fourcc_code('X', 'B', '2', '4');

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
		EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
		if (dpy == EGL_NO_DISPLAY)
			return EGL_NO_DISPLAY;

		EGLint maj = 0, min = 0;
		if (eglInitialize(dpy, &maj, &min))
			return dpy;

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

	static std::array<float, 9> mat3FromCv64(const cv::Mat &m)
	{
		std::array<float, 9> out{};
		// GLSL expects column-major layout when transpose is GL_FALSE.
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

struct GpuRgbTransform::Impl
{
	EGLDisplay dpy{EGL_NO_DISPLAY};
	EGLContext ctx{EGL_NO_CONTEXT};
	EGLSurface surf{EGL_NO_SURFACE};

	GLuint vao{0};
	GLuint vbo{0};

	GLuint prog{0};
	GLint uMat{-1};
	GLint uTex{-1};

	GLuint texIn[2]{0, 0};
	GLuint unpackPbo[2]{0, 0};
	GLsync texFence[2]{0, 0};
	int activeTex{0};
	std::size_t uploadBytes{0};

	bool dmabufEnabled{false};
	bool dmabufSupported{false};
	PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR{nullptr};
	PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR{nullptr};
	PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_glEGLImageTargetTexture2DOES{nullptr};
	std::unordered_map<int, EGLImageKHR> dmabufImages;
	bool dmabufWarnedOnce{false};

	GLuint texOut{0};
	GLuint rboOut{0};
	GLuint fbo{0};
	GLuint pbo[2]{0, 0};
	GLsync fence[2]{0, 0};
	int pboIndex{0};
	std::size_t readbackBytes{0};
	bool useRgb565{false};
	std::vector<uint8_t> lastRgb;
	bool haveLast{false};

	bool profile{false};
	bool profileSync{false};
	uint64_t profFrames{0};
	uint64_t accUploadNs{0};
	uint64_t accPackNs{0};
	uint64_t accGlUploadNs{0};
	uint64_t accDrawNs{0};
	uint64_t accReadKickNs{0};
	uint64_t accConsumeNs{0};
	uint64_t accConvertNs{0};
	uint64_t accTotalNs{0};
	uint64_t accXformTotalNs{0};
	uint64_t lastPrintNs{0};

	int srcW{0}, srcH{0};
	int dstW{0}, dstH{0};
	std::array<float, 9> inv{};

	GLint texFilter{GL_NEAREST};

	Impl(int srcWidth, int srcHeight, const cv::Point2f *srcQuad, int dstWidth, int dstHeight, bool highQuality)
		: srcW(srcWidth), srcH(srcHeight), dstW(dstWidth), dstH(dstHeight)
	{
		texFilter = highQuality ? GL_LINEAR : GL_NEAREST;
		profile = envEnabled("HYPERCAM_GL_PROFILE");
		profileSync = envEnabled("HYPERCAM_GL_PROFILE_SYNC");
		useRgb565 = envEnabled("HYPERCAM_GL_RGB565");
		dmabufEnabled = true;
		lastPrintNs = monoNs();

		dpy = getEglDisplayHeadless();
		if (dpy == EGL_NO_DISPLAY)
		{
			EGLint err = eglGetError();
			throw std::runtime_error("EGL display init failed (" + eglErrStr(err) + ")");
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

		{
			const char *eglExt = eglQueryString(dpy, EGL_EXTENSIONS);
			const char *glExt = reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS));
			bool haveEgl = (eglExt && std::string(eglExt).find("EGL_EXT_image_dma_buf_import") != std::string::npos);
			bool haveGl = (glExt && std::string(glExt).find("GL_OES_EGL_image") != std::string::npos);
			p_eglCreateImageKHR = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImageKHR"));
			p_eglDestroyImageKHR = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(eglGetProcAddress("eglDestroyImageKHR"));
			p_glEGLImageTargetTexture2DOES = reinterpret_cast<PFNGLEGLIMAGETARGETTEXTURE2DOESPROC>(eglGetProcAddress("glEGLImageTargetTexture2DOES"));
			dmabufSupported = haveEgl && haveGl && p_eglCreateImageKHR && p_eglDestroyImageKHR && p_glEGLImageTargetTexture2DOES;
			if (!dmabufSupported)
				std::cerr << "[gpuRgb] INFO: DMA-BUF import unavailable; using CPU upload" << std::endl;
		}

		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// Inverse homography for output->input sampling.
		cv::Point2f dstQuad[4] = {
			cv::Point2f(0.f, 0.f),
			cv::Point2f(static_cast<float>(dstW), 0.f),
			cv::Point2f(0.f, static_cast<float>(dstH)),
			cv::Point2f(static_cast<float>(dstW), static_cast<float>(dstH))};
		cv::Mat H = cv::getPerspectiveTransform(srcQuad, dstQuad);
		cv::Mat invCv;
		cv::invert(H, invCv);
		inv = mat3FromCv64(invCv);

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

		// Projective mapping evaluated per-vertex and interpolated as homogeneous
		// coords (exact); the fragment shader only divides + fetches.
		const char *vsSrc = R"(
#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
uniform mat3 uInv;
uniform vec2 uDstSize;
out vec3 vSrc;
void main(){
  gl_Position = vec4(aPos, 0.0, 1.0);
  vec2 pix = (aPos * 0.5 + 0.5) * uDstSize - 0.5;
  vSrc = uInv * vec3(pix, 1.0);
}
)";

		const char *fsSrc = R"(
#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uInvTexSize;
in vec3 vSrc;
out vec4 outColor;
void main(){
  vec2 s = vSrc.xy / vSrc.z;
  vec2 uv = (s + vec2(0.5, 0.5)) * uInvTexSize;
  vec3 c = texture(uTex, uv).rgb;
  outColor = vec4(c, 1.0);
}
)";

		GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
		GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
		prog = linkProgram(vs, fs);
		uMat = glGetUniformLocation(prog, "uInv");
		uTex = glGetUniformLocation(prog, "uTex");
		glDeleteShader(vs);
		glDeleteShader(fs);

		// Constant for the transform's lifetime; set once.
		glUseProgram(prog);
		glUniformMatrix3fv(uMat, 1, GL_FALSE, inv.data());
		glUniform1i(uTex, 0);
		glUniform2f(glGetUniformLocation(prog, "uDstSize"),
					static_cast<float>(dstW), static_cast<float>(dstH));
		glUniform2f(glGetUniformLocation(prog, "uInvTexSize"),
					1.f / static_cast<float>(srcW), 1.f / static_cast<float>(srcH));
		glUseProgram(0);

		// Upload as RGBA8 (4 bytes/pixel). Many GLES drivers handle this far more
		// efficiently than tightly-packed 3-byte GL_RGB uploads.
		uploadBytes = static_cast<std::size_t>(srcW) * static_cast<std::size_t>(srcH) * 4u;
		glGenTextures(2, texIn);
		for (int i = 0; i < 2; i++)
		{
			glBindTexture(GL_TEXTURE_2D, texIn[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texFilter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texFilter);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, srcW, srcH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		}

		// PBOs for async-ish uploads (avoid stalls on glTexSubImage2D)
		glGenBuffers(2, unpackPbo);
		for (int i = 0; i < 2; i++)
		{
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpackPbo[i]);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, uploadBytes, nullptr, GL_STREAM_DRAW);
		}
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		const std::size_t pixels = static_cast<std::size_t>(dstW) * static_cast<std::size_t>(dstH);
		readbackBytes = pixels * (useRgb565 ? 2u : 4u);
		lastRgb.resize(pixels * 3u);

		glGenBuffers(2, pbo);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[0]);
		glBufferData(GL_PIXEL_PACK_BUFFER, readbackBytes, nullptr, GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[1]);
		glBufferData(GL_PIXEL_PACK_BUFFER, readbackBytes, nullptr, GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		if (useRgb565)
		{
			// Render directly into a 16bpp target to reduce readback bandwidth.
			glGenRenderbuffers(1, &rboOut);
			glBindRenderbuffer(GL_RENDERBUFFER, rboOut);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB565, dstW, dstH);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rboOut);
			glBindRenderbuffer(GL_RENDERBUFFER, 0);
		}
		else
		{
			// Output texture (RGBA8 is broadly renderable; we read back as RGBA)
			glGenTextures(1, &texOut);
			glBindTexture(GL_TEXTURE_2D, texOut);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, dstW, dstH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);
		}
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			throw std::runtime_error("FBO RGB not complete");
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	~Impl()
	{
		if (dmabufSupported && p_eglDestroyImageKHR)
		{
			for (auto &kv : dmabufImages)
			{
				if (kv.second != EGL_NO_IMAGE_KHR)
					p_eglDestroyImageKHR(dpy, kv.second);
			}
			dmabufImages.clear();
		}

		if (dpy != EGL_NO_DISPLAY)
		{
			eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
			if (ctx != EGL_NO_CONTEXT)
				eglDestroyContext(dpy, ctx);
			if (surf != EGL_NO_SURFACE)
				eglDestroySurface(dpy, surf);
			eglTerminate(dpy);
		}
		if (fbo)
			glDeleteFramebuffers(1, &fbo);
		if (rboOut)
			glDeleteRenderbuffers(1, &rboOut);
		for (int i = 0; i < 2; i++)
		{
			if (fence[i])
			{
				glDeleteSync(fence[i]);
				fence[i] = 0;
			}
		}
		if (pbo[0] || pbo[1])
			glDeleteBuffers(2, pbo);
		for (int i = 0; i < 2; i++)
		{
			if (texFence[i])
			{
				glDeleteSync(texFence[i]);
				texFence[i] = 0;
			}
		}
		if (unpackPbo[0] || unpackPbo[1])
			glDeleteBuffers(2, unpackPbo);
		if (texIn[0] || texIn[1])
			glDeleteTextures(2, texIn);
		if (texOut)
			glDeleteTextures(1, &texOut);
		if (prog)
			glDeleteProgram(prog);
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

	EGLImageKHR getDmabufImage(int fd, int offsetBytes, int strideBytes, uint32_t fourcc)
	{
		auto it = dmabufImages.find(fd);
		if (it != dmabufImages.end())
			return it->second;

		if (!dmabufSupported || !p_eglCreateImageKHR)
			return EGL_NO_IMAGE_KHR;

		EGLint attrs[] = {
			EGL_WIDTH, srcW,
			EGL_HEIGHT, srcH,
			EGL_LINUX_DRM_FOURCC_EXT, static_cast<EGLint>(fourcc),
			EGL_DMA_BUF_PLANE0_FD_EXT, static_cast<EGLint>(fd),
			EGL_DMA_BUF_PLANE0_OFFSET_EXT, static_cast<EGLint>(offsetBytes),
			EGL_DMA_BUF_PLANE0_PITCH_EXT, static_cast<EGLint>(strideBytes),
			EGL_NONE};

		EGLImageKHR img = p_eglCreateImageKHR(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, attrs);
		if (img == EGL_NO_IMAGE_KHR)
		{
			if (!dmabufWarnedOnce)
			{
				dmabufWarnedOnce = true;
				std::cerr << "[gpuRgb] WARN: eglCreateImageKHR(dma-buf) failed; falling back to CPU upload" << std::endl;
			}
			dmabufImages.emplace(fd, EGL_NO_IMAGE_KHR);
			return EGL_NO_IMAGE_KHR;
		}

		dmabufImages.emplace(fd, img);
		return img;
	}

	void uploadDmabuf(int fd, int offsetBytes, int strideBytes, GpuRgbTransform::InputFormat fmt)
	{
		uint64_t t0 = 0;
		if (profile)
			t0 = monoNs();

		if (!dmabufEnabled || !dmabufSupported)
		{
			if (profile)
				accUploadNs += (monoNs() - t0);
			return;
		}
		if (fmt != GpuRgbTransform::InputFormat::Xbgr8888 && fmt != GpuRgbTransform::InputFormat::Xrgb8888)
		{
			if (profile)
				accUploadNs += (monoNs() - t0);
			return;
		}

		const int candidate = activeTex ^ 1;
		if (!fenceReady(texFence[candidate]))
		{
			if (profile)
				accUploadNs += (monoNs() - t0);
			return;
		}
		if (texFence[candidate])
		{
			glDeleteSync(texFence[candidate]);
			texFence[candidate] = 0;
		}

		uint32_t fourcc = (fmt == GpuRgbTransform::InputFormat::Xrgb8888) ? DRM_FORMAT_XRGB8888 : DRM_FORMAT_XBGR8888;
		EGLImageKHR img = getDmabufImage(fd, offsetBytes, strideBytes, fourcc);
		if (img == EGL_NO_IMAGE_KHR)
		{
			if (profile)
				accUploadNs += (monoNs() - t0);
			return;
		}

		glBindTexture(GL_TEXTURE_2D, texIn[candidate]);
		p_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, reinterpret_cast<GLeglImageOES>(img));
		activeTex = candidate;

		if (profile)
			accUploadNs += (monoNs() - t0);
	}

	void upload(const uint8_t *inRgb, int strideBytes)
	{
		upload(inRgb, strideBytes, GpuRgbTransform::InputFormat::Packed3);
	}

	void upload(const uint8_t *inPixels, int strideBytes, GpuRgbTransform::InputFormat fmt)
	{
		uint64_t t0 = 0;
		if (profile)
			t0 = monoNs();

		// Choose the other texture if it's not currently in-flight.
		const int candidate = activeTex ^ 1;
		if (!fenceReady(texFence[candidate]))
		{
			// Non-blocking: reuse previous texture content (effectively repeat a frame)
			if (profile)
				accUploadNs += (monoNs() - t0);
			return;
		}
		if (texFence[candidate])
		{
			glDeleteSync(texFence[candidate]);
			texFence[candidate] = 0;
		}

		// Orphan + map PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpackPbo[candidate]);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, uploadBytes, nullptr, GL_STREAM_DRAW);
		uint8_t *dst = static_cast<uint8_t *>(glMapBufferRange(
			GL_PIXEL_UNPACK_BUFFER, 0, uploadBytes,
			GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
		if (!dst)
		{
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			throw std::runtime_error("Failed to map upload PBO");
		}

		// Fill a tightly packed RGBA buffer in the upload PBO.
		// - Packed3: pack + alpha=255
		// - Xbgr8888 (RGBX): memcpy rows (X ignored)
		// - Xrgb8888 (BGRX): swap R/B while copying (X ignored)
		uint64_t tPack0 = 0;
		if (profile)
			tPack0 = monoNs();
		const int tightStrideOut = srcW * 4;
		if (fmt == GpuRgbTransform::InputFormat::Xbgr8888)
		{
			// RGBX -> RGBA (same RGB order, alpha ignored)
			for (int y = 0; y < srcH; y++)
			{
				const uint8_t *srcRow = inPixels + static_cast<std::size_t>(y) * static_cast<std::size_t>(strideBytes);
				uint8_t *dstRow = dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(tightStrideOut);
				std::memcpy(dstRow, srcRow, static_cast<std::size_t>(tightStrideOut));
			}
		}
		else if (fmt == GpuRgbTransform::InputFormat::Xrgb8888)
		{
			// BGRX -> RGBA (swap R/B)
			for (int y = 0; y < srcH; y++)
			{
				const uint8_t *srcRow = inPixels + static_cast<std::size_t>(y) * static_cast<std::size_t>(strideBytes);
				uint8_t *dstRow = dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(tightStrideOut);

				int x = 0;
#if defined(__aarch64__)
				const bool neonAnyStride = envEnabled("HYPERCAM_GL_PACK_NEON_ANY_STRIDE");
				if (neonAnyStride || strideBytes == srcW * 4)
				{
					for (; x + 15 < srcW; x += 16)
					{
						uint8x16x4_t bgra = vld4q_u8(srcRow + x * 4);
						uint8x16x4_t rgba;
						rgba.val[0] = bgra.val[2];
						rgba.val[1] = bgra.val[1];
						rgba.val[2] = bgra.val[0];
						rgba.val[3] = bgra.val[3];
						vst4q_u8(dstRow + x * 4, rgba);
					}
				}
#endif
				for (; x < srcW; x++)
				{
					const uint8_t *s = srcRow + x * 4;
					uint8_t *d = dstRow + x * 4;
					d[0] = s[2];
					d[1] = s[1];
					d[2] = s[0];
					d[3] = s[3];
				}
			}
		}
		else
		{
			for (int y = 0; y < srcH; y++)
			{
				const uint8_t *srcRow = inPixels + static_cast<std::size_t>(y) * static_cast<std::size_t>(strideBytes);
				uint8_t *dstRow = dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(tightStrideOut);

				int x = 0;
#if defined(__aarch64__)
				const bool neonAnyStride = envEnabled("HYPERCAM_GL_PACK_NEON_ANY_STRIDE");
				if (neonAnyStride || strideBytes == srcW * 3)
				{
					// 16 pixels at a time: load 3 interleaved streams, store 4 interleaved (A=255).
					const uint8x16_t a = vdupq_n_u8(255);
					for (; x + 15 < srcW; x += 16)
					{
						uint8x16x3_t rgb = vld3q_u8(srcRow + x * 3);
						uint8x16x4_t rgba;
						rgba.val[0] = rgb.val[0];
						rgba.val[1] = rgb.val[1];
						rgba.val[2] = rgb.val[2];
						rgba.val[3] = a;
						vst4q_u8(dstRow + x * 4, rgba);
					}
				}
#endif

				for (; x < srcW; x++)
				{
					const uint8_t *s = srcRow + x * 3;
					uint8_t *d = dstRow + x * 4;
					d[0] = s[0];
					d[1] = s[1];
					d[2] = s[2];
					d[3] = 255;
				}
			}
		}
		if (profile)
			accPackNs += (monoNs() - tPack0);
		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

		// Upload from PBO into the texture
		glBindTexture(GL_TEXTURE_2D, texIn[candidate]);
		uint64_t tGl0 = 0;
		if (profile)
			tGl0 = monoNs();
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, srcW, srcH, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		if (profile)
			accGlUploadNs += (monoNs() - tGl0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		activeTex = candidate;

		if (profile)
			accUploadNs += (monoNs() - t0);
	}

	void tryConsumePbo(int idx)
	{
		if (!fence[idx])
			return;

		EGLint r = glClientWaitSync(fence[idx], 0, 0);
		if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED)
			return;

		glDeleteSync(fence[idx]);
		fence[idx] = 0;

		uint64_t t0 = 0;
		if (profile)
			t0 = monoNs();

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[idx]);
		void *ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, readbackBytes, GL_MAP_READ_BIT);
		if (ptr)
		{
			uint64_t tConv0 = 0;
			if (profile)
				tConv0 = monoNs();
			uint8_t *dst = lastRgb.data();
			const std::size_t n = static_cast<std::size_t>(dstW) * static_cast<std::size_t>(dstH);

			if (useRgb565)
			{
				// RGB565 -> RGB888
				const uint16_t *src16 = static_cast<const uint16_t *>(ptr);
				std::size_t i = 0;
#if defined(__aarch64__)
				const uint16x8_t mask5 = vdupq_n_u16(0x1Fu);
				const uint16x8_t mask6 = vdupq_n_u16(0x3Fu);
				for (; i + 15 < n; i += 16)
				{
					const uint16x8_t pix0 = vld1q_u16(src16 + i);
					const uint16x8_t pix1 = vld1q_u16(src16 + i + 8);

					const uint16x8_t r5_0 = vshrq_n_u16(pix0, 11);
					const uint16x8_t r5_1 = vshrq_n_u16(pix1, 11);
					const uint16x8_t g6_0 = vandq_u16(vshrq_n_u16(pix0, 5), mask6);
					const uint16x8_t g6_1 = vandq_u16(vshrq_n_u16(pix1, 5), mask6);
					const uint16x8_t b5_0 = vandq_u16(pix0, mask5);
					const uint16x8_t b5_1 = vandq_u16(pix1, mask5);

					const uint16x8_t r8_0 = vorrq_u16(vshlq_n_u16(r5_0, 3), vshrq_n_u16(r5_0, 2));
					const uint16x8_t r8_1 = vorrq_u16(vshlq_n_u16(r5_1, 3), vshrq_n_u16(r5_1, 2));
					const uint16x8_t g8_0 = vorrq_u16(vshlq_n_u16(g6_0, 2), vshrq_n_u16(g6_0, 4));
					const uint16x8_t g8_1 = vorrq_u16(vshlq_n_u16(g6_1, 2), vshrq_n_u16(g6_1, 4));
					const uint16x8_t b8_0 = vorrq_u16(vshlq_n_u16(b5_0, 3), vshrq_n_u16(b5_0, 2));
					const uint16x8_t b8_1 = vorrq_u16(vshlq_n_u16(b5_1, 3), vshrq_n_u16(b5_1, 2));

					uint8x16x3_t rgb;
					rgb.val[0] = vcombine_u8(vmovn_u16(r8_0), vmovn_u16(r8_1));
					rgb.val[1] = vcombine_u8(vmovn_u16(g8_0), vmovn_u16(g8_1));
					rgb.val[2] = vcombine_u8(vmovn_u16(b8_0), vmovn_u16(b8_1));
					vst3q_u8(dst + i * 3, rgb);
				}
#endif
				for (; i < n; i++)
				{
					const uint16_t p = src16[i];
					const uint8_t r5 = static_cast<uint8_t>((p >> 11) & 0x1F);
					const uint8_t g6 = static_cast<uint8_t>((p >> 5) & 0x3F);
					const uint8_t b5 = static_cast<uint8_t>(p & 0x1F);
					dst[i * 3 + 0] = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
					dst[i * 3 + 1] = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
					dst[i * 3 + 2] = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
				}
			}
			else
			{
				// RGBA -> RGB directly from mapped PBO memory
				const uint8_t *src = static_cast<const uint8_t *>(ptr);
				std::size_t i = 0;
#if defined(__aarch64__)
				for (; i + 15 < n; i += 16)
				{
					uint8x16x4_t rgba = vld4q_u8(src + i * 4);
					uint8x16x3_t rgb;
					rgb.val[0] = rgba.val[0];
					rgb.val[1] = rgba.val[1];
					rgb.val[2] = rgba.val[2];
					vst3q_u8(dst + i * 3, rgb);
				}
#endif
				for (; i < n; i++)
				{
					const std::size_t si = i * 4;
					const std::size_t di = i * 3;
					dst[di + 0] = src[si + 0];
					dst[di + 1] = src[si + 1];
					dst[di + 2] = src[si + 2];
				}
			}

			glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
			if (profile)
				accConvertNs += (monoNs() - tConv0);

			haveLast = true;
		}
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		if (profile)
			accConsumeNs += (monoNs() - t0);
	}

	void render(uint8_t *outRgb)
	{
		uint64_t tTotal0 = 0;
		uint64_t t0 = 0;
		if (profile)
			tTotal0 = monoNs();

		// Pipeline the readback:
		// - Kick off async read into current PBO
		// - Try to consume previous PBO if ready
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glViewport(0, 0, dstW, dstH);
		glUseProgram(prog);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texIn[activeTex]);
		glBindVertexArray(vao);
		if (profile)
			t0 = monoNs();
		glDrawArrays(GL_TRIANGLES, 0, 3);
		// Mark the currently used input texture as in-flight
		if (texFence[activeTex])
		{
			glDeleteSync(texFence[activeTex]);
			texFence[activeTex] = 0;
		}
		texFence[activeTex] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
		if (profileSync)
			glFinish();
		if (profile)
			accDrawNs += (monoNs() - t0);

		const int cur = pboIndex;
		const int prev = (pboIndex + 1) & 1;
		pboIndex = prev;

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[cur]);
		// Readback into PBO. Default path uses RGBA/UNSIGNED_BYTE for broad GLES compatibility.
		if (profile)
			t0 = monoNs();
		if (useRgb565)
			glReadPixels(0, 0, dstW, dstH, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, 0);
		else
			glReadPixels(0, 0, dstW, dstH, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		if (profileSync)
			glFinish();
		if (profile)
			accReadKickNs += (monoNs() - t0);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		if (fence[cur])
		{
			glDeleteSync(fence[cur]);
			fence[cur] = 0;
		}
		fence[cur] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
		glFlush();

		tryConsumePbo(prev);

		// Ensure the very first frame returns something non-black.
		if (!haveLast)
		{
			glFinish();
			tryConsumePbo(cur);
		}

		if (haveLast)
			std::memcpy(outRgb, lastRgb.data(), lastRgb.size());
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		if (profile)
		{
			accTotalNs += (monoNs() - tTotal0);
			profFrames++;
			uint64_t now = monoNs();
			if (now - lastPrintNs > 1000000000ull)
			{
				double denom = profFrames ? static_cast<double>(profFrames) : 1.0;
				auto toMs = [&](uint64_t ns)
				{ return (static_cast<double>(ns) / denom) / 1e6; };
				std::cerr << "[gpuRgb] avg ms: total(render)=" << toMs(accTotalNs)
						  << " xform(upload+render)=" << toMs(accXformTotalNs)
						  << " upload=" << toMs(accUploadNs)
						  << " pack=" << toMs(accPackNs)
						  << " texUpload=" << toMs(accGlUploadNs)
						  << " draw=" << toMs(accDrawNs)
						  << " readKick=" << toMs(accReadKickNs)
						  << " consume(map+convert)=" << toMs(accConsumeNs)
						  << " rgba->rgb=" << toMs(accConvertNs)
						  << (useRgb565 ? " (rgb565)" : "")
						  << (profileSync ? " (sync)" : "")
						  << std::endl;
				profFrames = 0;
				accUploadNs = accPackNs = accGlUploadNs = accDrawNs = accReadKickNs = accConsumeNs = accConvertNs = accTotalNs = accXformTotalNs = 0;
				lastPrintNs = now;
			}
		}
	}
};

GpuRgbTransform::GpuRgbTransform(int srcWidth, int srcHeight, const cv::Point2f *srcQuad, int dstWidth, int dstHeight, bool highQuality)
{
	_impl = new Impl(srcWidth, srcHeight, srcQuad, dstWidth, dstHeight, highQuality);
}

GpuRgbTransform::~GpuRgbTransform()
{
	delete _impl;
	_impl = nullptr;
}

void GpuRgbTransform::transform(const uint8_t *inRgb, int strideBytes, uint8_t *outRgb)
{
	uint64_t t0 = 0;
	if (_impl->profile)
		t0 = monoNs();
	_impl->upload(inRgb, strideBytes, InputFormat::Packed3);
	_impl->render(outRgb);
	if (_impl->profile)
		_impl->accXformTotalNs += (monoNs() - t0);
}

void GpuRgbTransform::transform(const uint8_t *inPixels, int strideBytes, InputFormat fmt, uint8_t *outRgb)
{
	uint64_t t0 = 0;
	if (_impl->profile)
		t0 = monoNs();
	_impl->upload(inPixels, strideBytes, fmt);
	_impl->render(outRgb);
	if (_impl->profile)
		_impl->accXformTotalNs += (monoNs() - t0);
}

void GpuRgbTransform::transformDmabuf(int fd, int offsetBytes, int strideBytes, InputFormat fmt, uint8_t *outRgb)
{
	uint64_t t0 = 0;
	if (_impl->profile)
		t0 = monoNs();
	_impl->uploadDmabuf(fd, offsetBytes, strideBytes, fmt);
	_impl->render(outRgb);
	if (_impl->profile)
		_impl->accXformTotalNs += (monoNs() - t0);
}

#else

GpuRgbTransform::GpuRgbTransform(int, int, const cv::Point2f *, int, int, bool)
{
	throw std::runtime_error("GPU transform support not compiled in (missing EGL/GLES dev libs)");
}

GpuRgbTransform::~GpuRgbTransform() {}

void GpuRgbTransform::transform(const uint8_t *, int, uint8_t *) {}

#endif
