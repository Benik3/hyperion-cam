#include "gpuNv12Transform.h"

#ifdef HAVE_GPU_TRANSFORM

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
	static constexpr uint32_t fourcc_code(char a, char b, char c, char d)
	{
		return (static_cast<uint32_t>(a)) |
			   (static_cast<uint32_t>(b) << 8) |
			   (static_cast<uint32_t>(c) << 16) |
			   (static_cast<uint32_t>(d) << 24);
	}

	// Plane import formats
	static constexpr uint32_t DRM_FORMAT_R8 = fourcc_code('R', '8', ' ', ' ');
	static constexpr uint32_t DRM_FORMAT_GR88 = fourcc_code('G', 'R', '8', '8');

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

	// 3D-LUT GLSL chunk (test mode, see HYPERCAM_LUT_TEST). Full-range BT.601
	// YUV<->RGB conversion must be LIMITED-range BT.601: the ISP emits
	// SMPTE170M NV12, and the calibration measures in the ISP's RGB-capture
	// space. Limited-range here puts the shader's LUT input domain on that
	// same full-range RGB scale; a full-range matrix would compress LUT
	// inputs by ~14% vs the calibrated domain.
	static std::string lutGlslChunk(bool trilinear, int n)
	{
		std::string s = "uniform highp sampler3D uLut;\n";
		s += "const float LUTN = " + std::to_string(n) + ".0;\n";
		s += R"(
vec3 yuv2rgb(float y, vec2 c){
  float yl = (y - 16.0/255.0) * (255.0/219.0);
  float u = (c.x - 128.0/255.0) * (255.0/224.0);
  float v = (c.y - 128.0/255.0) * (255.0/224.0);
  return clamp(vec3(yl + 1.402*v, yl - 0.344136*u - 0.714136*v, yl + 1.772*u), 0.0, 1.0);
}
)";
		if (trilinear)
		{
			s += R"(
vec3 applyLut(vec3 rgb){
  vec3 coord = rgb * ((LUTN - 1.0) / LUTN) + 0.5 / LUTN;
  return texture(uLut, coord).rgb;
}
)";
		}
		else
		{
			// Tetrahedral: 4 texel fetches; branch picks the tetrahedron by
			// sorting the fractional coords.
			s += R"(
vec3 applyLut(vec3 rgb){
  vec3 p = rgb * (LUTN - 1.0);
  ivec3 i0 = ivec3(min(floor(p), vec3(LUTN - 2.0)));
  vec3 f = p - vec3(i0);
  ivec3 i1 = i0 + 1;
  vec3 c000 = texelFetch(uLut, i0, 0).rgb;
  vec3 c111 = texelFetch(uLut, i1, 0).rgb;
  vec3 r;
  if (f.x >= f.y) {
    if (f.y >= f.z) {
      vec3 ca = texelFetch(uLut, ivec3(i1.x, i0.y, i0.z), 0).rgb;
      vec3 cb = texelFetch(uLut, ivec3(i1.x, i1.y, i0.z), 0).rgb;
      r = (1.0-f.x)*c000 + (f.x-f.y)*ca + (f.y-f.z)*cb + f.z*c111;
    } else if (f.x >= f.z) {
      vec3 ca = texelFetch(uLut, ivec3(i1.x, i0.y, i0.z), 0).rgb;
      vec3 cb = texelFetch(uLut, ivec3(i1.x, i0.y, i1.z), 0).rgb;
      r = (1.0-f.x)*c000 + (f.x-f.z)*ca + (f.z-f.y)*cb + f.y*c111;
    } else {
      vec3 ca = texelFetch(uLut, ivec3(i0.x, i0.y, i1.z), 0).rgb;
      vec3 cb = texelFetch(uLut, ivec3(i1.x, i0.y, i1.z), 0).rgb;
      r = (1.0-f.z)*c000 + (f.z-f.x)*ca + (f.x-f.y)*cb + f.y*c111;
    }
  } else {
    if (f.x >= f.z) {
      vec3 ca = texelFetch(uLut, ivec3(i0.x, i1.y, i0.z), 0).rgb;
      vec3 cb = texelFetch(uLut, ivec3(i1.x, i1.y, i0.z), 0).rgb;
      r = (1.0-f.y)*c000 + (f.y-f.x)*ca + (f.x-f.z)*cb + f.z*c111;
    } else if (f.y >= f.z) {
      vec3 ca = texelFetch(uLut, ivec3(i0.x, i1.y, i0.z), 0).rgb;
      vec3 cb = texelFetch(uLut, ivec3(i0.x, i1.y, i1.z), 0).rgb;
      r = (1.0-f.y)*c000 + (f.y-f.z)*ca + (f.z-f.x)*cb + f.x*c111;
    } else {
      vec3 ca = texelFetch(uLut, ivec3(i0.x, i0.y, i1.z), 0).rgb;
      vec3 cb = texelFetch(uLut, ivec3(i0.x, i1.y, i1.z), 0).rgb;
      r = (1.0-f.z)*c000 + (f.z-f.y)*ca + (f.y-f.x)*cb + f.x*c111;
    }
  }
  return r;
}
)";
		}
		return s;
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

	bool dmabufEnabled{false};
	bool dmabufSupported{false};
	bool dmabufWarnedOnce{false};
	PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR{nullptr};
	PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR{nullptr};
	PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_glEGLImageTargetTexture2DOES{nullptr};

	struct DmabufKey
	{
		int fd;
		int offset;
		int stride;
		int width;
		int height;
		uint32_t fourcc;

		bool operator==(const DmabufKey &o) const
		{
			return fd == o.fd && offset == o.offset && stride == o.stride && width == o.width && height == o.height && fourcc == o.fourcc;
		}
	};

	struct DmabufKeyHash
	{
		std::size_t operator()(const DmabufKey &k) const
		{
#if SIZE_MAX == 0xFFFFFFFFU
			std::size_t h = 2166136261U;
			auto mix = [&](std::size_t v)
			{
				h ^= v + 0x9e3779b9U + (h << 6) + (h >> 2);
			};
#else
			std::size_t h = 1469598103934665603ULL;
			auto mix = [&](std::size_t v)
			{
				h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
			};
#endif
			mix(static_cast<std::size_t>(k.fd));
			mix(static_cast<std::size_t>(k.offset));
			mix(static_cast<std::size_t>(k.stride));
			mix(static_cast<std::size_t>(k.width));
			mix(static_cast<std::size_t>(k.height));
			mix(static_cast<std::size_t>(k.fourcc));
			return h;
		}
	};

	std::unordered_map<DmabufKey, EGLImageKHR, DmabufKeyHash> dmabufImages;

	// Optional pipelined readback via PBOs (reduces blocking on glReadPixels).
	bool usePbo{false};
	// Synchronous single-PBO readback (default): glReadPixels into a PBO lets
	// the GPU do the tiled->linear conversion, then fence + map + memcpy.
	// Same-frame freshness, cheaper than direct glReadPixels to CPU memory.
	bool syncPbo{false};
	GLuint pboSyncY{0};
	GLuint pboSyncUV{0};
	// RGBA-packed readback (see ctor comment).
	bool packed{false};
	int packedYW{0};
	int packedUVW{0};
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

	// 3D-LUT test mode (HYPERCAM_LUT_TEST): identity LUT applied in the packed
	// fragment shaders to measure the runtime cost of the LED-calibration LUT
	// (doc/led-lut-calibration.md). Tetrahedral by default,
	// HYPERCAM_LUT_TRILINEAR=1 for the HW-filtered variant,
	// HYPERCAM_LUT_SIZE=N (default 17) for size scaling.
	bool lutEnabled{false};
	bool lutTrilinear{false};
	int lutN{17};
	GLuint texLut{0};

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

	GLint texFilter{GL_NEAREST};

	Impl(int srcWidth, int srcHeight, const cv::Point2f *srcQuad, int dstWidth, int dstHeight, bool highQuality)
		: srcW(srcWidth), srcH(srcHeight), dstW(dstWidth), dstH(dstHeight)
	{
		texFilter = highQuality ? GL_LINEAR : GL_NEAREST;
		profile = envEnabled("HYPERCAM_GL_PROFILE");
		profileSync = envEnabled("HYPERCAM_GL_PROFILE_SYNC");
		usePbo = envEnabled("HYPERCAM_NV12_PBO");
		// Measured slower than direct glReadPixels on RPi4/Mesa v3d (the driver
		// de-tiles on the CPU either way and the PBO adds an extra copy); opt-in.
		syncPbo = !usePbo && envEnabled("HYPERCAM_GL_SYNCPBO");
		// Pack 4 Y pixels (or 2 UV pairs) per RGBA8 texel: glReadPixels cost on
		// v3d is dominated by per-pixel CPU de-tiling, so 1/4 the pixels at 4x
		// the depth reads back much faster. Needs dstW % 4 == 0.
		packed = !usePbo && (dstW % 4 == 0) && !envEnabled("HYPERCAM_GL_NO_PACK");
		packedYW = packed ? dstW / 4 : dstW;
		packedUVW = packed ? dstW / 4 : dstW / 2;
		dmabufEnabled = true;
		lastPrintNs = monoNs();

		lutEnabled = envEnabled("HYPERCAM_LUT_TEST");
		lutTrilinear = envEnabled("HYPERCAM_LUT_TRILINEAR");
		if (const char *ls = std::getenv("HYPERCAM_LUT_SIZE"))
		{
			int v = std::atoi(ls);
			if (v >= 2 && v <= 64)
				lutN = v;
		}
		if (lutEnabled && !packed)
		{
			// Only the packed (production) shaders have LUT variants; the PBO
			// render branch also doesn't bind the extra texture units.
			std::cerr << "[gpuNv12] WARN: HYPERCAM_LUT_TEST requires the packed readback path; LUT disabled" << std::endl;
			lutEnabled = false;
		}
		if (lutEnabled)
			std::cerr << "[gpuNv12] LUT test enabled: " << lutN << "^3 "
					  << (lutTrilinear ? "trilinear" : "tetrahedral") << std::endl;

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
				std::cerr << "[gpuNv12] INFO: DMA-BUF import unavailable; using CPU upload" << std::endl;
		}

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

		// The projective mapping is evaluated in the vertex shader and linearly
		// interpolated as homogeneous coords (exact: each component is a linear
		// function of the destination pixel position). The fragment shader only
		// does the perspective divide + one texture fetch - much cheaper per
		// pixel than a mat3 multiply + textureSize() query.
		// uXScale: full-res output pixels per framebuffer pixel along x
		// (1 unpacked; 4 for packed Y, 2 for packed UV).
		const char *vsSrc = R"(
#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
uniform mat3 uInv;
uniform vec2 uDstSize;
uniform float uXScale;
out vec3 vSrc;
void main(){
  gl_Position = vec4(aPos, 0.0, 1.0);
  // Pixel coords this clip-space position maps to (== gl_FragCoord.xy - 0.5).
  vec2 pix = (aPos * 0.5 + 0.5) * uDstSize - 0.5;
  vSrc = uInv * vec3(pix.x * uXScale, pix.y, 1.0);
}
)";

		const char *fsYSrc = R"(
#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uInvTexSize;
in vec3 vSrc;
out vec4 outColor;
void main(){
  vec2 s = vSrc.xy / vSrc.z;
  vec2 uv = (s + vec2(0.5, 0.5)) * uInvTexSize;
  float y = texture(uTex, uv).r;
  outColor = vec4(y, 0.0, 0.0, 1.0);
}
)";

		const char *fsUVSrc = R"(
#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uInvTexSize;
in vec3 vSrc;
out vec4 outColor;
void main(){
  vec2 s = vSrc.xy / vSrc.z;
  vec2 uv = (s + vec2(0.5, 0.5)) * uInvTexSize;
  vec2 c = texture(uTex, uv).rg;
  outColor = vec4(c.r, c.g, 0.0, 1.0);
}
)";

		// Packed variants: uDx is the source-coord step per full-res output pixel
		// (column 0 of the inverse homography); vSrc is the coord of the first lane.
		const char *fsYPackedSrc = R"(
#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uInvTexSize;
uniform vec3 uDx;
in vec3 vSrc;
out vec4 outColor;
void main(){
  vec3 s = vSrc;
  vec2 t0 = (s.xy / s.z + vec2(0.5)) * uInvTexSize; s += uDx;
  vec2 t1 = (s.xy / s.z + vec2(0.5)) * uInvTexSize; s += uDx;
  vec2 t2 = (s.xy / s.z + vec2(0.5)) * uInvTexSize; s += uDx;
  vec2 t3 = (s.xy / s.z + vec2(0.5)) * uInvTexSize;
  outColor = vec4(texture(uTex, t0).r, texture(uTex, t1).r,
                  texture(uTex, t2).r, texture(uTex, t3).r);
}
)";

		const char *fsUVPackedSrc = R"(
#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uInvTexSize;
uniform vec3 uDx;
in vec3 vSrc;
out vec4 outColor;
void main(){
  vec3 s = vSrc;
  vec2 t0 = (s.xy / s.z + vec2(0.5)) * uInvTexSize; s += uDx;
  vec2 t1 = (s.xy / s.z + vec2(0.5)) * uInvTexSize;
  outColor = vec4(texture(uTex, t0).rg, texture(uTex, t1).rg);
}
)";

		// LUT variants of the packed shaders. Both planes need full RGB, so each
		// pass samples both source textures; the same normalized coordinate
		// addresses both (resolution-independent; half-texel chroma siting is
		// below NEAREST quantization).
		const std::string lutHeader = R"(#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uInvTexSize;
uniform vec3 uDx;
in vec3 vSrc;
out vec4 outColor;
)";

		const std::string fsYPackedLutSrc = lutHeader +
											"uniform sampler2D uTexC;\n" +
											lutGlslChunk(lutTrilinear, lutN) + R"(
float lumOf(vec2 t){
  vec3 rgb = applyLut(yuv2rgb(texture(uTex, t).r, texture(uTexC, t).rg));
  return dot(rgb, vec3(0.299, 0.587, 0.114)) * (219.0/255.0) + 16.0/255.0;
}
void main(){
  vec3 s = vSrc;
  vec2 t0 = (s.xy / s.z + vec2(0.5)) * uInvTexSize; s += uDx;
  vec2 t1 = (s.xy / s.z + vec2(0.5)) * uInvTexSize; s += uDx;
  vec2 t2 = (s.xy / s.z + vec2(0.5)) * uInvTexSize; s += uDx;
  vec2 t3 = (s.xy / s.z + vec2(0.5)) * uInvTexSize;
  outColor = vec4(lumOf(t0), lumOf(t1), lumOf(t2), lumOf(t3));
}
)";

		const std::string fsUVPackedLutSrc = lutHeader +
											 "uniform sampler2D uTexY2;\n" +
											 lutGlslChunk(lutTrilinear, lutN) + R"(
vec2 chromaOf(vec2 t){
  vec3 rgb = applyLut(yuv2rgb(texture(uTexY2, t).r, texture(uTex, t).rg));
  return vec2(dot(rgb, vec3(-0.168736, -0.331264, 0.5)) * (224.0/255.0) + 128.0/255.0,
              dot(rgb, vec3(0.5, -0.418688, -0.081312)) * (224.0/255.0) + 128.0/255.0);
}
void main(){
  vec3 s = vSrc;
  vec2 t0 = (s.xy / s.z + vec2(0.5)) * uInvTexSize; s += uDx;
  vec2 t1 = (s.xy / s.z + vec2(0.5)) * uInvTexSize;
  outColor = vec4(chromaOf(t0), chromaOf(t1));
}
)";

		GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
		GLuint fsY = compileShader(GL_FRAGMENT_SHADER,
								   lutEnabled ? fsYPackedLutSrc.c_str() : (packed ? fsYPackedSrc : fsYSrc));
		GLuint fsUV = compileShader(GL_FRAGMENT_SHADER,
									lutEnabled ? fsUVPackedLutSrc.c_str() : (packed ? fsUVPackedSrc : fsUVSrc));

		progY = linkProgram(vs, fsY);
		uMatY = glGetUniformLocation(progY, "uInv");
		uTexY = glGetUniformLocation(progY, "uTex");
		progUV = linkProgram(vs, fsUV);
		uMatUV = glGetUniformLocation(progUV, "uInv");
		uTexUV = glGetUniformLocation(progUV, "uTex");

		glDeleteShader(vs);
		glDeleteShader(fsY);
		glDeleteShader(fsUV);

		// All uniforms are constant for the lifetime of the transform; set once.
		glUseProgram(progY);
		glUniformMatrix3fv(uMatY, 1, GL_FALSE, invY.data());
		glUniform1i(uTexY, 0);
		glUniform2f(glGetUniformLocation(progY, "uDstSize"),
					static_cast<float>(packedYW), static_cast<float>(dstH));
		glUniform1f(glGetUniformLocation(progY, "uXScale"), packed ? 4.f : 1.f);
		glUniform2f(glGetUniformLocation(progY, "uInvTexSize"),
					1.f / static_cast<float>(srcW), 1.f / static_cast<float>(srcH));
		if (packed)
			glUniform3f(glGetUniformLocation(progY, "uDx"), invY[0], invY[1], invY[2]);
		glUseProgram(progUV);
		glUniformMatrix3fv(uMatUV, 1, GL_FALSE, invUV.data());
		glUniform1i(uTexUV, 0);
		glUniform2f(glGetUniformLocation(progUV, "uDstSize"),
					static_cast<float>(packedUVW), static_cast<float>(dstH / 2));
		glUniform1f(glGetUniformLocation(progUV, "uXScale"), packed ? 2.f : 1.f);
		glUniform2f(glGetUniformLocation(progUV, "uInvTexSize"),
					2.f / static_cast<float>(srcW), 2.f / static_cast<float>(srcH));
		if (packed)
			glUniform3f(glGetUniformLocation(progUV, "uDx"), invUV[0], invUV[1], invUV[2]);
		if (lutEnabled)
		{
			glUseProgram(progY);
			glUniform1i(glGetUniformLocation(progY, "uTexC"), 1);
			glUniform1i(glGetUniformLocation(progY, "uLut"), 2);
			glUseProgram(progUV);
			glUniform1i(glGetUniformLocation(progUV, "uTexY2"), 1);
			glUniform1i(glGetUniformLocation(progUV, "uLut"), 2);
		}
		glUseProgram(0);

		if (lutEnabled)
		{
			// Identity LUT (test mode): node i holds round(i*255/(N-1)).
			glGenTextures(1, &texLut);
			glBindTexture(GL_TEXTURE_3D, texLut);
			const GLint lutFilter = lutTrilinear ? GL_LINEAR : GL_NEAREST;
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, lutFilter);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, lutFilter);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			std::vector<uint8_t> lut(static_cast<std::size_t>(lutN) * lutN * lutN * 4u);
			std::size_t o = 0;
			for (int b = 0; b < lutN; b++)
			{
				for (int g = 0; g < lutN; g++)
				{
					for (int r = 0; r < lutN; r++)
					{
						lut[o++] = static_cast<uint8_t>((r * 255 + (lutN - 1) / 2) / (lutN - 1));
						lut[o++] = static_cast<uint8_t>((g * 255 + (lutN - 1) / 2) / (lutN - 1));
						lut[o++] = static_cast<uint8_t>((b * 255 + (lutN - 1) / 2) / (lutN - 1));
						lut[o++] = 255;
					}
				}
			}
			glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, lutN, lutN, lutN, 0, GL_RGBA, GL_UNSIGNED_BYTE, lut.data());
			glBindTexture(GL_TEXTURE_3D, 0);
		}

		// Input textures
		glGenTextures(1, &texY);
		glBindTexture(GL_TEXTURE_2D, texY);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texFilter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texFilter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, srcW, srcH, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

		glGenTextures(1, &texUV);
		glBindTexture(GL_TEXTURE_2D, texUV);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texFilter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texFilter);
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
		if (packed)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, packedYW, dstH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		else
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
		if (packed)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, packedUVW, dstH / 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		else
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, dstW / 2, dstH / 2, 0, GL_RG, GL_UNSIGNED_BYTE, nullptr);

		glGenFramebuffers(1, &fboUV);
		glBindFramebuffer(GL_FRAMEBUFFER, fboUV);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texUVOut, 0);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			throw std::runtime_error("FBO UV not complete");

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		yBytes = static_cast<std::size_t>(dstW) * static_cast<std::size_t>(dstH);
		uvBytes = (static_cast<std::size_t>(dstW) * static_cast<std::size_t>(dstH)) / 2u;

		if (syncPbo)
		{
			glGenBuffers(1, &pboSyncY);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, pboSyncY);
			glBufferData(GL_PIXEL_PACK_BUFFER, yBytes, nullptr, GL_STREAM_READ);
			glGenBuffers(1, &pboSyncUV);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, pboSyncUV);
			glBufferData(GL_PIXEL_PACK_BUFFER, uvBytes, nullptr, GL_STREAM_READ);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		}

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
		if (pboSyncY)
			glDeleteBuffers(1, &pboSyncY);
		if (pboSyncUV)
			glDeleteBuffers(1, &pboSyncUV);
		if (texLut)
			glDeleteTextures(1, &texLut);
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

	EGLImageKHR getDmabufImage(const DmabufKey &k)
	{
		auto it = dmabufImages.find(k);
		if (it != dmabufImages.end())
			return it->second;

		if (!dmabufSupported || !p_eglCreateImageKHR)
			return EGL_NO_IMAGE_KHR;

		EGLint attrs[] = {
			EGL_WIDTH, k.width,
			EGL_HEIGHT, k.height,
			EGL_LINUX_DRM_FOURCC_EXT, static_cast<EGLint>(k.fourcc),
			EGL_DMA_BUF_PLANE0_FD_EXT, static_cast<EGLint>(k.fd),
			EGL_DMA_BUF_PLANE0_OFFSET_EXT, static_cast<EGLint>(k.offset),
			EGL_DMA_BUF_PLANE0_PITCH_EXT, static_cast<EGLint>(k.stride),
			EGL_NONE};

		EGLImageKHR img = p_eglCreateImageKHR(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, attrs);
		if (img == EGL_NO_IMAGE_KHR)
		{
			if (!dmabufWarnedOnce)
			{
				dmabufWarnedOnce = true;
				std::cerr << "[gpuNv12] WARN: eglCreateImageKHR(dma-buf) failed; using CPU upload" << std::endl;
			}
			dmabufImages.emplace(k, EGL_NO_IMAGE_KHR);
			return EGL_NO_IMAGE_KHR;
		}

		dmabufImages.emplace(k, img);
		return img;
	}

	bool uploadNV12Dmabuf(int fdY, int offsetYBytes, int strideY, int fdUV, int offsetUVBytes, int strideUV)
	{
		uint64_t t0 = 0;
		if (profile)
			t0 = monoNs();

		if (!dmabufEnabled || !dmabufSupported || !p_glEGLImageTargetTexture2DOES)
		{
			if (profile)
				accUploadNs += (monoNs() - t0);
			return false;
		}

		uint64_t ty0 = 0;
		if (profile)
			ty0 = monoNs();
		DmabufKey ky{fdY, offsetYBytes, strideY, srcW, srcH, DRM_FORMAT_R8};
		EGLImageKHR imgY = getDmabufImage(ky);
		if (imgY == EGL_NO_IMAGE_KHR)
		{
			if (profile)
				accUploadYNs += (monoNs() - ty0);
			if (profile)
				accUploadNs += (monoNs() - t0);
			return false;
		}
		glBindTexture(GL_TEXTURE_2D, texY);
		p_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, reinterpret_cast<GLeglImageOES>(imgY));
		if (profile)
			accUploadYNs += (monoNs() - ty0);

		uint64_t tuv0 = 0;
		if (profile)
			tuv0 = monoNs();
		DmabufKey kuv{fdUV, offsetUVBytes, strideUV, srcW / 2, srcH / 2, DRM_FORMAT_GR88};
		EGLImageKHR imgUV = getDmabufImage(kuv);
		if (imgUV == EGL_NO_IMAGE_KHR)
		{
			if (profile)
				accUploadUVNs += (monoNs() - tuv0);
			if (profile)
				accUploadNs += (monoNs() - t0);
			return false;
		}
		glBindTexture(GL_TEXTURE_2D, texUV);
		p_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, reinterpret_cast<GLeglImageOES>(imgUV));
		if (profile)
			accUploadUVNs += (monoNs() - tuv0);

		if (profileSync)
			glFinish();
		if (profile)
			accUploadNs += (monoNs() - t0);
		return true;
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

	// Read the current framebuffer either directly into CPU memory, or via a
	// PBO (GPU performs the tiled->linear conversion) + fence + map + memcpy.
	void readbackPixels(GLuint pbo, int w, int h, GLenum fmt, size_t bytes, uint8_t *out)
	{
		if (!syncPbo)
		{
			glReadPixels(0, 0, w, h, fmt, GL_UNSIGNED_BYTE, out);
			return;
		}

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
		glReadPixels(0, 0, w, h, fmt, GL_UNSIGNED_BYTE, 0);
		GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
		if (fence)
		{
			glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 100000000ull); // 100ms cap
			glDeleteSync(fence);
		}
		void *p = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes, GL_MAP_READ_BIT);
		if (p)
		{
			std::memcpy(out, p, bytes);
			glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		}
		else
		{
			// Mapping failed; fall back to a direct read.
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
			glReadPixels(0, 0, w, h, fmt, GL_UNSIGNED_BYTE, out);
		}
	}

	void renderY(uint8_t *outY)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fboY);
		glViewport(0, 0, packedYW, dstH);
		glUseProgram(progY);
		if (lutEnabled)
		{
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, texUV);
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_3D, texLut);
		}
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texY);
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
		readbackPixels(pboSyncY, packedYW, dstH, packed ? GL_RGBA : GL_RED, yBytes, outY);
		if (profileSync)
			glFinish();
		if (profile)
			accReadYNs += (monoNs() - tRead0);
	}

	void renderUV(uint8_t *outUV)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fboUV);
		glViewport(0, 0, packedUVW, dstH / 2);
		glUseProgram(progUV);
		if (lutEnabled)
		{
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, texY);
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_3D, texLut);
		}
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texUV);
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
		readbackPixels(pboSyncUV, packedUVW, dstH / 2, packed ? GL_RGBA : GL_RG, uvBytes, outUV);
		if (profileSync)
			glFinish();
		if (profile)
			accReadUVNs += (monoNs() - tRead0);
	}

	void renderAfterUpload(uint64_t tXform0, uint64_t tTotal0, uint8_t *outY, uint8_t *outUV)
	{
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
			auto toMs = [&](uint64_t ns)
			{ return (static_cast<double>(ns) / denom) / 1e6; };
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
		renderAfterUpload(tXform0, tTotal0, outY, outUV);
	}

	bool transformNv12Dmabuf(int fdY, int offsetYBytes, int strideY, int fdUV, int offsetUVBytes, int strideUV, uint8_t *outY, uint8_t *outUV)
	{
		uint64_t tXform0 = 0;
		uint64_t tTotal0 = 0;
		if (profile)
		{
			tXform0 = monoNs();
			tTotal0 = tXform0;
		}

		if (!uploadNV12Dmabuf(fdY, offsetYBytes, strideY, fdUV, offsetUVBytes, strideUV))
			return false;

		renderAfterUpload(tXform0, tTotal0, outY, outUV);
		return true;
	}
};

GpuNv12Transform::GpuNv12Transform(int srcWidth, int srcHeight, const cv::Point2f *srcQuad, int dstWidth, int dstHeight, bool highQuality)
{
	_impl = new Impl(srcWidth, srcHeight, srcQuad, dstWidth, dstHeight, highQuality);
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

bool GpuNv12Transform::transformDmabuf(int fdY, int offsetYBytes, int strideY, int fdUV, int offsetUVBytes, int strideUV, uint8_t *outY, uint8_t *outUV)
{
	return _impl->transformNv12Dmabuf(fdY, offsetYBytes, strideY, fdUV, offsetUVBytes, strideUV, outY, outUV);
}

#else

GpuNv12Transform::GpuNv12Transform(int, int, const cv::Point2f *, int, int, bool)
{
	throw std::runtime_error("GPU transform support not compiled in (missing EGL/GLES3 dev libs)");
}

GpuNv12Transform::~GpuNv12Transform() {}

void GpuNv12Transform::transform(const uint8_t *, const uint8_t *, int, int, uint8_t *, uint8_t *) {}

bool GpuNv12Transform::transformDmabuf(int, int, int, int, int, int, uint8_t *, uint8_t *)
{
	return false;
}

#endif
