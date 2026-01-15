#include <iostream>
#include <pthread.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <csignal>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <memory>
#include <boost/program_options.hpp>
#include <libcamera/libcamera.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "event_loop.h"
#include "FlatBufferConnection.h"
#include "hyperhdr_request_generated.h"
#include "opencvTransform.h"

#ifdef HAVE_GPU_TRANSFORM
#include "gpuNv12Transform.h"
#include "gpuRgbTransform.h"
#endif

using namespace libcamera;
namespace po = boost::program_options;

static std::shared_ptr<Camera> _camera;
bool _useNv12 = false;
bool _useGpu = false;
bool _cpuOnly = false;

static int _strideY = 0;
static int _strideUV = 0;
static int _strideBgr = 0;
static int _bppBgr = 3;
static bool _swapRb4 = false;

struct Nv12RingBuffer
{
	std::vector<std::vector<uint8_t>> y;
	std::vector<std::vector<uint8_t>> uv;
	std::size_t index{0};
	int width{0};
	int height{0};

	void reset(int w, int h, std::size_t ringSize = 3)
	{
		width = w;
		height = h;
		index = 0;
		y.assign(ringSize, std::vector<uint8_t>(static_cast<std::size_t>(w) * static_cast<std::size_t>(h)));
		uv.assign(ringSize, std::vector<uint8_t>(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) / 2u));
	}

	struct View
	{
		uint8_t *y;
		uint8_t *uv;
		int strideY;
		int strideUV;
		int width;
		int height;
	};

	View next()
	{
		std::size_t idx = index++ % y.size();
		return View{y[idx].data(), uv[idx].data(), width, width, width, height};
	}
};

static Nv12RingBuffer _nv12Out;

struct RgbRingBuffer
{
	std::vector<std::vector<uint8_t>> rgb;
	std::size_t index{0};
	int width{0};
	int height{0};

	void reset(int w, int h, std::size_t ringSize = 3)
	{
		width = w;
		height = h;
		index = 0;
		rgb.assign(ringSize, std::vector<uint8_t>(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u));
	}

	struct View
	{
		uint8_t *rgb;
		int width;
		int height;
	};

	View next()
	{
		std::size_t idx = index++ % rgb.size();
		return View{rgb[idx].data(), width, height};
	}
};

static RgbRingBuffer _rgbOut;

std::map<libcamera::FrameBuffer *, std::vector<libcamera::Span<uint8_t>>> _mappedBuffers;
static EventLoop _loop;
static EventLoop _loopFb;
static EventLoop _loopTr;
FlatBufferConnection _flatbuf;
OpencvTransform *_transformation;

static constexpr const char *HYPERCAM_VERSION = "1.0.0";

#ifdef HAVE_GPU_TRANSFORM
GpuNv12Transform *_gpuNv12 = nullptr;
GpuRgbTransform *_gpuRgb = nullptr;
#endif
cv::Point2f _imgQuad[4];
int _imgWidth = 0;
int _imgHeight = 0;
int _imgWidthToSend;
int _imgHeightToSend;

// Output crop buffer (used when output-crop is enabled)
static std::vector<uint8_t> _outputCropBuffer;

int _width;
int _height;
int32_t _shutter;
float _gain;
std::string _wbgainsStr;
float _rGain;
float _bGain;
float _focus = -1.0f;
float _saturation = 1.0f;
float _contrast = 1.0f;
float _gamma = 0.0f;
bool _hflip = false;
bool _vflip = false;
int _rotate = 0;
std::string _origin;
std::string _conn;
int _port;
int _priority;
std::string _transformStr;
std::string _imgOutPath;
std::string _imgOutCropHelpPath;
std::string _cropStr;
std::string _outputCropStr;
bool _transformEn = false;
bool _cropEn = false;
int _cropX = 0;
int _cropY = 0;
int _cropWidth = 0;
int _cropHeight = 0;
bool _outputCropEn = false;
int _outputCropX = 0;
int _outputCropY = 0;
int _outputCropWidth = 0;
int _outputCropHeight = 0;
bool _verbose = false;
bool _help = false;
bool _version = false;

// For --imgout-crophelp: store the auto-crop rectangle to draw on full sensor image
static bool _imgOutCropHelpDrawRect = false;
static int _imgOutCropHelpRectX = 0;
static int _imgOutCropHelpRectY = 0;
static int _imgOutCropHelpRectWidth = 0;
static int _imgOutCropHelpRectHeight = 0;

static bool _autoFocusOnce = false;
static std::atomic<int> _autoFocusPhase{0}; // 0=disabled, 1=running, 2=locked
static std::atomic<bool> _autoFocusTriggerSent{false};
static std::atomic<float> _autoFocusLockedPos{-1.0f};

static std::atomic<bool> _imgOutDone{false};

// FPS control
static float _targetFps = 0.0f; // 0 = disabled (no FPS limit)

static bool _autoWb = false;
static bool _autoGain = false;
static bool _autoCalib = false;
static bool _autoSat = false;
static std::string _autoSatTargetSpec;
static int _autoSatTargetR = 230;
static int _autoSatTargetG = 50;
static int _autoSatTargetB = 50;
static int _autoCalibMaxIters = 8;
static int _autoCalibWarmupFrames = 2;
static int _autoCalibDelayMs = 500;
static float _autoCalibTolWb = 2.0f;
static float _autoCalibTolGrey = 3.0f;
static int _autoGainSamples = 8;
static float _autoGainRange = 4.0f;

static int _autoSatMaxIters = 24;
static int _autoSatPhase = 0; // 0=init, 1=search-up, 2=bisect
static float _autoSatLo = 0.0f;
static float _autoSatHi = 0.0f;

static std::atomic<bool> _autoCalibBusy{false};
static std::atomic<int> _autoCalibFrameCount{0};
static std::atomic<int> _autoCalibIterCount{0};
static std::atomic<int64_t> _autoCalibNextAllowedNs{0};
static std::atomic<int> _autoCalibStage{0}; // 0=init, 1=wb, 2=gain
static std::mutex _autoCalibMutex;
static float _autoLastR = 0.0f;
static float _autoLastG = 0.0f;
static float _autoLastB = 0.0f;

// Auto-gain curve fitting state
static int _autoGainCurvePhase = 0; // 0=off, 1=sampling, 2=confirm
static std::vector<float> _autoGainCurveGains;
static std::vector<float> _autoGainCurveYs;
static size_t _autoGainCurveNextIdx = 0;

static inline bool autoCalibEnabled()
{
	return _autoCalib || _autoWb || _autoGain || _autoSat;
}

static inline float clampf(float v, float lo, float hi)
{
	return std::max(lo, std::min(hi, v));
}

static inline bool parseRgbTriplet(std::string spec, int &outR, int &outG, int &outB)
{
	if (spec.empty())
		return false;
	for (char &ch : spec)
	{
		if (ch == ';' || ch == ':')
			ch = ',';
	}

	int r = 0, g = 0, b = 0;
	if (sscanf(spec.c_str(), " %d , %d , %d ", &r, &g, &b) != 3)
		return false;
	if (r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255)
		return false;
	if (r == g && g == b)
		return false;
	outR = r;
	outG = g;
	outB = b;
	return true;
}

static inline void applyFocusControls(ControlList &c)
{
	if (_autoFocusOnce)
	{
		const int phase = _autoFocusPhase.load(std::memory_order_acquire);
		if (phase == 1)
		{
			c.set(controls::AfMode, controls::AfModeEnum::AfModeAuto);
			if (!_autoFocusTriggerSent.exchange(true, std::memory_order_acq_rel))
				c.set(controls::AfTrigger, controls::AfTriggerEnum::AfTriggerStart);
			return;
		}

		if (phase == 2)
		{
			c.set(controls::AfMode, controls::AfModeEnum::AfModeManual);
			const float pos = _autoFocusLockedPos.load(std::memory_order_acquire);
			if (pos >= 0.0f)
				c.set(controls::LensPosition, pos);
			return;
		}
	}

	// Default: manual focus.
	c.set(controls::AfMode, controls::AfModeEnum::AfModeManual);
	if (_focus >= 0.0f)
		c.set(controls::LensPosition, _focus);
}

static inline void handleAutoFocusMetadata(Request *request)
{
	if (!_autoFocusOnce)
		return;
	if (_autoFocusPhase.load(std::memory_order_acquire) != 1)
		return;

	ControlList &md = request->metadata();
	auto afStateOpt = md.get(controls::AfState);
	if (!afStateOpt)
		return;

	const auto afState = static_cast<controls::AfStateEnum>(*afStateOpt);
	if (afState != controls::AfStateEnum::AfStateFocused && afState != controls::AfStateEnum::AfStateFailed)
		return;

	float pos = -1.0f;
	auto posOpt = md.get(controls::LensPosition);
	if (posOpt)
		pos = *posOpt;

	_autoFocusLockedPos.store(pos, std::memory_order_release);
	_autoFocusPhase.store(2, std::memory_order_release);

	if (pos >= 0.0f)
		std::cout << "AutoFocus done: focus=" << pos << std::endl;
	else
		std::cout << "AutoFocus done: LensPosition unavailable" << std::endl;
}

static bool fitGainLogLinear(const std::vector<float> &gains, const std::vector<float> &ys, float targetY, float &outGain)
{
	// Fit log(y) = a*log(gain) + b and solve for y=targetY.
	if (gains.size() != ys.size() || gains.empty())
		return false;
	if (targetY <= 0.0f)
		return false;

	double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
	int n = 0;
	for (size_t i = 0; i < gains.size(); i++)
	{
		const float g = gains[i];
		const float y = ys[i];
		// Filter unusable/saturated points.
		if (!(g > 0.0f) || !(y > 1.0f) || y >= 254.5f)
			continue;
		const double x = std::log(static_cast<double>(g));
		const double ly = std::log(static_cast<double>(y));
		sumX += x;
		sumY += ly;
		sumXX += x * x;
		sumXY += x * ly;
		n++;
	}
	if (n < 2)
		return false;

	const double denom = (static_cast<double>(n) * sumXX) - (sumX * sumX);
	if (std::fabs(denom) < 1e-9)
		return false;
	const double a = ((static_cast<double>(n) * sumXY) - (sumX * sumY)) / denom;
	const double b = (sumY - a * sumX) / static_cast<double>(n);
	if (!(a > 0.0))
		return false;

	const double tgt = std::log(static_cast<double>(targetY));
	const double xT = (tgt - b) / a;
	const double gT = std::exp(xT);
	if (!std::isfinite(gT) || gT <= 0.0)
		return false;
	outGain = static_cast<float>(gT);
	return true;
}

static bool envEnabled(const char *name)
{
	const char *v = std::getenv(name);
	if (!v)
		return false;
	return (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
}

static inline int64_t nowNs()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
			   std::chrono::high_resolution_clock::now().time_since_epoch())
		.count();
}

struct CpuProfileState
{
	bool enabled{false};
	uint64_t lastPrintNs{0};
	uint64_t frames{0};
	uint64_t accXformNs{0};
};

static inline void cpuProfileTick(CpuProfileState &s, const char *tag, uint64_t xformNs)
{
	if (!s.enabled)
		return;

	s.accXformNs += xformNs;
	s.frames++;

	uint64_t now = static_cast<uint64_t>(nowNs());
	if (s.lastPrintNs == 0)
		s.lastPrintNs = now;

	if (now - s.lastPrintNs > 1000000000ull)
	{
		double denom = s.frames ? static_cast<double>(s.frames) : 1.0;
		double avgMs = (static_cast<double>(s.accXformNs) / denom) / 1e6;
		std::cerr << tag << " avg ms: xform=" << avgMs << "\n";
		s.frames = 0;
		s.accXformNs = 0;
		s.lastPrintNs = now;
	}
}

static CpuProfileState _cpuRgbProf{envEnabled("HYPERCAM_CPU_PROFILE")};
static CpuProfileState _cpuNv12Prof{envEnabled("HYPERCAM_CPU_PROFILE")};

static std::atomic<int> _pendingProcess{0};
static std::atomic<int> _pendingSend{0};

// Connection error log gating: print once per disconnect, clear after successful reconnect.
static std::atomic<bool> _flatbufConnErrorPrinted{false};

// Reconnect throttling (prevents the FB loop from blocking on connect attempts every frame).
static int _flatbufConnectTimeoutMs = 250;
static int _flatbufReconnectIntervalMs = 250;
static std::atomic<int64_t> _flatbufNextConnectAttemptNs{0};

static inline bool flatbufTryConnect()
{
	const int64_t now = nowNs();
	const int64_t next = _flatbufNextConnectAttemptNs.load(std::memory_order_acquire);
	if (next != 0 && now < next)
		return false;

	if (_flatbuf.connectToServer(_origin.c_str(), _conn.c_str(), _port, _priority, !_verbose))
	{
		_flatbufConnErrorPrinted.store(false, std::memory_order_release);
		_flatbufNextConnectAttemptNs.store(0, std::memory_order_release);
		return true;
	}

	const int64_t delayNs = static_cast<int64_t>(_flatbufReconnectIntervalMs) * 1000000ll;
	_flatbufNextConnectAttemptNs.store(now + (delayNs > 0 ? delayNs : 0), std::memory_order_release);
	return false;
}

// Optional backlog limits (0 = unlimited). When set, frames are dropped early to keep latency bounded.
static int _maxPendingProcess = 0;
static int _maxPendingSend = 0;
static std::atomic<uint64_t> _droppedProcess{0};
static std::atomic<uint64_t> _droppedSend{0};

// Output crop ring buffer (used only when maxPendingSend > 0 to avoid per-frame allocations).
static RgbRingBuffer _cropOut;

static inline void imgOutExit(int code)
{
	_loop.exit(code);
	_loopFb.exit(code);
	_loopTr.exit(code);
}

static bool imgOutWriteBgr(const uint8_t *bgr, int width, int height, int strideBytes)
{
	try
	{
		cv::Mat bgrMat(height, width, CV_8UC3, const_cast<uint8_t *>(bgr), static_cast<size_t>(strideBytes));

		// For --imgout-crophelp, draw red rectangle showing the auto-crop area
		if (_imgOutCropHelpDrawRect)
		{
			cv::rectangle(bgrMat,
						  cv::Point(_imgOutCropHelpRectX, _imgOutCropHelpRectY),
						  cv::Point(_imgOutCropHelpRectX + _imgOutCropHelpRectWidth,
									_imgOutCropHelpRectY + _imgOutCropHelpRectHeight),
						  cv::Scalar(0, 0, 255), // Red color in BGR
						  4);					 // Thickness
			if (_verbose)
			{
				std::cout << "INFO: Drew crop area rectangle at (" << _imgOutCropHelpRectX << "," << _imgOutCropHelpRectY
						  << ") size " << _imgOutCropHelpRectWidth << "x" << _imgOutCropHelpRectHeight << std::endl;
			}
		}

		return cv::imwrite(_imgOutPath, bgrMat);
	}
	catch (const cv::Exception &e)
	{
		std::cout << "ERROR: Failed to write image: " << e.what() << std::endl;
		return false;
	}
}

static void autoCalibProcessBgr(const cv::Mat &bgr)
{
	const int w = bgr.cols;
	const int h = bgr.rows;
	if (w <= 0 || h <= 0)
	{
		std::cout << "ERROR: Invalid image size for auto calibration" << std::endl;
		imgOutExit(EXIT_FAILURE);
		return;
	}

	// Outline box ROI: edge strips (top/bottom/left/right), thickness = 1/8 of image.
	// This focuses calibration on the picture edges rather than the center.
	const int tX = std::max(1, w / 8);
	const int tY = std::max(1, h / 8);
	if (tX <= 0 || tY <= 0 || tX > w || tY > h)
	{
		std::cout << "ERROR: Invalid ROI thickness for auto calibration" << std::endl;
		imgOutExit(EXIT_FAILURE);
		return;
	}

	cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));
	mask(cv::Rect(0, 0, w, tY)).setTo(255);
	mask(cv::Rect(0, h - tY, w, tY)).setTo(255);
	mask(cv::Rect(0, 0, tX, h)).setTo(255);
	mask(cv::Rect(w - tX, 0, tX, h)).setTo(255);

	cv::Scalar meanBgr = cv::mean(bgr, mask);
	const float b = static_cast<float>(meanBgr[0]);
	const float g = static_cast<float>(meanBgr[1]);
	const float r = static_cast<float>(meanBgr[2]);

	{
		std::lock_guard<std::mutex> lock(_autoCalibMutex);
		_autoLastR = r;
		_autoLastG = g;
		_autoLastB = b;
	}

	int frames = _autoCalibFrameCount.fetch_add(1, std::memory_order_acq_rel) + 1;
	if (frames <= _autoCalibWarmupFrames)
	{
		if (_verbose)
			std::cout << "AutoCalib warmup frame " << frames << ": avg RGB=" << r << "," << g << "," << b << std::endl;
		_autoCalibBusy.store(false, std::memory_order_release);
		return;
	}

	const float avg = (r + g + b) / 3.0f;
	const float tolCh = std::max(0.1f, _autoCalibTolWb);
	const float tolGrey = std::max(0.1f, _autoCalibTolGrey);
	const bool wbOk = (std::fabs(r - g) <= tolCh) && (std::fabs(b - g) <= tolCh);
	int iters = _autoCalibIterCount.load(std::memory_order_acquire);

	const bool needWb = (_autoWb || _autoCalib);
	const bool needGain = (_autoGain || _autoCalib);
	const bool needSat = _autoSat;

	// Auto saturation calibration (one-shot).
	// Show a uniform target color (default RGB=230,25,25). Low saturation drives the ROI towards
	// grey; high saturation overshoots/starts clipping. We search for a saturation value that yields
	// the target ROI color.
	if (needSat)
	{
		if (needWb || needGain)
		{
			std::cout << "ERROR: --autosat must be run alone (do not combine with --autowb/--autogain/--autocalib)" << std::endl;
			imgOutExit(EXIT_FAILURE);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		if (iters >= _autoSatMaxIters)
		{
			std::cout << "AutoSat reached max iterations: saturation=" << _saturation
					  << " avg RGB=" << r << "," << g << "," << b << std::endl;
			imgOutExit(EXIT_FAILURE);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		constexpr float satMax = 32.0f;
		constexpr float satTol = 0.02f;
		constexpr float clip = 254.5f;
		constexpr float tolRgb = 3.0f;
		const float targetR = static_cast<float>(_autoSatTargetR);
		const float targetG = static_cast<float>(_autoSatTargetG);
		const float targetB = static_cast<float>(_autoSatTargetB);
		const bool clipped = (r >= clip) || (g >= clip) || (b >= clip);
		const bool rgbOk = (std::fabs(r - targetR) <= tolRgb) && (std::fabs(g - targetG) <= tolRgb) && (std::fabs(b - targetB) <= tolRgb);
		// Metric for bisection/bracketing:
		// Saturation tends to scale chroma around a neutral axis. For general targets (e.g. green),
		// use the projection of measured chroma onto target chroma direction.
		const float y = (r + g + b) / 3.0f;
		const float cr = r - y;
		const float cg = g - y;
		const float cb = b - y;
		const float ty = (targetR + targetG + targetB) / 3.0f;
		const float tcr = targetR - ty;
		const float tcg = targetG - ty;
		const float tcb = targetB - ty;
		const float metric = (cr * tcr) + (cg * tcg) + (cb * tcb);
		const float metricTarget = (tcr * tcr) + (tcg * tcg) + (tcb * tcb);
		if (metricTarget < 1e-3f)
		{
			std::cout << "ERROR: --autosat target must not be grey (R,G,B all equal)" << std::endl;
			imgOutExit(EXIT_FAILURE);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		std::lock_guard<std::mutex> lock(_autoCalibMutex);
		const float curSat = clampf(_saturation, 0.0f, satMax);
		_saturation = curSat;

		if (_autoSatPhase == 0)
		{
			_autoSatPhase = 1;
			_autoSatLo = 0.0f;
			_autoSatHi = 0.0f;
			_saturation = 0.0f;
			_autoCalibIterCount.store(0, std::memory_order_release);
			if (_verbose)
				std::cout << "AutoSat init: target RGB=" << targetR << "," << targetG << "," << targetB
						  << " set saturation=" << _saturation << " (avg RGB=" << r << "," << g << "," << b << ")" << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		if (rgbOk)
		{
			std::cout << "AutoSat done: saturation=" << _saturation << " avg RGB=" << r << "," << g << "," << b << std::endl;
			imgOutExit(EXIT_SUCCESS);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		if (_autoSatPhase == 1)
		{
			// Bracket the solution: find lo where metric < metricTarget and hi where metric >= metricTarget (or clipped).
			if (clipped || metric >= metricTarget)
			{
				if (curSat <= 1e-3f)
				{
					std::cout << "AutoSat failed: target already reached at saturation=" << curSat
							  << " (reduce exposure/gain or use a darker target)" << std::endl;
					imgOutExit(EXIT_FAILURE);
					_autoCalibBusy.store(false, std::memory_order_release);
					return;
				}

				_autoSatHi = curSat;
				// _autoSatLo should already be the last saturation below target chroma.
				if (_autoSatLo >= _autoSatHi)
					_autoSatLo = std::max(0.0f, _autoSatHi * 0.5f);
				_autoSatPhase = 2;

				_saturation = clampf((_autoSatLo + _autoSatHi) * 0.5f, 0.0f, satMax);
				_autoCalibIterCount.fetch_add(1, std::memory_order_acq_rel);
				if (_verbose)
					std::cout << "AutoSat bracket: lo=" << _autoSatLo << " hi=" << _autoSatHi
							  << " -> next saturation=" << _saturation
							  << " (metric=" << metric << " target=" << metricTarget << ")" << std::endl;
				if (_autoCalibDelayMs > 0)
					_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
				_autoCalibBusy.store(false, std::memory_order_release);
				return;
			}

			// Still below target and not clipped: step saturation upward.
			_autoSatLo = curSat;
			float next = (curSat <= 1e-3f) ? 0.25f : (curSat * 1.5f);
			next = clampf(next, 0.0f, satMax);
			if (next <= curSat + 1e-6f)
				next = clampf(curSat + 0.1f, 0.0f, satMax);

			if (next >= satMax - 1e-6f && metric < metricTarget)
			{
				std::cout << "AutoSat failed: did not reach target up to saturation=" << satMax
						  << " (avg RGB=" << r << "," << g << "," << b << ", metric=" << metric << ")" << std::endl;
				imgOutExit(EXIT_FAILURE);
				_autoCalibBusy.store(false, std::memory_order_release);
				return;
			}

			_saturation = next;
			_autoCalibIterCount.fetch_add(1, std::memory_order_acq_rel);
			if (_verbose)
				std::cout << "AutoSat step: set saturation=" << _saturation
						  << " (avg RGB=" << r << "," << g << "," << b << ", metric=" << metric << ")" << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		// Phase 2: bisect within [lo,hi]. Current measurement corresponds to current saturation.
		if (_autoSatPhase == 2)
		{
			if (clipped || metric >= metricTarget)
				_autoSatHi = curSat;
			else
				_autoSatLo = curSat;

			if ((_autoSatHi - _autoSatLo) <= satTol)
			{
				_saturation = clampf(_autoSatHi, 0.0f, satMax);
				std::cout << "AutoSat done: saturation=" << _saturation
						  << " avg RGB=" << r << "," << g << "," << b << std::endl;
				imgOutExit(EXIT_SUCCESS);
				_autoCalibBusy.store(false, std::memory_order_release);
				return;
			}

			_saturation = clampf((_autoSatLo + _autoSatHi) * 0.5f, 0.0f, satMax);
			_autoCalibIterCount.fetch_add(1, std::memory_order_acq_rel);
			if (_verbose)
				std::cout << "AutoSat bisect: lo=" << _autoSatLo << " hi=" << _autoSatHi
						  << " -> next saturation=" << _saturation
						  << " (metric=" << metric << " target=" << metricTarget << ")" << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		// Should not happen.
		_autoSatPhase = 0;
		_autoCalibBusy.store(false, std::memory_order_release);
		return;
	}
	const float targetGrey = 127.0f;
	const float gainRef = needWb ? g : avg;
	const float gainOkRef = avg;
	const bool gainOk2 = (std::fabs(gainOkRef - targetGrey) <= tolGrey);

	int stage = _autoCalibStage.load(std::memory_order_acquire);
	if (needWb && needGain)
	{
		if (stage == 0)
		{
			stage = 1;
			_autoCalibStage.store(1, std::memory_order_release);
		}
	}
	else if (needWb)
	{
		stage = 1;
		_autoCalibStage.store(1, std::memory_order_release);
	}
	else if (needGain)
	{
		stage = 2;
		_autoCalibStage.store(2, std::memory_order_release);
	}

	if (needWb && needGain)
	{
		if (stage == 2 && wbOk && gainOk2)
		{
			std::cout << "AutoCalib done: gain=" << _gain << " wbgain=" << _rGain << "," << _bGain
					  << " avg RGB=" << r << "," << g << "," << b << std::endl;
			imgOutExit(EXIT_SUCCESS);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}
	}
	else if (needWb && wbOk)
	{
		std::cout << "AutoWB done: wbgain=" << _rGain << "," << _bGain << " avg RGB=" << r << "," << g << "," << b << std::endl;
		imgOutExit(EXIT_SUCCESS);
		_autoCalibBusy.store(false, std::memory_order_release);
		return;
	}
	else if (needGain && gainOk2)
	{
		std::cout << "AutoGain done: gain=" << _gain << " avg RGB=" << r << "," << g << "," << b << std::endl;
		imgOutExit(EXIT_SUCCESS);
		_autoCalibBusy.store(false, std::memory_order_release);
		return;
	}

	// AutoCalib staging: do WB first; once WB is locked, reset iteration counter for gain stage.
	if (needWb && needGain)
	{
		if (stage == 1)
		{
			if (wbOk)
			{
				_autoCalibStage.store(2, std::memory_order_release);
				_autoCalibIterCount.store(0, std::memory_order_release);
				_autoGainCurvePhase = 0;
				_autoGainCurveGains.clear();
				_autoGainCurveYs.clear();
				_autoGainCurveNextIdx = 0;
				if (_verbose)
					std::cout << "AutoCalib WB locked: wbgain=" << _rGain << "," << _bGain
							  << " (measured avg RGB=" << r << "," << g << "," << b << ")" << std::endl;
				if (_autoCalibDelayMs > 0)
					_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
				_autoCalibBusy.store(false, std::memory_order_release);
				return;
			}

			if (iters >= _autoCalibMaxIters)
			{
				std::cout << "AutoCalib reached max iterations (WB stage): gain=" << _gain << " wbgain=" << _rGain << "," << _bGain
						  << " avg RGB=" << r << "," << g << "," << b << std::endl;
				imgOutExit(EXIT_SUCCESS);
				_autoCalibBusy.store(false, std::memory_order_release);
				return;
			}

			{
				std::lock_guard<std::mutex> lock(_autoCalibMutex);
				if (r > 1e-3f && b > 1e-3f)
				{
					_rGain = clampf(_rGain * (g / r), 0.05f, 32.0f);
					_bGain = clampf(_bGain * (g / b), 0.05f, 32.0f);
				}
			}
			_autoCalibIterCount.fetch_add(1, std::memory_order_acq_rel);
			if (_verbose)
				std::cout << "AutoWB iter " << (iters + 1) << ": set wbgain=" << _rGain << "," << _bGain
						  << " (measured avg RGB=" << r << "," << g << "," << b << ")" << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}
	}

	if (needGain)
	{
		// Gain stage max-iteration check.
		// Curve-fitting uses multiple frames for sampling; don't count those as "iterations".
		// Only enforce the cap once we've moved into the confirm phase.
		const bool enforceMaxIters = (_autoGainCurvePhase == 2);
		if (enforceMaxIters && iters >= _autoCalibMaxIters)
		{
			const char *stageName = (needWb && needGain) ? "GAIN stage" : "gain";
			std::cout << "AutoCalib reached max iterations (" << stageName << "): gain=" << _gain << " wbgain=" << _rGain << "," << _bGain
					  << " avg RGB=" << r << "," << g << "," << b << std::endl;
			imgOutExit(EXIT_SUCCESS);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}
	}

	// Gain calibration (curve-fit only).
	if (needGain)
	{
		// Initialize the sampling plan once.
		if (_autoGainCurvePhase == 0)
		{
			_autoGainCurvePhase = 1;
			_autoGainCurveGains.clear();
			_autoGainCurveYs.clear();
			_autoGainCurveNextIdx = 0;

			const float minG = clampf(_gain / std::max(1.01f, _autoGainRange), 0.05f, 32.0f);
			const float maxG = clampf(_gain * std::max(1.01f, _autoGainRange), 0.05f, 32.0f);
			const int samples = std::max(3, _autoGainSamples);
			_autoGainCurveGains.resize(static_cast<size_t>(samples));
			for (int i = 0; i < samples; i++)
			{
				const double t = (samples == 1) ? 0.0 : (static_cast<double>(i) / static_cast<double>(samples - 1));
				const double gLog = std::log(static_cast<double>(minG)) + t * (std::log(static_cast<double>(maxG)) - std::log(static_cast<double>(minG)));
				_autoGainCurveGains[static_cast<size_t>(i)] = static_cast<float>(std::exp(gLog));
			}
			// Force the first sample to be the current gain for a quick first datapoint.
			_autoGainCurveGains[0] = _gain;
			// Ensure strictly positive and within bounds.
			for (float &gg : _autoGainCurveGains)
				gg = clampf(gg, 0.05f, 32.0f);
			if (_verbose)
				std::cout << "AutoGain: sampling " << samples << " points in [" << minG << "," << maxG << "]" << std::endl;
		}

		if (_autoGainCurvePhase == 2)
		{
			// Confirmation measurement at predicted gain. Only finish when we're close enough to target.
			if (gainOk2)
			{
				std::cout << "AutoGain done: gain=" << _gain << " wbgain=" << _rGain << "," << _bGain
						  << " avg RGB=" << r << "," << g << "," << b << std::endl;
				imgOutExit(EXIT_SUCCESS);
				_autoCalibBusy.store(false, std::memory_order_release);
				return;
			}

			// Predicted gain was off; do a corrective multiplicative update and confirm again next frame.
			const float ref = gainOkRef;
			if (ref > 1e-3f)
			{
				const float ratio = clampf(targetGrey / ref, 0.5f, 2.0f);
				_gain = clampf(_gain * ratio, 0.05f, 32.0f);
			}
			_autoCalibIterCount.fetch_add(1, std::memory_order_acq_rel);
			if (_verbose)
				std::cout << "AutoGain confirm adjust: set gain=" << _gain
						  << " (measured avg=" << gainOkRef << ", target " << targetGrey << ")" << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		// Record the current measurement for the current gain.
		_autoGainCurveYs.push_back(gainRef);
		if (_verbose)
			std::cout << "AutoGain sample: gain=" << _gain << " ref=" << gainRef << " (avg RGB=" << r << "," << g << "," << b << ")" << std::endl;

		// Schedule next gain sample or fit.
		_autoGainCurveNextIdx++;
		if (_autoGainCurveNextIdx < _autoGainCurveGains.size())
		{
			_gain = _autoGainCurveGains[_autoGainCurveNextIdx];
			if (_verbose)
				std::cout << "AutoGain set next gain=" << _gain << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		// Fit using the sampled (gain,y) pairs.
		float predicted = 0.0f;
		if (!fitGainLogLinear(_autoGainCurveGains, _autoGainCurveYs, targetGrey, predicted))
		{
			std::cout << "AutoGain: fit failed" << std::endl;
			imgOutExit(EXIT_FAILURE);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}
		else
		{
			predicted = clampf(predicted, 0.05f, 32.0f);
			_gain = predicted;
			_autoGainCurvePhase = 2;
			if (_verbose)
				std::cout << "AutoGain predicted gain=" << _gain << " (target " << targetGrey << ")" << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}
	}
}

static void autoCalibFromRaw(uint8_t *img)
{
	if (_bppBgr == 3)
	{
		cv::Mat in3(_imgHeightToSend, _imgWidthToSend, CV_8UC3, img, static_cast<size_t>(_strideBgr));
		cv::Mat bgr;
		cv::cvtColor(in3, bgr, cv::COLOR_RGB2BGR);
		autoCalibProcessBgr(bgr);
	}
	else if (_bppBgr == 4)
	{
		cv::Mat in4(_imgHeightToSend, _imgWidthToSend, CV_8UC4, img, static_cast<size_t>(_strideBgr));
		cv::Mat bgr;
		int code = _swapRb4 ? cv::COLOR_BGRA2BGR : cv::COLOR_RGBA2BGR;
		cv::cvtColor(in4, bgr, code);
		autoCalibProcessBgr(bgr);
	}
	else
	{
		std::cout << "ERROR: Unsupported pixel format for auto calibration" << std::endl;
		imgOutExit(EXIT_FAILURE);
		_autoCalibBusy.store(false, std::memory_order_release);
	}
}

static void autoCalibFromTransformedCpu(uint8_t *img)
{
	if (!_transformation)
	{
		std::cout << "ERROR: Transform not initialized" << std::endl;
		imgOutExit(EXIT_FAILURE);
		_autoCalibBusy.store(false, std::memory_order_release);
		return;
	}

	std::vector<uint8_t> out(static_cast<std::size_t>(_imgWidthToSend) * static_cast<std::size_t>(_imgHeightToSend) * 3u);
	_transformation->transformInto(img, _strideBgr, out.data(), _imgWidthToSend * 3);
	cv::Mat bgr(_imgHeightToSend, _imgWidthToSend, CV_8UC3, out.data());
	autoCalibProcessBgr(bgr);
}

static void autoCalibFromTransformedGpu(uint8_t *img, int fd, int off)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)img;
	(void)fd;
	(void)off;
	std::cout << "ERROR: GPU transform not compiled in" << std::endl;
	imgOutExit(EXIT_FAILURE);
	_autoCalibBusy.store(false, std::memory_order_release);
	return;
#else
	if (!_gpuRgb)
	{
		std::cout << "ERROR: GPU transform not initialized" << std::endl;
		imgOutExit(EXIT_FAILURE);
		_autoCalibBusy.store(false, std::memory_order_release);
		return;
	}

	std::vector<uint8_t> rgb(static_cast<std::size_t>(_imgWidthToSend) * static_cast<std::size_t>(_imgHeightToSend) * 3u);
	GpuRgbTransform::InputFormat fmt = GpuRgbTransform::InputFormat::Packed3;
	if (_bppBgr == 4)
		fmt = _swapRb4 ? GpuRgbTransform::InputFormat::Xrgb8888 : GpuRgbTransform::InputFormat::Xbgr8888;

	if (_bppBgr == 4)
		_gpuRgb->transformDmabuf(fd, off, _strideBgr, fmt, rgb.data());
	else
		_gpuRgb->transform(img, _strideBgr, fmt, rgb.data());

	cv::Mat rgbMat(_imgHeightToSend, _imgWidthToSend, CV_8UC3, rgb.data());
	cv::Mat bgr;
	cv::cvtColor(rgbMat, bgr, cv::COLOR_RGB2BGR);
	autoCalibProcessBgr(bgr);
#endif
}

// Forward declaration
static std::vector<uint8_t> cropRgb(const uint8_t *src, int srcWidth, int srcHeight, int srcStride,
									int cropX, int cropY, int cropWidth, int cropHeight);

static void imgOutSaveRaw(uint8_t *img)
{
	bool ok = false;
	cv::Mat bgr;

	if (_bppBgr == 3)
	{
		// libcamera's 3-byte formats on some pipelines end up in RGB byte order.
		// OpenCV expects BGR when encoding files, so convert to ensure correct colors.
		cv::Mat in3(_imgHeightToSend, _imgWidthToSend, CV_8UC3, img, static_cast<size_t>(_strideBgr));
		cv::cvtColor(in3, bgr, cv::COLOR_RGB2BGR);
	}
	else if (_bppBgr == 4)
	{
		cv::Mat in4(_imgHeightToSend, _imgWidthToSend, CV_8UC4, img, static_cast<size_t>(_strideBgr));
		// XBGR8888 is RGBX in memory; XRGB8888 is BGRX in memory.
		int code = _swapRb4 ? cv::COLOR_BGRA2BGR : cv::COLOR_RGBA2BGR;
		cv::cvtColor(in4, bgr, code);
	}
	else
	{
		std::cout << "ERROR: Unsupported pixel format for --imgout" << std::endl;
		imgOutExit(EXIT_FAILURE);
		return;
	}

	// Apply output crop if enabled
	if (_outputCropEn)
	{
		if (_outputCropX + _outputCropWidth > bgr.cols || _outputCropY + _outputCropHeight > bgr.rows)
		{
			std::cout << "ERROR: Output crop region exceeds image dimensions" << std::endl;
			imgOutExit(EXIT_FAILURE);
			return;
		}
		cv::Rect roi(_outputCropX, _outputCropY, _outputCropWidth, _outputCropHeight);
		bgr = bgr(roi);
	}

	ok = cv::imwrite(_imgOutPath, bgr);

	if (!ok)
	{
		std::cout << "ERROR: Failed to write image to: " << _imgOutPath << std::endl;
		imgOutExit(EXIT_FAILURE);
		return;
	}

	if (_verbose)
		std::cout << "Saved image to: " << _imgOutPath << std::endl;
	imgOutExit(EXIT_SUCCESS);
}

static void imgOutSaveTransformedCpu(uint8_t *img)
{
	if (!_transformation)
	{
		std::cout << "ERROR: Transform not initialized" << std::endl;
		imgOutExit(EXIT_FAILURE);
		return;
	}

	std::vector<uint8_t> out(static_cast<std::size_t>(_imgWidthToSend) * static_cast<std::size_t>(_imgHeightToSend) * 3u);
	_transformation->transformInto(img, _strideBgr, out.data(), _imgWidthToSend * 3);

	int outputWidth = _imgWidthToSend;
	int outputHeight = _imgHeightToSend;
	const uint8_t *outputData = out.data();
	int outputStride = _imgWidthToSend * 3;

	std::vector<uint8_t> cropped;
	if (_outputCropEn)
	{
		cropped = cropRgb(out.data(), _imgWidthToSend, _imgHeightToSend, _imgWidthToSend * 3,
						  _outputCropX, _outputCropY, _outputCropWidth, _outputCropHeight);
		if (cropped.empty())
		{
			imgOutExit(EXIT_FAILURE);
			return;
		}
		outputWidth = _outputCropWidth;
		outputHeight = _outputCropHeight;
		outputData = cropped.data();
		outputStride = _outputCropWidth * 3;
	}

	bool ok = imgOutWriteBgr(outputData, outputWidth, outputHeight, outputStride);
	if (!ok)
	{
		std::cout << "ERROR: Failed to write image to: " << _imgOutPath << std::endl;
		imgOutExit(EXIT_FAILURE);
		return;
	}
	if (_verbose)
		std::cout << "Saved image to: " << _imgOutPath << std::endl;
	imgOutExit(EXIT_SUCCESS);
}

static void imgOutSaveTransformedGpu(uint8_t *img, int fd, int off)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)img;
	(void)fd;
	(void)off;
	std::cout << "ERROR: GPU transform not compiled in" << std::endl;
	imgOutExit(EXIT_FAILURE);
	return;
#else
	if (!_gpuRgb)
	{
		std::cout << "ERROR: GPU transform not initialized" << std::endl;
		imgOutExit(EXIT_FAILURE);
		return;
	}

	std::vector<uint8_t> rgb(static_cast<std::size_t>(_imgWidthToSend) * static_cast<std::size_t>(_imgHeightToSend) * 3u);
	GpuRgbTransform::InputFormat fmt = GpuRgbTransform::InputFormat::Packed3;
	if (_bppBgr == 4)
		fmt = _swapRb4 ? GpuRgbTransform::InputFormat::Xrgb8888 : GpuRgbTransform::InputFormat::Xbgr8888;

	if (_bppBgr == 4)
		_gpuRgb->transformDmabuf(fd, off, _strideBgr, fmt, rgb.data());
	else
		_gpuRgb->transform(img, _strideBgr, fmt, rgb.data());

	cv::Mat rgbMat(_imgHeightToSend, _imgWidthToSend, CV_8UC3, rgb.data());
	cv::Mat bgr;
	cv::cvtColor(rgbMat, bgr, cv::COLOR_RGB2BGR);

	// Apply output crop if enabled
	if (_outputCropEn)
	{
		if (_outputCropX + _outputCropWidth > bgr.cols || _outputCropY + _outputCropHeight > bgr.rows)
		{
			std::cout << "ERROR: Output crop region exceeds image dimensions" << std::endl;
			imgOutExit(EXIT_FAILURE);
			return;
		}
		cv::Rect roi(_outputCropX, _outputCropY, _outputCropWidth, _outputCropHeight);
		bgr = bgr(roi);
	}

	bool ok = cv::imwrite(_imgOutPath, bgr);
	if (!ok)
	{
		std::cout << "ERROR: Failed to write image to: " << _imgOutPath << std::endl;
		imgOutExit(EXIT_FAILURE);
		return;
	}
	if (_verbose)
		std::cout << "Saved image to: " << _imgOutPath << std::endl;
	imgOutExit(EXIT_SUCCESS);
#endif
}

void *flatLoop(void *arg)
{
	_loopFb.exec();
	return 0;
}

// Crop RGB24 image data
static std::vector<uint8_t> cropRgb(const uint8_t *src, int srcWidth, int srcHeight, int srcStride,
									int cropX, int cropY, int cropWidth, int cropHeight)
{
	// Validate crop region
	if (cropX < 0 || cropY < 0 || cropWidth <= 0 || cropHeight <= 0)
	{
		std::cout << "ERROR: Invalid crop parameters: x=" << cropX << ", y=" << cropY
				  << ", width=" << cropWidth << ", height=" << cropHeight << std::endl;
		return std::vector<uint8_t>();
	}

	if (cropX + cropWidth > srcWidth || cropY + cropHeight > srcHeight)
	{
		std::cout << "ERROR: Crop region (" << cropX << "," << cropY << "," << cropWidth << "," << cropHeight
				  << ") exceeds source dimensions (" << srcWidth << "x" << srcHeight << ")" << std::endl;
		return std::vector<uint8_t>();
	}

	std::vector<uint8_t> result(static_cast<size_t>(cropWidth) * static_cast<size_t>(cropHeight) * 3u);

	for (int y = 0; y < cropHeight; y++)
	{
		const uint8_t *srcRow = src + ((cropY + y) * srcStride) + (cropX * 3);
		uint8_t *dstRow = result.data() + (y * cropWidth * 3);
		std::memcpy(dstRow, srcRow, static_cast<size_t>(cropWidth) * 3u);
	}

	return result;
}

static bool cropRgbInto(const uint8_t *src, int srcWidth, int srcHeight, int srcStride,
						int cropX, int cropY, int cropWidth, int cropHeight,
						uint8_t *dst, int dstStrideBytes)
{
	if (!dst)
		return false;
	if (dstStrideBytes <= 0)
		dstStrideBytes = cropWidth * 3;
	if (cropX < 0 || cropY < 0 || cropWidth <= 0 || cropHeight <= 0)
		return false;
	if (cropX + cropWidth > srcWidth || cropY + cropHeight > srcHeight)
		return false;

	for (int y = 0; y < cropHeight; y++)
	{
		const uint8_t *srcRow = src + ((cropY + y) * srcStride) + (cropX * 3);
		uint8_t *dstRow = dst + (y * dstStrideBytes);
		std::memcpy(dstRow, srcRow, static_cast<size_t>(cropWidth) * 3u);
	}
	return true;
}

static void flatBufSend(const uint8_t *img)
{
	const int64_t startNs = nowNs();
	if (flatbufTryConnect())
	{
		if (_flatbuf.setImage(img, _imgWidthToSend, _imgHeightToSend) < 0)
		{
			std::cout << "ERROR: Flatbuffer write timeout\n";
		}
	}
	else
	{
		bool expected = false;
		if (_flatbufConnErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
			std::cout << "ERROR: Flatbuffer connection timeout\n";
	}

	int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame sending was too slow: " << elapsedMicros << " us\n";
	}
}

static void flatBufSendNv12(const uint8_t *y, const uint8_t *uv, int strideY, int strideUV, int width, int height)
{
	const int64_t startNs = nowNs();
	if (flatbufTryConnect())
	{
		if (_flatbuf.setNV12Image(y, uv, width, height, strideY, strideUV) < 0)
		{
			std::cout << "ERROR: Flatbuffer write timeout\n";
		}
	}
	else
	{
		bool expected = false;
		if (_flatbufConnErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
			std::cout << "ERROR: Flatbuffer connection timeout\n";
	}

	int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame sending was too slow: " << elapsedMicros << " us\n";
	}
}

static void flatBufSendRequest(flatbuffers::DetachedBuffer &&req)
{
	const int64_t startNs = nowNs();
	if (flatbufTryConnect())
	{
		if (_flatbuf.sendPrebuiltRequest(req.data(), static_cast<uint32_t>(req.size())) < 0)
		{
			std::cout << "ERROR: Flatbuffer write timeout\n";
		}
	}
	else
	{
		bool expected = false;
		if (_flatbufConnErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
			std::cout << "ERROR: Flatbuffer connection timeout\n";
	}

	int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame sending was too slow: " << elapsedMicros << " us\n";
	}
}

static flatbuffers::DetachedBuffer buildRawImageRequestUninitialized(uint16_t width, uint16_t height, uint8_t **outRgb)
{
	const size_t imgBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
	flatbuffers::FlatBufferBuilder builder(imgBytes + 1024u);

	auto imgVec = builder.CreateUninitializedVector(imgBytes, outRgb);
	auto rawImg = hyperhdrnet::CreateRawImage(builder, imgVec, width, height);
	auto imageReq = hyperhdrnet::CreateImage(builder, hyperhdrnet::ImageType_RawImage, rawImg.Union(), -1);
	auto req = hyperhdrnet::CreateRequest(builder, hyperhdrnet::Command_Image, imageReq.Union());
	builder.Finish(req);

	return builder.Release();
}

static flatbuffers::DetachedBuffer buildNV12ImageRequestUninitialized(uint16_t width, uint16_t height, uint16_t strideY, uint16_t strideUV,
																	  uint8_t **outY, uint8_t **outUV)
{
	const size_t yBytes = static_cast<size_t>(strideY) * static_cast<size_t>(height);
	const size_t uvBytes = static_cast<size_t>(strideUV) * (static_cast<size_t>(height) / 2u);
	flatbuffers::FlatBufferBuilder builder(yBytes + uvBytes + 1024u);

	auto yVec = builder.CreateUninitializedVector(yBytes, outY);
	auto uvVec = builder.CreateUninitializedVector(uvBytes, outUV);
	auto nv12Img = hyperhdrnet::CreateNV12Image(builder, yVec, uvVec, width, height, strideY, strideUV);
	auto imageReq = hyperhdrnet::CreateImage(builder, hyperhdrnet::ImageType_NV12Image, nv12Img.Union(), -1);
	auto req = hyperhdrnet::CreateRequest(builder, hyperhdrnet::Command_Image, imageReq.Union());
	builder.Finish(req);

	return builder.Release();
}
void *transformLoop(void *arg)
{
	_transformation = new OpencvTransform(_imgWidth, _imgHeight, _imgQuad, &_imgWidthToSend, &_imgHeightToSend, _useNv12);
	if (_useNv12)
		_nv12Out.reset(_imgWidthToSend, _imgHeightToSend);
	else
		_rgbOut.reset(_imgWidthToSend, _imgHeightToSend);

#ifdef HAVE_GPU_TRANSFORM
	if (_useNv12 && _useGpu)
		_gpuNv12 = new GpuNv12Transform(_imgWidth, _imgHeight, _imgQuad, _imgWidthToSend, _imgHeightToSend);
	if (!_useNv12 && _useGpu)
		_gpuRgb = new GpuRgbTransform(_imgWidth, _imgHeight, _imgQuad, _imgWidthToSend, _imgHeightToSend);
#endif
	_loopTr.exec();
	return 0;
}

static void transformImg(uint8_t *img)
{
	const int64_t startNs = nowNs();
	uint8_t *outRgb = nullptr;
	const uint16_t outW = static_cast<uint16_t>(_imgWidthToSend);
	const uint16_t outH = static_cast<uint16_t>(_imgHeightToSend);
	auto req = buildRawImageRequestUninitialized(outW, outH, &outRgb);
	_transformation->transformInto(img, _strideBgr, outRgb, static_cast<int>(outW) * 3);
	cpuProfileTick(_cpuRgbProf, "[cpuRgb]", static_cast<uint64_t>(nowNs() - startNs));

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transformation was too slow: " << elapsedMicros << " us\n";
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	auto reqPtr = std::make_shared<flatbuffers::DetachedBuffer>(std::move(req));
	_loopFb.callLater([reqPtr]() mutable
					  { flatBufSendRequest(std::move(*reqPtr)); });
}

static void transformImgNv12(uint8_t *y, uint8_t *uv)
{
	const int64_t startNs = nowNs();
	// Avoid NV12 <-> BGR conversions: warp Y and UV planes directly.
	// UV is half-resolution, so use maps computed in UV coordinate space.
	static thread_local bool inited = false;
	static thread_local cv::Mat mapYx, mapYy;
	static thread_local cv::Mat mapUVx, mapUVy;
	if (!inited)
	{
		cv::Point2f dstQuadY[4] = {
			cv::Point2f(0.f, 0.f),
			cv::Point2f(static_cast<float>(_imgWidthToSend), 0.f),
			cv::Point2f(0.f, static_cast<float>(_imgHeightToSend)),
			cv::Point2f(static_cast<float>(_imgWidthToSend), static_cast<float>(_imgHeightToSend))};
		cv::Mat warpY = cv::getPerspectiveTransform(_imgQuad, dstQuadY);
		cv::Mat warpYInv;
		cv::invert(warpY, warpYInv);
		mapYx.create(_imgHeightToSend, _imgWidthToSend, CV_32FC1);
		mapYy.create(_imgHeightToSend, _imgWidthToSend, CV_32FC1);

		const double y00 = warpYInv.at<double>(0, 0);
		const double y01 = warpYInv.at<double>(0, 1);
		const double y02 = warpYInv.at<double>(0, 2);
		const double y10 = warpYInv.at<double>(1, 0);
		const double y11 = warpYInv.at<double>(1, 1);
		const double y12 = warpYInv.at<double>(1, 2);
		const double y20 = warpYInv.at<double>(2, 0);
		const double y21 = warpYInv.at<double>(2, 1);
		const double y22 = warpYInv.at<double>(2, 2);

		for (int dy = 0; dy < _imgHeightToSend; dy++)
		{
			float *mx = mapYx.ptr<float>(dy);
			float *my = mapYy.ptr<float>(dy);
			for (int dx = 0; dx < _imgWidthToSend; dx++)
			{
				const double denom = y20 * dx + y21 * dy + y22;
				if (denom == 0.0)
				{
					mx[dx] = -1.f;
					my[dx] = -1.f;
					continue;
				}
				mx[dx] = static_cast<float>((y00 * dx + y01 * dy + y02) / denom);
				my[dx] = static_cast<float>((y10 * dx + y11 * dy + y12) / denom);
			}
		}

		cv::Point2f srcQuadUV[4];
		cv::Point2f dstQuadUV[4] = {
			cv::Point2f(0.f, 0.f),
			cv::Point2f(static_cast<float>(_imgWidthToSend) / 2.f, 0.f),
			cv::Point2f(0.f, static_cast<float>(_imgHeightToSend) / 2.f),
			cv::Point2f(static_cast<float>(_imgWidthToSend) / 2.f, static_cast<float>(_imgHeightToSend) / 2.f)};
		for (int i = 0; i < 4; i++)
			srcQuadUV[i] = cv::Point2f(_imgQuad[i].x / 2.f, _imgQuad[i].y / 2.f);
		cv::Mat warpUV = cv::getPerspectiveTransform(srcQuadUV, dstQuadUV);
		cv::Mat warpUVInv;
		cv::invert(warpUV, warpUVInv);

		const int uvW = _imgWidthToSend / 2;
		const int uvH = _imgHeightToSend / 2;
		mapUVx.create(uvH, uvW, CV_32FC1);
		mapUVy.create(uvH, uvW, CV_32FC1);

		const double u00 = warpUVInv.at<double>(0, 0);
		const double u01 = warpUVInv.at<double>(0, 1);
		const double u02 = warpUVInv.at<double>(0, 2);
		const double u10 = warpUVInv.at<double>(1, 0);
		const double u11 = warpUVInv.at<double>(1, 1);
		const double u12 = warpUVInv.at<double>(1, 2);
		const double u20 = warpUVInv.at<double>(2, 0);
		const double u21 = warpUVInv.at<double>(2, 1);
		const double u22 = warpUVInv.at<double>(2, 2);

		for (int dy = 0; dy < uvH; dy++)
		{
			float *mx = mapUVx.ptr<float>(dy);
			float *my = mapUVy.ptr<float>(dy);
			for (int dx = 0; dx < uvW; dx++)
			{
				const double denom = u20 * dx + u21 * dy + u22;
				if (denom == 0.0)
				{
					mx[dx] = -1.f;
					my[dx] = -1.f;
					continue;
				}
				mx[dx] = static_cast<float>((u00 * dx + u01 * dy + u02) / denom);
				my[dx] = static_cast<float>((u10 * dx + u11 * dy + u12) / denom);
			}
		}

		inited = true;
	}

	cv::Mat yIn(_imgHeight, _imgWidth, CV_8UC1, y, _strideY);
	cv::Mat uvIn(_imgHeight / 2, _imgWidth / 2, CV_8UC2, uv, _strideUV);

	uint8_t *outY = nullptr;
	uint8_t *outUV = nullptr;
	const uint16_t outW = static_cast<uint16_t>(_imgWidthToSend);
	const uint16_t outH = static_cast<uint16_t>(_imgHeightToSend);
	const uint16_t outStrideY = outW;
	const uint16_t outStrideUV = outW;
	auto req = buildNV12ImageRequestUninitialized(outW, outH, outStrideY, outStrideUV, &outY, &outUV);

	cv::Mat yOut(_imgHeightToSend, _imgWidthToSend, CV_8UC1, outY, outStrideY);
	cv::Mat uvOut(_imgHeightToSend / 2, _imgWidthToSend / 2, CV_8UC2, outUV, outStrideUV);

	cv::remap(yIn, yOut, mapYx, mapYy, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::remap(uvIn, uvOut, mapUVx, mapUVy, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cpuProfileTick(_cpuNv12Prof, "[cpuNv12]", static_cast<uint64_t>(nowNs() - startNs));

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transformation was too slow: " << elapsedMicros << " us\n";
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	auto reqPtr = std::make_shared<flatbuffers::DetachedBuffer>(std::move(req));
	_loopFb.callLater([reqPtr]() mutable
					  { flatBufSendRequest(std::move(*reqPtr)); });
}

static void transformImgNv12GpuDmabuf(uint8_t *y, uint8_t *uv, int fdY, int offY, int fdUV, int offUV)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)y;
	(void)uv;
	(void)fdY;
	(void)offY;
	(void)fdUV;
	(void)offUV;
	std::cout << "ERROR: GPU transform not compiled in" << std::endl;
	_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	return;
#else
	const int64_t startNs = nowNs();

	if (!_gpuNv12)
	{
		std::cout << "ERROR: GPU transform not initialized" << std::endl;
		_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
		return;
	}

	uint8_t *outY = nullptr;
	uint8_t *outUV = nullptr;
	const uint16_t outW = static_cast<uint16_t>(_imgWidthToSend);
	const uint16_t outH = static_cast<uint16_t>(_imgHeightToSend);
	const uint16_t outStrideY = outW;
	const uint16_t outStrideUV = outW;
	auto req = buildNV12ImageRequestUninitialized(outW, outH, outStrideY, outStrideUV, &outY, &outUV);

	bool ok = _gpuNv12->transformDmabuf(fdY, offY, _strideY, fdUV, offUV, _strideUV, outY, outUV);
	if (!ok)
	{
		// Fallback to CPU upload path if DMA-BUF import isn't supported.
		_gpuNv12->transform(y, uv, _strideY, _strideUV, outY, outUV);
	}

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transformation was too slow: " << elapsedMicros << " us\n";
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	auto reqPtr = std::make_shared<flatbuffers::DetachedBuffer>(std::move(req));
	_loopFb.callLater([reqPtr]() mutable
					  { flatBufSendRequest(std::move(*reqPtr)); });
#endif
}

static void transformImgRgbGpu(uint8_t *img)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)img;
	std::cout << "ERROR: GPU transform not compiled in" << std::endl;
	_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	return;
#else
	const int64_t startNs = nowNs();

	if (!_gpuRgb)
	{
		std::cout << "ERROR: GPU transform not initialized" << std::endl;
		_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
		return;
	}

	uint8_t *outRgb = nullptr;
	const uint16_t outW = static_cast<uint16_t>(_imgWidthToSend);
	const uint16_t outH = static_cast<uint16_t>(_imgHeightToSend);
	auto req = buildRawImageRequestUninitialized(outW, outH, &outRgb);

	GpuRgbTransform::InputFormat fmt = GpuRgbTransform::InputFormat::Packed3;
	if (_bppBgr == 4)
		fmt = _swapRb4 ? GpuRgbTransform::InputFormat::Xrgb8888 : GpuRgbTransform::InputFormat::Xbgr8888;
	_gpuRgb->transform(img, _strideBgr, fmt, outRgb);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transformation was too slow: " << elapsedMicros << " us\n";
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	auto reqPtr = std::make_shared<flatbuffers::DetachedBuffer>(std::move(req));
	_loopFb.callLater([reqPtr]() mutable
					  { flatBufSendRequest(std::move(*reqPtr)); });
#endif
}

static void transformImgRgbGpuDmabuf(int fd, int offsetBytes)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)fd;
	(void)offsetBytes;
	std::cout << "ERROR: GPU transform not compiled in" << std::endl;
	_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	return;
#else
	const int64_t startNs = nowNs();

	if (!_gpuRgb)
	{
		std::cout << "ERROR: GPU transform not initialized" << std::endl;
		_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
		return;
	}

	uint8_t *outRgb = nullptr;
	const uint16_t outW = static_cast<uint16_t>(_imgWidthToSend);
	const uint16_t outH = static_cast<uint16_t>(_imgHeightToSend);
	auto req = buildRawImageRequestUninitialized(outW, outH, &outRgb);

	GpuRgbTransform::InputFormat fmt = GpuRgbTransform::InputFormat::Packed3;
	if (_bppBgr == 4)
		fmt = _swapRb4 ? GpuRgbTransform::InputFormat::Xrgb8888 : GpuRgbTransform::InputFormat::Xbgr8888;
	_gpuRgb->transformDmabuf(fd, offsetBytes, _strideBgr, fmt, outRgb);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transformation was too slow: " << elapsedMicros << " us\n";
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	auto reqPtr = std::make_shared<flatbuffers::DetachedBuffer>(std::move(req));
	_loopFb.callLater([reqPtr]() mutable
					  { flatBufSendRequest(std::move(*reqPtr)); });
#endif
}

static void processRequest(Request *request);

static void requestComplete(Request *request)
{
	if (request->status() == Request::RequestCancelled)
		return;

	handleAutoFocusMetadata(request);

	_loop.callLater(std::bind(&processRequest, request));
}

static void processRequest(Request *request)
{
	const Request::BufferMap &buffers = request->buffers();
	for (auto bufferPair : buffers)
	{
		FrameBuffer *buffer = bufferPair.second;
		auto it = _mappedBuffers.find(buffer);
		if (it == _mappedBuffers.end() || it->second.empty())
			continue;

		if (autoCalibEnabled())
		{
			const int64_t now = nowNs();
			const int64_t nextOk = _autoCalibNextAllowedNs.load(std::memory_order_acquire);
			if (now >= nextOk)
			{
				// Only process one frame at a time.
				if (!_autoCalibBusy.exchange(true, std::memory_order_acq_rel))
				{
					uint8_t *img = it->second[0].data();
					if (_transformEn)
					{
						if (_useGpu)
						{
							int fd = -1;
							int off = 0;
							if (_bppBgr == 4)
							{
								const FrameBuffer::Plane &p0 = buffer->planes()[0];
								fd = p0.fd.get();
								off = static_cast<int>(p0.offset);
							}
							_loopTr.callLater([img, fd, off]()
											  { autoCalibFromTransformedGpu(img, fd, off); });
						}
						else
						{
							_loopTr.callLater([img]()
											  { autoCalibFromTransformedCpu(img); });
						}
					}
					else
					{
						autoCalibFromRaw(img);
					}
				}
			}
		}

		if (!_imgOutPath.empty())
		{
			if (_imgOutDone.exchange(true))
				return;

			uint8_t *img = it->second[0].data();
			if (_transformEn)
			{
				if (_useGpu)
				{
					int fd = -1;
					int off = 0;
					if (_bppBgr == 4)
					{
						const FrameBuffer::Plane &p0 = buffer->planes()[0];
						fd = p0.fd.get();
						off = static_cast<int>(p0.offset);
					}
					_loopTr.callLater([img, fd, off]()
									  { imgOutSaveTransformedGpu(img, fd, off); });
				}
				else
				{
					_loopTr.callLater([img]()
									  { imgOutSaveTransformedCpu(img); });
				}
			}
			else
			{
				imgOutSaveRaw(img);
			}

			// Don't reuse/requeue: we are exiting.
			return;
		}

		if (_transformEn)
		{
			if (_maxPendingProcess > 0)
			{
				const int cur = _pendingProcess.load(std::memory_order_acquire);
				if (cur >= _maxPendingProcess)
				{
					_droppedProcess.fetch_add(1, std::memory_order_relaxed);
					continue;
				}
			}

			_pendingProcess.fetch_add(1, std::memory_order_acq_rel);

			if (_useNv12)
			{
				if (it->second.size() < 2)
					continue;
				uint8_t *y = it->second[0].data();
				uint8_t *uv = it->second[1].data();
				if (_useGpu)
				{
					// Always try DMA-BUF import on the GPU NV12 path (falls back internally if unsupported)
					if (true)
					{
						const FrameBuffer::Plane &p0 = buffer->planes()[0];
						const FrameBuffer::Plane &p1 = buffer->planes()[1];
						int fdY = p0.fd.get();
						int fdUV = p1.fd.get();
						int offY = static_cast<int>(p0.offset);
						int offUV = static_cast<int>(p1.offset);
						_loopTr.callLater([y, uv, fdY, offY, fdUV, offUV]()
										  { transformImgNv12GpuDmabuf(y, uv, fdY, offY, fdUV, offUV); });
					}
				}
				else
					_loopTr.callLater([y, uv]()
									  { transformImgNv12(y, uv); });
			}
			else
			{
				uint8_t *img = it->second[0].data();
				if (_useGpu)
				{
					// Always try DMA-BUF import on the GPU RGB path when the camera format is 4 Bpp.
					if (_bppBgr == 4)
					{
						const FrameBuffer::Plane &p0 = buffer->planes()[0];
						int fd = p0.fd.get();
						int off = static_cast<int>(p0.offset);
						_loopTr.callLater([fd, off]()
										  { transformImgRgbGpuDmabuf(fd, off); });
					}
					else
					{
						_loopTr.callLater([img]()
										  { transformImgRgbGpu(img); });
					}
				}
				else
					_loopTr.callLater([img]()
									  { transformImg(img); });
			}
		}
		else
		{
			if (_maxPendingSend > 0)
			{
				const int cur = _pendingSend.load(std::memory_order_acquire);
				if (cur >= _maxPendingSend)
				{
					_droppedSend.fetch_add(1, std::memory_order_relaxed);
					continue;
				}
			}

			_pendingSend.fetch_add(1, std::memory_order_acq_rel);

			if (_useNv12)
			{
				if (it->second.size() < 2)
					continue;
				uint8_t *y = it->second[0].data();
				uint8_t *uv = it->second[1].data();
				_loopFb.callLater([y, uv]()
								  { flatBufSendNv12(y, uv, _strideY, _strideUV, _imgWidthToSend, _imgHeightToSend); });
			}
			else
			{
				uint8_t *img = it->second[0].data();
				if (_outputCropEn)
				{
					// Crop the raw image before sending.
					// Default behavior allocates a new buffer per frame. When maxPendingSend is enabled,
					// we can reuse a small ring buffer to reduce allocation jitter.
					if (_maxPendingSend > 0)
					{
						auto view = _cropOut.next();
						if (!cropRgbInto(img, _imgWidthToSend, _imgHeightToSend, _strideBgr,
										 _outputCropX, _outputCropY, _outputCropWidth, _outputCropHeight,
										 view.rgb, _outputCropWidth * 3))
						{
							_pendingSend.fetch_sub(1, std::memory_order_acq_rel);
							continue;
						}
						uint8_t *croppedPtr = view.rgb;
						_loopFb.callLater([croppedPtr]()
										  {
							const int64_t startNs = nowNs();
							if (flatbufTryConnect())
							{
								if (_flatbuf.setImage(croppedPtr, _outputCropWidth, _outputCropHeight) < 0)
								{
									std::cout << "ERROR: Flatbuffer write timeout\n";
								}
							}
							else
							{
								bool expected = false;
								if (_flatbufConnErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
									std::cout << "ERROR: Flatbuffer connection timeout\n";
							}

							int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
							if (before > 1 && _verbose)
							{
								int64_t elapsedMicros = (nowNs() - startNs) / 1000;
								std::cout << "WARNING: Frame sending was too slow: " << elapsedMicros << " us\n";
							} });
					}
					else
					{
						std::vector<uint8_t> cropped = cropRgb(img, _imgWidthToSend, _imgHeightToSend, _strideBgr,
															   _outputCropX, _outputCropY, _outputCropWidth, _outputCropHeight);
						auto croppedPtr = std::make_shared<std::vector<uint8_t>>(std::move(cropped));
						_loopFb.callLater([croppedPtr]()
										  {
						const int64_t startNs = nowNs();
						if (flatbufTryConnect())
						{
							if (_flatbuf.setImage(croppedPtr->data(), _outputCropWidth, _outputCropHeight) < 0)
							{
								std::cout << "ERROR: Flatbuffer write timeout\n";
							}
						}
						else
						{
							bool expected = false;
							if (_flatbufConnErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
								std::cout << "ERROR: Flatbuffer connection timeout\n";
						}

						int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
						if (before > 1 && _verbose)
						{
							int64_t elapsedMicros = (nowNs() - startNs) / 1000;
							std::cout << "WARNING: Frame sending was too slow: " << elapsedMicros << " us\n";
						} });
					}
				}
				else
				{
					_loopFb.callLater([img]()
									  { flatBufSend(img); });
				}
			}
		}
	}

	request->reuse(Request::ReuseBuffers);

	// Only update controls per-frame when auto-calibration/focus features are enabled
	if (autoCalibEnabled() || _autoFocusOnce)
	{
		ControlList &c = request->controls();

		// Apply focus controls (changes when auto-focus is active)
		applyFocusControls(c);

		// Apply saturation/contrast/gamma (changes when auto-saturation is active)
		c.set(controls::Saturation, _saturation);
		c.set(controls::Contrast, _contrast);
		if (_gamma > 0.0f)
			c.set(controls::Gamma, _gamma);

		if (autoCalibEnabled())
		{
			// Program per-request controls so that any updated _gain/_rGain/_bGain take effect
			// without restarting the camera.
			c.set(controls::AeEnable, false);
			c.set(controls::AwbEnable, false);
			c.set(controls::ExposureTime, _shutter);
			c.set(controls::AnalogueGain, _gain);
			c.set(controls::ColourGains, libcamera::Span<const float, 2>({_rGain, _bGain}));
		}
	}

	_camera->queueRequest(request);
}
// Main loop
int main(int argc, char *argv[])
{
	po::options_description options("Options");
	options.add_options()																																								   //
		("width,w", po::value<int>(&_width)->default_value(640), "Width")																												   //
		("height,h", po::value<int>(&_height)->default_value(480), "Height")																											   //
		("shutter,s", po::value<int32_t>(&_shutter)->default_value(16666), "Shutter speed in us")																						   //
		("fps", po::value<float>(&_targetFps)->default_value(0.0f), "Target FPS limit (0=disabled, adds delay to maintain frame rate)")													   //
		("gain,g", po::value<float>(&_gain)->default_value(1.0), "Camera gain")																											   //
		("wbgain,b", po::value<std::string>(&_wbgainsStr)->default_value("1.0,1.0"), "White balance gains")																				   //
		("origin,o", po::value<std::string>(&_origin)->default_value("camera"), "Name of the origin")																					   //
		("connection,c", po::value<std::string>(&_conn)->default_value("127.0.0.1"), "Socket address (IP or Unix socket path)")															   //
		("port,p", po::value<int>(&_port)->default_value(19400), "TCP port")																											   //
		("priority,r", po::value<int>(&_priority)->default_value(120), "Hyperion priority")																								   //
		("transform,t", po::value<std::string>(&_transformStr), "Picture transformation in format x1,y1:x2,y2:x3,y3:x4,y4")																   //
		("output-crop", po::value<std::string>(&_outputCropStr), "Output crop in format x,y,width,height (applied after transform)")													   //
		("nv12", po::bool_switch(&_useNv12)->default_value(false), "Enable NV12 format output (default is BGR888)")																		   //
		("cpuonly", po::bool_switch(&_cpuOnly)->default_value(false), "Force CPU transform even if GPU is available")																	   //
		("verbose,v", po::bool_switch(&_verbose)->default_value(false), "Verbose output")																								   //
		("imgout", po::value<std::string>(&_imgOutPath), "Capture one image (RGB) to file (png/bmp/...) and exit")																		   //
		("imgout-crophelp", po::value<std::string>(&_imgOutCropHelpPath), "Capture image showing sensor area used at requested resolution (helps determine --crop coordinates)")		   //
		("autowb", po::bool_switch(&_autoWb)->default_value(false), "One-shot auto white balance calibration (gray 127) and exit")														   //
		("autogain", po::bool_switch(&_autoGain)->default_value(false), "One-shot auto gain calibration (gray 127) and exit")															   //
		("autocalib", po::bool_switch(&_autoCalib)->default_value(false), "One-shot auto white balance + gain calibration (gray 127) and exit")											   //
		("autosat", po::value<std::string>(&_autoSatTargetSpec)->implicit_value(""), "One-shot auto saturation calibration and exit (optional arg: R,G,B target; default 230,50,50)")	   //
		("autosat-iters", po::value<int>(&_autoSatMaxIters)->default_value(24), "Auto saturation max iterations")																		   //
		("autocalib-iters", po::value<int>(&_autoCalibMaxIters)->default_value(8), "Auto calibration max iterations")																	   //
		("autocalib-delay-ms", po::value<int>(&_autoCalibDelayMs)->default_value(500), "Delay between auto calibration updates (ms)")													   //
		("autocalib-tol-wb", po::value<float>(&_autoCalibTolWb)->default_value(2.0), "Auto WB convergence tolerance (abs channel delta: |R-G| and |B-G|)")								   //
		("autocalib-tol-grey", po::value<float>(&_autoCalibTolGrey)->default_value(3.0), "Auto gain convergence tolerance (abs avg RGB error from 127)")								   //
		("autocalib-gain-samples", po::value<int>(&_autoGainSamples)->default_value(8), "Auto gain samples (curve-fit)")																   //
		("autocalib-gain-range", po::value<float>(&_autoGainRange)->default_value(4.0), "Auto gain range multiplier (min=gain/range, max=gain*range)")									   //
		("autofocus-once", po::bool_switch(&_autoFocusOnce)->default_value(false), "Run autofocus once at startup, print focus, then lock manual focus")								   //
		("focus,f", po::value<float>(&_focus)->default_value(-1.0f), "Manual focus lens position (-1 keeps default)")																	   //
		("saturation", po::value<float>(&_saturation)->default_value(1.0f), "Camera saturation")																						   //
		("contrast", po::value<float>(&_contrast)->default_value(1.0f), "Camera contrast")																								   //
		("gamma", po::value<float>(&_gamma)->default_value(0.0f), "Camera gamma (0.0=use camera default)")																				   //
		("hflip", po::bool_switch(&_hflip)->default_value(false), "Horizontal flip (native camera transform)")																			   //
		("vflip", po::bool_switch(&_vflip)->default_value(false), "Vertical flip (native camera transform)")																			   //
		("rotate", po::value<int>(&_rotate)->default_value(0), "Rotation in degrees: 0, 90, 180, 270 (native camera transform)")														   //
		("crop", po::value<std::string>(&_cropStr), "Crop region in format x,y,width,height (native camera ScalerCrop)")																   //
		("version", po::bool_switch(&_version)->default_value(false), "Print version and exit")																							   //
		("help", po::bool_switch(&_help)->default_value(false), "Print help")																											   //
		("max-pending-process", po::value<int>(&_maxPendingProcess)->default_value(0), "Max in-flight transform jobs (0=unlimited). Set to 1 for lowest latency (drops frames when busy)") //
		("max-pending-send", po::value<int>(&_maxPendingSend)->default_value(0), "Max in-flight send jobs (0=unlimited). Set to 1-2 for lowest latency (drops frames when busy)")		   //
		("connect-timeout-ms", po::value<int>(&_flatbufConnectTimeoutMs)->default_value(250), "Socket connect timeout in ms (lower = lower latency when server is down)")				   //
		("reconnect-interval-ms", po::value<int>(&_flatbufReconnectIntervalMs)->default_value(250), "Min time between reconnect attempts in ms when disconnected");

	std::signal(SIGPIPE, SIG_IGN); // Ignore SIGPIPE to prevent crashes on broken pipes

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, options), vm);
	po::notify(vm);

	_flatbuf.setConnectTimeoutMs(_flatbufConnectTimeoutMs);
	_autoSat = (vm.count("autosat") > 0);
	if (_autoSat)
	{
		if (!_autoSatTargetSpec.empty())
		{
			int tr = 0, tg = 0, tb = 0;
			if (!parseRgbTriplet(_autoSatTargetSpec, tr, tg, tb))
			{
				std::cout << "ERROR: Invalid --autosat target. Use R,G,B with 0..255 (non-grey), e.g. --autosat 230,25,25" << std::endl;
				return EXIT_FAILURE;
			}
			_autoSatTargetR = tr;
			_autoSatTargetG = tg;
			_autoSatTargetB = tb;
		}
	}

	if (_version)
	{
		std::cout << "hyperion-cam " << HYPERCAM_VERSION << std::endl;
		return 0;
	}

	if (_help)
	{
		std::cout << std::endl
				  << options << std::endl;
		return 0;
	}

	std::cout << "INFO: hyperion-cam version " << HYPERCAM_VERSION << std::endl;

	if (_autoFocusOnce)
	{
		_autoFocusPhase.store(1, std::memory_order_release);
		_autoFocusTriggerSent.store(false, std::memory_order_release);
		_autoFocusLockedPos.store(-1.0f, std::memory_order_release);
	}

	if (vm.count("transform"))
	{
		if (sscanf(_transformStr.c_str(), "%f,%f:%f,%f:%f,%f:%f,%f",
				   &_imgQuad[0].x, &_imgQuad[0].y, &_imgQuad[1].x, &_imgQuad[1].y, &_imgQuad[2].x, &_imgQuad[2].y, &_imgQuad[3].x, &_imgQuad[3].y) == 8)
		{
			_transformEn = true;
		}
		else
		{
			std::cout << "ERROR: Invalid value for transformation" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (vm.count("crop"))
	{
		if (sscanf(_cropStr.c_str(), "%d,%d,%d,%d", &_cropX, &_cropY, &_cropWidth, &_cropHeight) == 4)
		{
			if (_cropWidth <= 0 || _cropHeight <= 0)
			{
				std::cout << "ERROR: Crop width and height must be positive" << std::endl;
				return EXIT_FAILURE;
			}
			_cropEn = true;
			if (_verbose)
				std::cout << "INFO: Crop enabled: x=" << _cropX << ", y=" << _cropY
						  << ", width=" << _cropWidth << ", height=" << _cropHeight << std::endl;
		}
		else
		{
			std::cout << "ERROR: Invalid crop format. Use x,y,width,height" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (vm.count("output-crop"))
	{
		if (sscanf(_outputCropStr.c_str(), "%d,%d,%d,%d", &_outputCropX, &_outputCropY, &_outputCropWidth, &_outputCropHeight) == 4)
		{
			if (_outputCropWidth <= 0 || _outputCropHeight <= 0)
			{
				std::cout << "ERROR: Output crop width and height must be positive" << std::endl;
				return EXIT_FAILURE;
			}
			if (_outputCropX < 0 || _outputCropY < 0)
			{
				std::cout << "ERROR: Output crop x and y must be non-negative" << std::endl;
				return EXIT_FAILURE;
			}
			_outputCropEn = true;
			if (_verbose)
			{
				std::cout << "INFO: Output crop enabled: x=" << _outputCropX << ", y=" << _outputCropY
						  << ", width=" << _outputCropWidth << ", height=" << _outputCropHeight << std::endl;
			}
		}
		else
		{
			std::cout << "ERROR: Invalid output-crop format. Use x,y,width,height" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (_outputCropEn && _maxPendingSend > 0)
	{
		const std::size_t ringSize = static_cast<std::size_t>(std::max(3, _maxPendingSend + 1));
		_cropOut.reset(_outputCropWidth, _outputCropHeight, ringSize);
	}

	// Handle --imgout-crophelp (replaces --imgout)
	if (!_imgOutCropHelpPath.empty())
	{
		if (!_imgOutPath.empty())
		{
			std::cout << "WARNING: Both --imgout and --imgout-crophelp specified, using --imgout-crophelp" << std::endl;
		}
		_imgOutPath = _imgOutCropHelpPath;
		// crophelp disables transforms and crops to show actual sensor area used
		_transformEn = false;
		_cropEn = false;
		_outputCropEn = false;
		if (_verbose)
			std::cout << "INFO: --imgout-crophelp enabled, will capture at sensor area used for requested resolution" << std::endl;
	}

	if (!_imgOutPath.empty())
	{
		// Still image output uses RGB/BGR only.
		if (_useNv12 && _verbose)
			std::cout << "INFO: --imgout forces RGB output; ignoring --nv12" << std::endl;
		_useNv12 = false;
	}

	if (autoCalibEnabled())
	{
		if (!_imgOutPath.empty())
		{
			std::cout << "ERROR: --imgout cannot be used together with auto calibration" << std::endl;
			return EXIT_FAILURE;
		}

		if (_useNv12 && _verbose)
			std::cout << "INFO: auto calibration forces RGB output; ignoring --nv12" << std::endl;
		_useNv12 = false;

		if (_autoCalibMaxIters < 1)
			_autoCalibMaxIters = 1;
		if (_autoCalibDelayMs < 0)
			_autoCalibDelayMs = 0;

		if (_autoGainSamples < 3)
			_autoGainSamples = 3;
		if (_autoGainRange < 1.01f)
			_autoGainRange = 1.01f;

		// Reset curve state on start.
		_autoGainCurvePhase = 0;
		_autoGainCurveGains.clear();
		_autoGainCurveYs.clear();
		_autoGainCurveNextIdx = 0;
		_autoCalibStage.store(0, std::memory_order_release);
		_autoCalibIterCount.store(0, std::memory_order_release);
		_autoCalibFrameCount.store(0, std::memory_order_release);
		_autoCalibNextAllowedNs.store(0, std::memory_order_release);
	}

	// Validate mutually exclusive options
	if (_transformEn && _outputCropEn)
	{
		std::cout << "ERROR: --transform and --output-crop are mutually exclusive" << std::endl;
		std::cout << "       If using transform, adjust the transform coordinates instead of cropping" << std::endl;
		return EXIT_FAILURE;
	}

	// GPU transform is the default when compiled in and --transform is used.
	// Use --cpuonly to force the CPU path.
#ifdef HAVE_GPU_TRANSFORM
	_useGpu = (_transformEn && !_cpuOnly);
#else
	_useGpu = false;
	if (_transformEn && !_cpuOnly && _verbose)
		std::cout << "INFO: GPU transform support was not compiled in; using CPU transform" << std::endl;
#endif

	if (sscanf(_wbgainsStr.c_str(), "%f,%f", &_rGain, &_bGain) != 2)
	{
		std::cout << "ERROR: wrong white balance gains!" << std::endl;
		return EXIT_FAILURE;
	}

	// Create a Camera Manager.
	std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
	cm->start();

	if (_verbose)
	{
		std::cout << "Camera Id list: " << std::endl;
		for (auto const &camera : cm->cameras())
			std::cout << "\t" << camera->id() << std::endl;
	}

	// Camera
	if (cm->cameras().empty())
	{
		std::cout << "ERROR: No cameras were identified on the system." << std::endl;
		cm->stop();
		return EXIT_FAILURE;
	}

	std::string cameraId = cm->cameras()[0]->id();
	_camera = cm->get(cameraId);
	_camera->acquire();

	// For --imgout-crophelp, configure at requested resolution to see what sensor area is used
	if (!_imgOutCropHelpPath.empty())
	{
		std::unique_ptr<CameraConfiguration> tempConfig =
			_camera->generateConfiguration({StreamRole::Viewfinder});
		if (tempConfig && !tempConfig->empty())
		{
			// First: Configure camera with requested resolution to get auto-crop area
			StreamConfiguration &tempStreamConfig = tempConfig->at(0);
			tempStreamConfig.size = libcamera::Size(_width, _height);
			tempConfig->validate();
			_camera->configure(tempConfig.get());

			// Query ScalerCropMaximum to see what sensor area is actually being used
			const ControlList &props = _camera->properties();
			auto cropMax = props.get(properties::ScalerCropMaximum);
			if (cropMax)
			{
				// Store the crop rectangle for drawing later
				_imgOutCropHelpDrawRect = true;
				_imgOutCropHelpRectX = cropMax->x;
				_imgOutCropHelpRectY = cropMax->y;
				_imgOutCropHelpRectWidth = cropMax->width;
				_imgOutCropHelpRectHeight = cropMax->height;
				if (_verbose)
				{
					std::cout << "INFO: At " << _width << "x" << _height << " resolution, camera uses sensor area: "
							  << cropMax->toString() << std::endl;
				}
			}

			// Second: Reconfigure to capture at full sensor resolution
			tempStreamConfig.size = libcamera::Size(16384, 16384);
			tempConfig->validate();
			_camera->configure(tempConfig.get());

			// Get the full sensor resolution
			auto fullCropMax = props.get(properties::ScalerCropMaximum);
			if (fullCropMax)
			{
				_width = fullCropMax->width;
				_height = fullCropMax->height;
				if (_verbose)
					std::cout << "INFO: Capturing full sensor at: " << _width << "x" << _height << std::endl;
			}
			else if (_verbose)
			{
				std::cout << "WARNING: ScalerCropMaximum not available" << std::endl;
			}
		}
		else if (_verbose)
		{
			std::cout << "WARNING: Could not query camera formats, using default resolution" << std::endl;
		}
	}

	// Stream
	std::unique_ptr<CameraConfiguration> config =
		_camera->generateConfiguration({StreamRole::Viewfinder});

	// Apply native libcamera transform (rotation + horizontal/vertical flip)
	if (_rotate != 0 || _hflip || _vflip)
	{
		// Validate rotation value
		if (_rotate != 0 && _rotate != 90 && _rotate != 180 && _rotate != 270)
		{
			std::cout << "ERROR: Invalid rotation value. Use 0, 90, 180, or 270 degrees." << std::endl;
			return EXIT_FAILURE;
		}

		libcamera::Orientation orientation;

		// Map rotation + flip combinations to libcamera Orientation
		if (_rotate == 0)
		{
			if (!_hflip && !_vflip)
				orientation = libcamera::Orientation::Rotate0;
			else if (_hflip && !_vflip)
				orientation = libcamera::Orientation::Rotate0Mirror;
			else if (!_hflip && _vflip)
				orientation = libcamera::Orientation::Rotate180Mirror;
			else // both hflip and vflip
				orientation = libcamera::Orientation::Rotate180;
		}
		else if (_rotate == 90)
		{
			if (!_hflip && !_vflip)
				orientation = libcamera::Orientation::Rotate90;
			else if (_hflip && !_vflip)
				orientation = libcamera::Orientation::Rotate90Mirror;
			else if (!_hflip && _vflip)
				orientation = libcamera::Orientation::Rotate270Mirror;
			else // both hflip and vflip
				orientation = libcamera::Orientation::Rotate270;
		}
		else if (_rotate == 180)
		{
			if (!_hflip && !_vflip)
				orientation = libcamera::Orientation::Rotate180;
			else if (_hflip && !_vflip)
				orientation = libcamera::Orientation::Rotate180Mirror; // same as vflip at 0°
			else if (!_hflip && _vflip)
				orientation = libcamera::Orientation::Rotate0Mirror; // same as hflip at 0°
			else													 // both hflip and vflip
				orientation = libcamera::Orientation::Rotate0;		 // 180° + both flips = identity
		}
		else // _rotate == 270
		{
			if (!_hflip && !_vflip)
				orientation = libcamera::Orientation::Rotate270;
			else if (_hflip && !_vflip)
				orientation = libcamera::Orientation::Rotate270Mirror;
			else if (!_hflip && _vflip)
				orientation = libcamera::Orientation::Rotate90Mirror;
			else // both hflip and vflip
				orientation = libcamera::Orientation::Rotate90;
		}

		config->orientation = orientation;
		if (_verbose)
			std::cout << "INFO: Native transform set: Rotate=" << _rotate << "°, HFlip=" << (_hflip ? "Yes" : "No")
					  << ", VFlip=" << (_vflip ? "Yes" : "No") << " (Orientation: " << orientation << ")" << std::endl;
	}

	// Stream configuration
	StreamConfiguration &streamConfig = config->at(0);

	if (_verbose)
	{
		std::cout << "Supported pixel formats: ";
		std::vector<PixelFormat> formats = streamConfig.formats().pixelformats();
		for (const PixelFormat format : formats)
		{
			std::cout << format.toString() << ", ";
		}
		std::cout << std::endl;
	}

	if (_useNv12)
	{
		streamConfig.pixelFormat = libcamera::formats::NV12;
	}
	else
	{
		// In GPU mode, try to get a 4 bytes/pixel format to reduce CPU packing costs.
		// libcamera pixel formats map to DRM fourcc:
		// - XBGR8888: memory is R,G,B,X (no swap needed)
		// - XRGB8888: memory is B,G,R,X (swap R/B)
		_bppBgr = 3;
		_swapRb4 = false;
		if (_useGpu)
		{
			// In GPU mode, prefer 4 bytes/pixel formats to enable DMA-BUF import (zero-copy upload).
			bool hasXbgr = false;
			bool hasXrgb = false;
			std::vector<PixelFormat> fmts = streamConfig.formats().pixelformats();
			for (const PixelFormat &f : fmts)
			{
				if (f == libcamera::formats::XBGR8888)
					hasXbgr = true;
				else if (f == libcamera::formats::XRGB8888)
					hasXrgb = true;
			}

			if (hasXbgr)
			{
				streamConfig.pixelFormat = libcamera::formats::XBGR8888;
				_bppBgr = 4;
				_swapRb4 = false;
			}
			else if (hasXrgb)
			{
				streamConfig.pixelFormat = libcamera::formats::XRGB8888;
				_bppBgr = 4;
				_swapRb4 = true;
			}
			else
			{
				streamConfig.pixelFormat = libcamera::formats::BGR888;
				_bppBgr = 3;
				_swapRb4 = false;
			}
		}
		else
		{
			streamConfig.pixelFormat = libcamera::formats::BGR888;
			_bppBgr = 3;
			_swapRb4 = false;
		}
	}

	if (_useNv12)
	{
		streamConfig.colorSpace = libcamera::ColorSpace::Smpte170m;
	}
	else
	{
		streamConfig.colorSpace = libcamera::ColorSpace::Rec709;
	}

	// When crop is enabled, use crop dimensions as output size (1:1, no scaling)
	// Use the requested width/height
	// Note: if --crop is used, the ScalerCrop control crops at sensor level first,
	// then the stream is scaled to the requested output size (allows downscaling after crop)
	libcamera::Size mySize(_width, _height);
	streamConfig.size = mySize;

	if (_cropEn && _verbose)
	{
		std::cout << "INFO: Sensor crop will be " << _cropWidth << "x" << _cropHeight
				  << ", output scaled to " << _width << "x" << _height << std::endl;
	}

	// Store original configuration string for comparison if adjusted
	std::string requestedConfig = streamConfig.toString();

	auto valRet = config->validate();
	if (valRet == CameraConfiguration::Status::Adjusted)
	{
		std::string adjustedConfig = streamConfig.toString();
		std::cout << "WARNING: Configuration Adjusted by camera" << std::endl;
		std::cout << "  Requested: " << requestedConfig << std::endl;
		std::cout << "  Adjusted:  " << adjustedConfig << std::endl;
	}
	else if (valRet == CameraConfiguration::Status::Invalid)
	{
		std::cout << "ERROR: Configuration invalid" << std::endl;
		return EXIT_FAILURE;
	}

	if (_useNv12)
	{
		if (streamConfig.pixelFormat != libcamera::formats::NV12)
		{
			std::cout << "ERROR: NV12 format is not supported" << std::endl;
			return EXIT_FAILURE;
		}
		if ((streamConfig.size.width % 2) != 0 || (streamConfig.size.height % 2) != 0)
		{
			std::cout << "ERROR: NV12 requires even width and height" << std::endl;
			return EXIT_FAILURE;
		}
		_strideY = static_cast<int>(streamConfig.stride);
		_strideUV = static_cast<int>(streamConfig.stride);
		if (_strideY <= 0)
			_strideY = streamConfig.size.width;
		if (_strideUV <= 0)
			_strideUV = streamConfig.size.width;
	}
	else
	{
		if (_useGpu && _bppBgr == 4)
		{
			if (streamConfig.pixelFormat != libcamera::formats::XBGR8888 && streamConfig.pixelFormat != libcamera::formats::XRGB8888)
			{
				std::cout << "ERROR: 4 Bpp X*RGB8888 format is not supported" << std::endl;
				return EXIT_FAILURE;
			}
			// If libcamera adjusted the format away from our preference, update swap flag.
			_swapRb4 = (streamConfig.pixelFormat == libcamera::formats::XRGB8888);
		}
		else if (streamConfig.pixelFormat != libcamera::formats::BGR888)
		{
			std::cout << "ERROR: BGR888 format is not supported" << std::endl;
			return EXIT_FAILURE;
		}
		_strideBgr = static_cast<int>(streamConfig.stride);
		if (_strideBgr <= 0)
			_strideBgr = static_cast<int>(streamConfig.size.width) * _bppBgr;
		if (_verbose)
		{
			std::cout << "Using RGB pixel format: " << streamConfig.pixelFormat.toString()
					  << " (bpp=" << _bppBgr << ", swapRB4=" << (_swapRb4 ? 1 : 0)
					  << ")" << std::endl;
		}
	}

	_imgHeight = streamConfig.size.height;
	_imgWidth = streamConfig.size.width;

	if (_verbose)
		std::cout << "INFO: Stream configured to: " << _imgWidth << "x" << _imgHeight << std::endl;

	if (!_transformEn)
	{
		_imgHeightToSend = _imgHeight;
		_imgWidthToSend = _imgWidth;
	}

	_camera->configure(config.get());

	// Show ScalerCropMaximum property after configuration (required by libcamera)
	if (_verbose || _cropEn)
	{
		const ControlList &props = _camera->properties();
		auto cropMax = props.get(properties::ScalerCropMaximum);
		if (cropMax)
		{
			std::cout << "INFO: ScalerCropMaximum (sensor area): " << cropMax->toString() << std::endl;

			// Validate crop against maximum if crop is enabled
			if (_cropEn)
			{
				if (_cropX < cropMax->x || _cropY < cropMax->y ||
					_cropX + _cropWidth > static_cast<int>(cropMax->x + cropMax->width) ||
					_cropY + _cropHeight > static_cast<int>(cropMax->y + cropMax->height))
				{
					std::cout << "WARNING: Crop region " << _cropX << "," << _cropY << "," << _cropWidth << "," << _cropHeight
							  << " exceeds sensor area " << cropMax->toString() << std::endl;
					std::cout << "         The crop will be adjusted by the camera" << std::endl;
				}
			}
		}
	}

	// Buffer Allocation
	FrameBufferAllocator *allocator = new FrameBufferAllocator(_camera);

	for (StreamConfiguration &cfg : *config)
	{
		int ret = allocator->allocate(cfg.stream());
		if (ret < 0)
		{
			std::cout << "ERROR: Can't allocate buffers" << std::endl;
			return EXIT_FAILURE;
		}

		size_t allocated = allocator->buffers(cfg.stream()).size();

		if (_verbose)
			std::cout << "Allocated " << allocated << " buffers for stream" << std::endl;
	}

	// Frame Capture
	Stream *stream = streamConfig.stream();
	const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);
	std::vector<std::unique_ptr<Request>> requests;
	for (unsigned int i = 0; i < buffers.size(); ++i)
	{
		std::unique_ptr<Request> request = _camera->createRequest();
		if (!request)
		{
			std::cout << "ERROR: Can't create request" << std::endl;
			return EXIT_FAILURE;
		}

		const std::unique_ptr<FrameBuffer> &buffer = buffers[i];
		int ret = request->addBuffer(stream, buffer.get());
		if (ret < 0)
		{
			std::cout << "ERROR: Can't set buffer for request"
					  << std::endl;
			return EXIT_FAILURE;
		}

		ControlList &controls = request->controls();
		applyFocusControls(controls);
		controls.set(controls::AwbEnable, false);
		controls.set(controls::ColourGains, libcamera::Span<const float, 2>({_rGain, _bGain}));
		controls.set(controls::Saturation, _saturation);
		controls.set(controls::Contrast, _contrast);

		requests.push_back(std::move(request));
	}

	_camera->requestCompleted.connect(requestComplete);

	// Buffers memory mapping
	for (const std::unique_ptr<libcamera::FrameBuffer> &buffer : buffers)
	{
		for (const FrameBuffer::Plane &plane : buffer->planes())
		{
			size_t mapLen = plane.offset + plane.length;
			void *memory = mmap(NULL, mapLen, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
			uint8_t *planePtr = static_cast<uint8_t *>(memory) + plane.offset;
			_mappedBuffers[buffer.get()].push_back(libcamera::Span<uint8_t>(planePtr, plane.length));
		}
	}

	// controls the setting of camera
	ControlList controls;
	controls.set(controls::AeEnable, false);
	controls.set(controls::AeFlickerMode, controls::AeFlickerModeEnum::FlickerOff);
	applyFocusControls(controls);
	controls.set(controls::ExposureTimeMode, controls::ExposureTimeModeEnum::ExposureTimeModeManual);
	controls.set(controls::ExposureTime, _shutter);
	controls.set(controls::AnalogueGainMode, controls::AnalogueGainModeEnum::AnalogueGainModeManual);
	controls.set(controls::AnalogueGain, _gain);
	controls.set(controls::Brightness, 0.0);
	controls.set(controls::Contrast, _contrast);
	controls.set(controls::AwbEnable, false);
	controls.set(controls::ColourGains, libcamera::Span<const float, 2>({_rGain, _bGain}));
	controls.set(controls::Saturation, _saturation);
	if (_gamma > 0.0f)
		controls.set(controls::Gamma, _gamma);
	controls.set(controls::Sharpness, 0.0);

	// Apply ScalerCrop if specified
	if (_cropEn)
	{
		libcamera::Rectangle cropRect(_cropX, _cropY, _cropWidth, _cropHeight);
		controls.set(controls::ScalerCrop, cropRect);
		if (_verbose)
			std::cout << "INFO: ScalerCrop set to: " << cropRect.toString() << std::endl;
	}

	// Set frame duration limits based on target FPS
	if (_targetFps > 0.0f)
	{
		// Convert FPS to frame duration in microseconds
		int64_t frameDurationUs = static_cast<int64_t>(1000000.0 / _targetFps);
		// Set both min and max to the same value to enforce specific FPS
		controls.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({frameDurationUs, frameDurationUs}));
		if (_verbose)
			std::cout << "FPS limit set to " << _targetFps << " (frame duration: " << frameDurationUs << " us)\n";
	}
	else
	{
		// Default: allow 100us to 1s frame duration (10000 FPS to 1 FPS)
		controls.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({INT64_C(100), INT64_C(1000000)}));
	}

	// Start Capture
	_camera->start(&controls);
	for (std::unique_ptr<Request> &request : requests)
		_camera->queueRequest(request.get());

	// Run thread for flatbuffer
	if (_imgOutPath.empty())
	{
		pthread_t ptidFb;
		pthread_create(&ptidFb, NULL, &flatLoop, NULL);
	}

	// Run thread for opencv transformation
	if (_transformEn)
	{
		pthread_t ptidCv;
		pthread_create(&ptidCv, NULL, &transformLoop, NULL);
	}

	// Run an EventLoop for camera
	_loop.exec();

	// Clean Up
	_camera->stop();
	allocator->free(stream);
	delete allocator;
	_camera->release();
	_camera.reset();
	cm->stop();

	return EXIT_SUCCESS;
}
