#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <csignal>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <cerrno>
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
#include "ledCalib.h"

#ifdef HAVE_GPU_TRANSFORM
#include "gpuNv12Transform.h"
#include "gpuRgbTransform.h"
#endif

static constexpr const char *HYPERCAM_VERSION = "1.0.1";

using namespace libcamera;
namespace po = boost::program_options;

static std::shared_ptr<Camera> _camera;
bool _useNv12 = false;
bool _useGpu = false;
bool _cpuOnly = false;
bool _hyperhdrCalib = false;

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

std::map<libcamera::FrameBuffer *, std::vector<libcamera::Span<uint8_t>>> _mappedBuffers;
static EventLoop _loop;
static EventLoop _loopFb;
static EventLoop _loopTr;
FlatBufferConnection _flatbuf;
OpencvTransform *_transformation;

#ifdef HAVE_GPU_TRANSFORM
GpuNv12Transform *_gpuNv12 = nullptr;
GpuRgbTransform *_gpuRgb = nullptr;
#endif
cv::Point2f _imgQuad[4];
int _imgWidth = 0;
int _imgHeight = 0;
int _imgWidthToSend;
int _imgHeightToSend;

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

// Real-time scheduling / CPU affinity (latency-first defaults).
static bool _noRealtime = false; // --no-realtime disables SCHED_FIFO
static int _rtPriority = 10;		 // --rt-priority (SCHED_FIFO, well below kernel threads ~50)
static int _sendTimeoutMs = 100; // --send-timeout-ms (SO_SNDTIMEO)

// Direct dispatch: run processRequest() on the libcamera camera-manager thread
// instead of bouncing through the capture event loop (saves one eventfd wakeup
// + context switch per frame). Only used in pure streaming mode; --imgout and
// auto-calibration keep the event-loop hop so their heavy work stays off the
// camera thread. --no-direct-dispatch restores the old behavior.
static bool _noDirectDispatch = false;
static bool _directDispatch = false;

// --hyperhdr-calib output geometry (NV12 only)
static bool _calibAdjustEn = false;
static int _calibCropX = 0;
static int _calibCropY = 0;
static int _calibCropW = 0;
static int _calibCropH = 0;
static int _calibOutW = 0;
static int _calibOutH = 0;
static bool _calibNeedCrop = false;
static bool _calibNeedScale = false;
static int _xformInterp = cv::INTER_NEAREST;

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

static bool _transformUserEn = false;
static int _transformForcedW = 0;
static int _transformForcedH = 0;

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
// Auto-shutter (used when gain is at minimum but image is too bright)
static bool _autoGainUseShutter = false;
static int _autoShutterCurvePhase = 0; // 0=off, 1=sampling, 2=confirm
static std::vector<float> _autoShutterCurveShutters;
static std::vector<float> _autoShutterCurveYs;
static size_t _autoShutterCurveNextIdx = 0;
static const int32_t _autoShutterMinUs = 100;
static float _autoShutterClampMin = 0.0f;
static float _autoShutterClampMax = 0.0f;

static inline bool autoCalibEnabled()
{
	return _autoCalib || _autoWb || _autoGain || _autoSat;
}

static inline float clampf(float v, float lo, float hi)
{
	return std::max(lo, std::min(hi, v));
}

static inline int makeEven(int v)
{
	v &= ~1;
	return (v < 2) ? 2 : v;
}

static void calcTransformOutputSize(int &outW, int &outH, bool alignEven)
{
	int w = static_cast<int>(std::lround(std::max(_imgQuad[1].x - _imgQuad[0].x, _imgQuad[3].x - _imgQuad[2].x)));
	int h = static_cast<int>(std::lround(std::max(_imgQuad[2].y - _imgQuad[0].y, _imgQuad[3].y - _imgQuad[1].y)));
	if (alignEven)
	{
		w = makeEven(w);
		h = makeEven(h);
	}
	outW = w;
	outH = h;
}

static void computeCalibGeometry(int srcW, int srcH)
{
	_calibNeedCrop = false;
	_calibNeedScale = false;
	_calibCropX = 0;
	_calibCropY = 0;
	_calibCropW = srcW;
	_calibCropH = srcH;

	if (srcW <= 0 || srcH <= 0)
		return;

	// hyperhdr-calib: always output 1920x1080 (16:9) regardless of input size.
	_calibOutW = makeEven(1920);
	_calibOutH = makeEven(1080);
	_calibNeedScale = (_calibOutW != srcW) || (_calibOutH != srcH);
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

	const ControlList &md = request->metadata();
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

static bool fitShutterLogLinear(const std::vector<float> &shutters, const std::vector<float> &ys, float targetY, float &outShutter)
{
	// Fit log(y) = a*log(shutter) + b and solve for y=targetY.
	if (shutters.size() != ys.size() || shutters.empty())
		return false;
	if (targetY <= 0.0f)
		return false;

	double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
	int n = 0;
	for (size_t i = 0; i < shutters.size(); i++)
	{
		const float s = shutters[i];
		const float y = ys[i];
		if (!(s > 0.0f) || !(y > 1.0f) || y >= 254.5f)
			continue;
		const double x = std::log(static_cast<double>(s));
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
	const double sT = std::exp(xT);
	if (!std::isfinite(sT) || sT <= 0.0)
		return false;
	outShutter = static_cast<float>(sT);
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

// SensorTimestamp is CLOCK_BOOTTIME; compare against the same clock.
static inline int64_t bootNowNs()
{
	struct timespec ts;
	clock_gettime(CLOCK_BOOTTIME, &ts);
	return static_cast<int64_t>(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}

// --latency-stats: per-frame sensor-exposure -> send-complete latency percentiles.
static bool _latencyStats = false;

struct LatencyStats
{
	// Recorded only from the single worker thread that completes sends
	// (transform thread in fused mode, send thread otherwise) - no locking.
	std::vector<uint32_t> samplesUs;
	int64_t windowStartNs{0};
	uint64_t lastDropProcess{0};
	uint64_t lastDropSend{0};

	void record(int64_t sensorTsNs, const std::atomic<uint64_t> &dropProcess, const std::atomic<uint64_t> &dropSend)
	{
		const int64_t now = bootNowNs();
		if (windowStartNs == 0)
		{
			windowStartNs = now;
			samplesUs.reserve(512);
		}

		const int64_t d = now - sensorTsNs;
		if (sensorTsNs > 0 && d > 0 && d < 10000000000ll)
			samplesUs.push_back(static_cast<uint32_t>(d / 1000));

		if (now - windowStartNs < 1000000000ll)
			return;

		if (!samplesUs.empty())
		{
			auto pct = [&](double p) -> double
			{
				size_t idx = static_cast<size_t>(p * static_cast<double>(samplesUs.size() - 1) + 0.5);
				std::nth_element(samplesUs.begin(), samplesUs.begin() + idx, samplesUs.end());
				return static_cast<double>(samplesUs[idx]) / 1000.0;
			};
			const double p50 = pct(0.50);
			const double p95 = pct(0.95);
			const double p99 = pct(0.99);
			const auto [mn, mx] = std::minmax_element(samplesUs.begin(), samplesUs.end());
			const uint64_t dp = dropProcess.load(std::memory_order_relaxed);
			const uint64_t ds = dropSend.load(std::memory_order_relaxed);
			fprintf(stderr, "[lat] n=%zu sensor->sent ms: p50=%.2f p95=%.2f p99=%.2f min=%.2f max=%.2f dropP=%llu dropS=%llu\n",
					samplesUs.size(), p50, p95, p99,
					static_cast<double>(*mn) / 1000.0, static_cast<double>(*mx) / 1000.0,
					static_cast<unsigned long long>(dp - lastDropProcess),
					static_cast<unsigned long long>(ds - lastDropSend));
			lastDropProcess = dp;
			lastDropSend = ds;
		}
		samplesUs.clear();
		windowStartNs = now;
	}
};

static LatencyStats _latStats;

// Apply latency-first thread tuning: pin to a dedicated core (only when there
// are enough cores to spare core 0 for the kernel/IRQs + libcamera's internal
// pipeline thread) and raise to SCHED_FIFO. Degrades gracefully: a failure to
// set RT priority (missing CAP_SYS_NICE) warns once and keeps running normally.
static std::atomic<bool> _rtWarned{false};
static void applyThreadRealtime(const char *name, int core)
{
	const long n = sysconf(_SC_NPROCESSORS_ONLN);
	const int numCores = (n > 0) ? static_cast<int>(n) : 1;
	const bool pin = (numCores >= 4 && core >= 0 && core < numCores);

	if (pin)
	{
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(core, &set);
		if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0 && _verbose)
			std::cout << "WARNING: failed to pin " << name << " thread to core " << core << std::endl;
	}

	if (_noRealtime)
		return;

	struct sched_param sp;
	std::memset(&sp, 0, sizeof(sp));
	sp.sched_priority = _rtPriority;
	if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
	{
		bool expected = false;
		if (_rtWarned.compare_exchange_strong(expected, true))
			std::cout << "WARNING: could not set SCHED_FIFO (need CAP_SYS_NICE / root); running at normal priority" << std::endl;
	}
	else if (_verbose)
	{
		std::cout << "INFO: " << name << " thread SCHED_FIFO prio " << _rtPriority;
		if (pin)
			std::cout << " pinned to core " << core;
		std::cout << std::endl;
	}
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
static std::atomic<bool> _outputCropNv12ErrorPrinted{false};

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

// LED 3D-LUT calibration measurement mode (--ledcalib, doc/led-lut-calibration.md).
// In this mode camera frames are never streamed to HyperHDR; the only flatbuffer
// traffic is small solid-color frames driving all LEDs to one color.
static bool _ledCalibMode = false;
static LedCalib *_ledCalib = nullptr;
static std::string _ledCalibPhase;
static std::string _calibTvRoiStr;
static std::string _calibLedRoiStr;
static LedCalibConfig _ledCalibCfg;

static void ledCalibSendSolid(uint8_t r, uint8_t g, uint8_t b)
{
	// Runs on the flatbuffer send loop. 64x36 solid RGB: HyperHDR averages
	// regions, so every LED gets the same color regardless of layout.
	constexpr int kW = 64;
	constexpr int kH = 36;
	static std::vector<uint8_t> buf(static_cast<size_t>(kW) * kH * 3u);
	for (size_t i = 0; i < buf.size(); i += 3)
	{
		buf[i] = r;
		buf[i + 1] = g;
		buf[i + 2] = b;
	}
	if (flatbufTryConnect())
	{
		if (_flatbuf.setImage(buf.data(), kW, kH) < 0)
			std::cout << "ERROR: Flatbuffer write timeout\n";
	}
	else
	{
		bool expected = false;
		if (_flatbufConnErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
			std::cout << "ERROR: Flatbuffer connection timeout\n";
	}
}

// Optional backlog limits (0 = unlimited). When set, frames are dropped early to keep latency bounded.
static int _maxPendingProcess = 0;
static int _maxPendingSend = 0;
static std::atomic<uint64_t> _droppedProcess{0};
static std::atomic<uint64_t> _droppedSend{0};

static inline void latencyRecord(int64_t sensorTsNs)
{
	if (_latencyStats)
		_latStats.record(sensorTsNs, _droppedProcess, _droppedSend);
}

// Output crop ring buffer (used only when maxPendingSend > 0 to avoid per-frame allocations).
static RgbRingBuffer _cropOut;
static Nv12RingBuffer _cropNv12Out;

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

	// If gain is already at minimum but the image is too bright, switch to shutter calibration.
	if (needGain && (!needWb || stage == 2))
	{
		if (!_autoGainUseShutter && _gain <= 1.0f && (gainOkRef > targetGrey + tolGrey))
		{
			_autoGainUseShutter = true;
			_autoGainCurvePhase = 0;
			_autoGainCurveGains.clear();
			_autoGainCurveYs.clear();
			_autoGainCurveNextIdx = 0;
			_autoShutterCurvePhase = 0;
			_autoShutterCurveShutters.clear();
			_autoShutterCurveYs.clear();
			_autoShutterCurveNextIdx = 0;
			std::cout << "AutoGain: gain=1.0 but image is too bright; switching to shutter calibration" << std::endl;
		}
	}

	if (needWb && needGain)
	{
		if (!_autoGainUseShutter && stage == 2 && wbOk && gainOk2)
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
	else if (!_autoGainUseShutter && needGain && gainOk2)
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
		const bool enforceMaxIters = _autoGainUseShutter ? (_autoShutterCurvePhase == 2) : (_autoGainCurvePhase == 2);
		if (enforceMaxIters && iters >= _autoCalibMaxIters)
		{
			const char *stageName = _autoGainUseShutter ? "SHUTTER stage" : ((needWb && needGain) ? "GAIN stage" : "gain");
			std::cout << "AutoCalib reached max iterations (" << stageName << "): gain=" << _gain << " wbgain=" << _rGain << "," << _bGain
					  << " avg RGB=" << r << "," << g << "," << b << std::endl;
			imgOutExit(EXIT_SUCCESS);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}
	}

	// Shutter calibration (used when gain is at minimum and image is too bright).
	if (needGain && _autoGainUseShutter)
	{
		if (_autoShutterCurvePhase == 0)
		{
			_autoShutterCurvePhase = 1;
			_autoShutterCurveShutters.clear();
			_autoShutterCurveYs.clear();
			_autoShutterCurveNextIdx = 0;

			const float minUs = static_cast<float>(_autoShutterMinUs);
			const float curUs = std::max(minUs, static_cast<float>(_shutter));
			const float range = std::max(1.01f, _autoGainRange);
			const float minS = clampf(curUs / range, minUs, curUs);
			const float maxS = curUs;
			_autoShutterClampMin = minS;
			_autoShutterClampMax = maxS;
			const int samples = std::max(3, _autoGainSamples);
			_autoShutterCurveShutters.resize(static_cast<size_t>(samples));
			for (int i = 0; i < samples; i++)
			{
				const double t = (samples == 1) ? 0.0 : (static_cast<double>(i) / static_cast<double>(samples - 1));
				const double sLog = std::log(static_cast<double>(minS)) + t * (std::log(static_cast<double>(maxS)) - std::log(static_cast<double>(minS)));
				_autoShutterCurveShutters[static_cast<size_t>(i)] = static_cast<float>(std::exp(sLog));
			}
			// Force the first sample to be the current shutter for a quick datapoint.
			_autoShutterCurveShutters[0] = curUs;
			for (float &ss : _autoShutterCurveShutters)
				ss = clampf(ss, minUs, curUs);
			if (_verbose)
				std::cout << "AutoShutter: sampling " << samples << " points in [" << minS << "," << maxS << "]" << std::endl;
		}

		if (_autoShutterCurvePhase == 2)
		{
			if (gainOk2)
			{
				std::cout << "AutoShutter done: shutter=" << _shutter
						  << " avg RGB=" << r << "," << g << "," << b << std::endl;
				std::cout << "WARNING: Image was too bright at gain=1.0; consider lowering shutter speed." << std::endl;
				imgOutExit(EXIT_SUCCESS);
				_autoCalibBusy.store(false, std::memory_order_release);
				return;
			}

			const float ref = gainOkRef;
			if (ref > 1e-3f)
			{
				const float ratio = clampf(targetGrey / ref, 0.5f, 2.0f);
				const float curUs = std::max(static_cast<float>(_autoShutterMinUs), static_cast<float>(_shutter));
				const float nextUs = clampf(curUs * ratio, _autoShutterClampMin, _autoShutterClampMax);
				_shutter = static_cast<int32_t>(std::lround(nextUs));
			}
			_autoCalibIterCount.fetch_add(1, std::memory_order_acq_rel);
			if (_verbose)
				std::cout << "AutoShutter confirm adjust: set shutter=" << _shutter
						  << " (measured avg=" << gainOkRef << ", target " << targetGrey << ")" << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		// Record the current measurement for the current shutter.
		_autoShutterCurveYs.push_back(gainRef);
		if (_verbose)
			std::cout << "AutoShutter sample: shutter=" << _shutter << " ref=" << gainRef << " (avg RGB=" << r << "," << g << "," << b << ")" << std::endl;

		_autoShutterCurveNextIdx++;
		if (_autoShutterCurveNextIdx < _autoShutterCurveShutters.size())
		{
			_shutter = static_cast<int32_t>(std::lround(_autoShutterCurveShutters[_autoShutterCurveNextIdx]));
			if (_verbose)
				std::cout << "AutoShutter set next shutter=" << _shutter << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}

		float predicted = 0.0f;
		if (!fitShutterLogLinear(_autoShutterCurveShutters, _autoShutterCurveYs, targetGrey, predicted))
		{
			std::cout << "AutoShutter: fit failed" << std::endl;
			imgOutExit(EXIT_FAILURE);
			_autoCalibBusy.store(false, std::memory_order_release);
			return;
		}
		else
		{
			predicted = clampf(predicted, _autoShutterClampMin, _autoShutterClampMax);
			_shutter = static_cast<int32_t>(std::lround(predicted));
			_autoShutterCurvePhase = 2;
			if (_verbose)
				std::cout << "AutoShutter predicted shutter=" << _shutter << " (target " << targetGrey << ")" << std::endl;
			if (_autoCalibDelayMs > 0)
				_autoCalibNextAllowedNs.store(nowNs() + static_cast<int64_t>(_autoCalibDelayMs) * 1000000LL, std::memory_order_release);
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

	const int outW = (_transformForcedW > 0) ? _transformForcedW : _imgWidthToSend;
	const int outH = (_transformForcedH > 0) ? _transformForcedH : _imgHeightToSend;
	std::vector<uint8_t> rgb(static_cast<std::size_t>(outW) * static_cast<std::size_t>(outH) * 3u);
	GpuRgbTransform::InputFormat fmt = GpuRgbTransform::InputFormat::Packed3;
	if (_bppBgr == 4)
		fmt = _swapRb4 ? GpuRgbTransform::InputFormat::Xrgb8888 : GpuRgbTransform::InputFormat::Xbgr8888;

	if (_bppBgr == 4)
		_gpuRgb->transformDmabuf(fd, off, _strideBgr, fmt, rgb.data());
	else
		_gpuRgb->transform(img, _strideBgr, fmt, rgb.data());

	cv::Mat rgbMat(outH, outW, CV_8UC3, rgb.data());
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
	applyThreadRealtime("send", 2);
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

static bool cropNv12Into(const uint8_t *srcY, const uint8_t *srcUV,
						 int srcWidth, int srcHeight, int srcStrideY, int srcStrideUV,
						 int cropX, int cropY, int cropWidth, int cropHeight,
						 uint8_t *dstY, uint8_t *dstUV, int dstStrideY, int dstStrideUV)
{
	if (!dstY || !dstUV)
		return false;
	if (dstStrideY <= 0)
		dstStrideY = cropWidth;
	if (dstStrideUV <= 0)
		dstStrideUV = cropWidth;
	if (cropX < 0 || cropY < 0 || cropWidth <= 0 || cropHeight <= 0)
		return false;
	if (cropX + cropWidth > srcWidth || cropY + cropHeight > srcHeight)
		return false;
	// NV12 requires even alignment for chroma.
	if ((cropX & 1) || (cropY & 1) || (cropWidth & 1) || (cropHeight & 1))
		return false;

	for (int y = 0; y < cropHeight; y++)
	{
		const uint8_t *srcRow = srcY + ((cropY + y) * srcStrideY) + cropX;
		uint8_t *dstRow = dstY + (y * dstStrideY);
		std::memcpy(dstRow, srcRow, static_cast<size_t>(cropWidth));
	}

	const int uvH = cropHeight / 2;
	const int uvY = cropY / 2;
	for (int y = 0; y < uvH; y++)
	{
		const uint8_t *srcRow = srcUV + ((uvY + y) * srcStrideUV) + cropX;
		uint8_t *dstRow = dstUV + (y * dstStrideUV);
		std::memcpy(dstRow, srcRow, static_cast<size_t>(cropWidth));
	}

	return true;
}

static void flatBufSend(const uint8_t *img, int64_t sensorTsNs)
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
	latencyRecord(sensorTsNs);

	int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame sending was too slow: " << elapsedMicros << " us\n";
	}
}

static void flatBufSendNv12(const uint8_t *y, const uint8_t *uv, int strideY, int strideUV, int width, int height, int64_t sensorTsNs)
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
	latencyRecord(sensorTsNs);

	int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame sending was too slow: " << elapsedMicros << " us\n";
	}
}

// Connect + send a prebuilt request, without touching the in-flight counters.
// Used by the fused transform paths, which send inline on the transform thread
// and account the combined transform+send job via _pendingProcess.
static void sendRequestRaw(const uint8_t *data, uint32_t size)
{
	if (flatbufTryConnect())
	{
		if (_flatbuf.sendPrebuiltRequest(data, size) < 0)
			std::cout << "ERROR: Flatbuffer write timeout\n";
	}
	else
	{
		bool expected = false;
		if (_flatbufConnErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
			std::cout << "ERROR: Flatbuffer connection timeout\n";
	}
}

// Reusable per-thread builder for the fused transform+send paths: the request
// is consumed (sent) before the next frame is built, so the builder's buffer
// can be recycled. Avoids a frame-sized malloc/free per frame (allocation jitter).
// Capacity is reserved up front: the buffer must NOT grow between the
// CreateUninitializedVector calls and writing through the returned pointers.
static flatbuffers::FlatBufferBuilder &fusedBuilder(size_t needBytes)
{
	static thread_local flatbuffers::FlatBufferBuilder builder(1024);
	static thread_local size_t reserved = 0;
	if (needBytes > reserved)
	{
		// Force the internal buffer to the needed capacity in one allocation;
		// Clear() keeps the allocation. (PreAlign with alignment 1 does NOT
		// grow the buffer - it only writes padding, which is 0 bytes here.)
		builder.Clear();
		uint8_t *scratch = nullptr;
		builder.CreateUninitializedVector(needBytes, &scratch);
		builder.Clear();
		reserved = needBytes;
	}
	return builder;
}

struct BuiltRequest
{
	const uint8_t *data;
	uint32_t size;
};

static BuiltRequest buildRawImageRequestUninitialized(uint16_t width, uint16_t height, uint8_t **outRgb)
{
	const size_t imgBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
	flatbuffers::FlatBufferBuilder &builder = fusedBuilder(imgBytes + 1024u);
	builder.Clear();

	auto imgVec = builder.CreateUninitializedVector(imgBytes, outRgb);
	auto rawImg = hyperhdrnet::CreateRawImage(builder, imgVec, width, height);
	auto imageReq = hyperhdrnet::CreateImage(builder, hyperhdrnet::ImageType_RawImage, rawImg.Union(), -1);
	auto req = hyperhdrnet::CreateRequest(builder, hyperhdrnet::Command_Image, imageReq.Union());
	builder.Finish(req);

	return BuiltRequest{builder.GetBufferPointer(), builder.GetSize()};
}

static BuiltRequest buildNV12ImageRequestUninitialized(uint16_t width, uint16_t height, uint16_t strideY, uint16_t strideUV,
													   uint8_t **outY, uint8_t **outUV)
{
	const size_t yBytes = static_cast<size_t>(strideY) * static_cast<size_t>(height);
	const size_t uvBytes = static_cast<size_t>(strideUV) * (static_cast<size_t>(height) / 2u);
	flatbuffers::FlatBufferBuilder &builder = fusedBuilder(yBytes + uvBytes + 1024u);
	builder.Clear();

	auto yVec = builder.CreateUninitializedVector(yBytes, outY);
	auto uvVec = builder.CreateUninitializedVector(uvBytes, outUV);
	auto nv12Img = hyperhdrnet::CreateNV12Image(builder, yVec, uvVec, width, height, strideY, strideUV);
	auto imageReq = hyperhdrnet::CreateImage(builder, hyperhdrnet::ImageType_NV12Image, nv12Img.Union(), -1);
	auto req = hyperhdrnet::CreateRequest(builder, hyperhdrnet::Command_Image, imageReq.Union());
	builder.Finish(req);

	return BuiltRequest{builder.GetBufferPointer(), builder.GetSize()};
}
void *transformLoop(void *arg)
{
	// Pin + RT before creating the EGL context so all GL work runs on this thread.
	applyThreadRealtime("transform", 2);
	const int forcedW = (_transformForcedW > 0) ? _transformForcedW : 0;
	const int forcedH = (_transformForcedH > 0) ? _transformForcedH : 0;
	_transformation = new OpencvTransform(_imgWidth, _imgHeight, _imgQuad, &_imgWidthToSend, &_imgHeightToSend, _useNv12, _xformInterp, forcedW, forcedH);
	if (forcedW > 0 && forcedH > 0)
	{
		_imgWidthToSend = forcedW;
		_imgHeightToSend = forcedH;
	}

#ifdef HAVE_GPU_TRANSFORM
	if (_useNv12 && _useGpu)
		_gpuNv12 = new GpuNv12Transform(_imgWidth, _imgHeight, _imgQuad, _imgWidthToSend, _imgHeightToSend, _hyperhdrCalib);
	if (!_useNv12 && _useGpu)
		_gpuRgb = new GpuRgbTransform(_imgWidth, _imgHeight, _imgQuad, _imgWidthToSend, _imgHeightToSend, _hyperhdrCalib);
#endif
	_loopTr.exec();
	return 0;
}

static void transformImg(uint8_t *img, int64_t sensorTsNs)
{
	const int64_t startNs = nowNs();
	uint8_t *outRgb = nullptr;
	const uint16_t outW = static_cast<uint16_t>(_imgWidthToSend);
	const uint16_t outH = static_cast<uint16_t>(_imgHeightToSend);
	auto req = buildRawImageRequestUninitialized(outW, outH, &outRgb);
	_transformation->transformInto(img, _strideBgr, outRgb, static_cast<int>(outW) * 3);
	cpuProfileTick(_cpuRgbProf, "[cpuRgb]", static_cast<uint64_t>(nowNs() - startNs));

	// Fused send: deliver inline on this transform worker thread, then release
	// the in-flight slot only after the send completes so newer frames are
	// dropped while we're busy (latest-frame-wins, bounded latency).
	sendRequestRaw(req.data, req.size);
	latencyRecord(sensorTsNs);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transform+send was too slow: " << elapsedMicros << " us\n";
	}
}

static void transformImgNv12(uint8_t *y, uint8_t *uv, int64_t sensorTsNs)
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

	cv::remap(yIn, yOut, mapYx, mapYy, _xformInterp, cv::BORDER_CONSTANT);
	cv::remap(uvIn, uvOut, mapUVx, mapUVy, _xformInterp, cv::BORDER_CONSTANT);
	cpuProfileTick(_cpuNv12Prof, "[cpuNv12]", static_cast<uint64_t>(nowNs() - startNs));

	// Fused send: deliver inline on this transform worker thread, then release
	// the in-flight slot only after the send completes so newer frames are
	// dropped while we're busy (latest-frame-wins, bounded latency).
	sendRequestRaw(req.data, req.size);
	latencyRecord(sensorTsNs);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transform+send was too slow: " << elapsedMicros << " us\n";
	}
}

static void transformImgNv12GpuDmabuf(uint8_t *y, uint8_t *uv, int fdY, int offY, int fdUV, int offUV, int64_t sensorTsNs)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)y;
	(void)uv;
	(void)fdY;
	(void)offY;
	(void)fdUV;
	(void)offUV;
	(void)sensorTsNs;
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

	// Fused send: deliver inline on this transform worker thread, then release
	// the in-flight slot only after the send completes so newer frames are
	// dropped while we're busy (latest-frame-wins, bounded latency).
	sendRequestRaw(req.data, req.size);
	latencyRecord(sensorTsNs);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transform+send was too slow: " << elapsedMicros << " us\n";
	}
#endif
}

static void transformImgRgbGpu(uint8_t *img, int64_t sensorTsNs)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)img;
	(void)sensorTsNs;
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

	// Fused send: deliver inline on this transform worker thread, then release
	// the in-flight slot only after the send completes so newer frames are
	// dropped while we're busy (latest-frame-wins, bounded latency).
	sendRequestRaw(req.data, req.size);
	latencyRecord(sensorTsNs);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transform+send was too slow: " << elapsedMicros << " us\n";
	}
#endif
}

static void transformImgRgbGpuDmabuf(int fd, int offsetBytes, int64_t sensorTsNs)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)fd;
	(void)offsetBytes;
	(void)sensorTsNs;
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

	// Fused send: deliver inline on this transform worker thread, then release
	// the in-flight slot only after the send completes so newer frames are
	// dropped while we're busy (latest-frame-wins, bounded latency).
	sendRequestRaw(req.data, req.size);
	latencyRecord(sensorTsNs);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;
		std::cout << "WARNING: Frame transform+send was too slow: " << elapsedMicros << " us\n";
	}
#endif
}

static void processRequest(Request *request);

static void requestComplete(Request *request)
{
	if (request->status() == Request::RequestCancelled)
		return;

	handleAutoFocusMetadata(request);

	if (_directDispatch)
	{
		// One-time: pin + RT the libcamera camera-manager thread, which is now
		// the frame dispatch thread.
		static std::atomic<bool> tuned{false};
		if (!tuned.exchange(true, std::memory_order_acq_rel))
			applyThreadRealtime("camera-dispatch", 1);

		processRequest(request);
		return;
	}

	_loop.callLater(std::bind(&processRequest, request));
}

static void processRequest(Request *request)
{
	int64_t sensorTsNs = 0;
	if (_latencyStats)
	{
		auto tsOpt = request->metadata().get(controls::SensorTimestamp);
		if (tsOpt)
			sensorTsNs = *tsOpt;
	}

	const Request::BufferMap &buffers = request->buffers();
	for (auto bufferPair : buffers)
	{
		FrameBuffer *buffer = bufferPair.second;
		auto it = _mappedBuffers.find(buffer);
		if (it == _mappedBuffers.end() || it->second.empty())
			continue;

		if (_ledCalibMode)
		{
			// Measurement only: camera frames are never streamed to HyperHDR in
			// this mode (the calibration engine drives the LEDs with synthetic
			// solid frames instead).
			if (_ledCalib)
			{
				uint8_t *img = it->second[0].data();
				_ledCalib->processFrame(img, _imgWidthToSend, _imgHeightToSend, _strideBgr, _bppBgr, _swapRb4);
			}
			continue;
		}

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
						_loopTr.callLater([y, uv, fdY, offY, fdUV, offUV, sensorTsNs]()
										  { transformImgNv12GpuDmabuf(y, uv, fdY, offY, fdUV, offUV, sensorTsNs); });
					}
				}
				else
					_loopTr.callLater([y, uv, sensorTsNs]()
									  { transformImgNv12(y, uv, sensorTsNs); });
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
						_loopTr.callLater([fd, off, sensorTsNs]()
										  { transformImgRgbGpuDmabuf(fd, off, sensorTsNs); });
					}
					else
					{
						_loopTr.callLater([img, sensorTsNs]()
										  { transformImgRgbGpu(img, sensorTsNs); });
					}
				}
				else
					_loopTr.callLater([img, sensorTsNs]()
									  { transformImg(img, sensorTsNs); });
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
				if (_outputCropEn)
				{
					if (_maxPendingSend > 0)
					{
						auto view = _cropNv12Out.next();
						if (!cropNv12Into(y, uv, _imgWidthToSend, _imgHeightToSend, _strideY, _strideUV,
										  _outputCropX, _outputCropY, _outputCropWidth, _outputCropHeight,
										  view.y, view.uv, _outputCropWidth, _outputCropWidth))
						{
							bool expected = false;
							if (_outputCropNv12ErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
								std::cout << "ERROR: NV12 output-crop failed (check bounds and even alignment)" << std::endl;
							_pendingSend.fetch_sub(1, std::memory_order_acq_rel);
							continue;
						}
						uint8_t *outY = view.y;
						uint8_t *outUV = view.uv;
						int outStrideY = _outputCropWidth;
						int outStrideUV = _outputCropWidth;
						int outW = _outputCropWidth;
						int outH = _outputCropHeight;
						_loopFb.callLater([outY, outUV, outStrideY, outStrideUV, outW, outH, sensorTsNs]()
										  { flatBufSendNv12(outY, outUV, outStrideY, outStrideUV, outW, outH, sensorTsNs); });
					}
					else
					{
						std::vector<uint8_t> yCrop(static_cast<size_t>(_outputCropWidth) * static_cast<size_t>(_outputCropHeight));
						std::vector<uint8_t> uvCrop(static_cast<size_t>(_outputCropWidth) * static_cast<size_t>(_outputCropHeight) / 2u);
						if (!cropNv12Into(y, uv, _imgWidthToSend, _imgHeightToSend, _strideY, _strideUV,
										  _outputCropX, _outputCropY, _outputCropWidth, _outputCropHeight,
										  yCrop.data(), uvCrop.data(), _outputCropWidth, _outputCropWidth))
						{
							bool expected = false;
							if (_outputCropNv12ErrorPrinted.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
								std::cout << "ERROR: NV12 output-crop failed (check bounds and even alignment)" << std::endl;
							_pendingSend.fetch_sub(1, std::memory_order_acq_rel);
							continue;
						}
						int outW = _outputCropWidth;
						int outH = _outputCropHeight;
						_loopFb.callLater([yCrop = std::move(yCrop), uvCrop = std::move(uvCrop), outW, outH, sensorTsNs]() mutable
										  { flatBufSendNv12(yCrop.data(), uvCrop.data(), outW, outW, outW, outH, sensorTsNs); });
					}
				}
				else
				{
					_loopFb.callLater([y, uv, sensorTsNs]()
									  { flatBufSendNv12(y, uv, _strideY, _strideUV, _imgWidthToSend, _imgHeightToSend, sensorTsNs); });
				}
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
						_loopFb.callLater([croppedPtr, sensorTsNs]()
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
							latencyRecord(sensorTsNs);

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
						_loopFb.callLater([cropped = std::move(cropped), sensorTsNs]() mutable
										  {
						const int64_t startNs = nowNs();
						if (flatbufTryConnect())
						{
								if (_flatbuf.setImage(cropped.data(), _outputCropWidth, _outputCropHeight) < 0)
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
						latencyRecord(sensorTsNs);

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
					_loopFb.callLater([img, sensorTsNs]()
									  { flatBufSend(img, sensorTsNs); });
				}
			}
		}
	}

	request->reuse(Request::ReuseBuffers);

	// Only update controls per-frame when auto-calibration/focus features are enabled
	if (autoCalibEnabled() || _autoFocusOnce || _ledCalibMode)
	{
		ControlList &c = request->controls();

		// Apply focus controls (changes when auto-focus is active)
		applyFocusControls(c);

		// Apply saturation/contrast/gamma (changes when auto-saturation is active)
		c.set(controls::Saturation, _saturation);
		c.set(controls::Contrast, _contrast);
		if (_gamma > 0.0f)
			c.set(controls::Gamma, _gamma);

		if (autoCalibEnabled() || _ledCalibMode)
		{
			// Program per-request controls so that any updated _gain/_rGain/_bGain take effect
			// without restarting the camera. For --ledcalib this keeps exposure/WB/gain
			// locked: the camera must be a consistent comparator across all phases.
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
	options.add_options()																																																											  //
		("width,w", po::value<int>(&_width)->default_value(640), "Width")																																															  //
		("height,h", po::value<int>(&_height)->default_value(480), "Height")																																														  //
		("shutter,s", po::value<int32_t>(&_shutter)->default_value(16666), "Shutter speed in us")																																									  //
		("fps", po::value<float>(&_targetFps)->default_value(0.0f), "Target FPS limit (0=disabled, adds delay to maintain frame rate)")																																  //
		("gain,g", po::value<float>(&_gain)->default_value(1.0), "Camera gain")																																														  //
		("wbgain,b", po::value<std::string>(&_wbgainsStr)->default_value("1.0,1.0"), "White balance gains")																																							  //
		("origin,o", po::value<std::string>(&_origin)->default_value("camera"), "Name of the origin")																																								  //
		("connection,c", po::value<std::string>(&_conn)->default_value("127.0.0.1"), "Socket address (IP or Unix socket path)")																																		  //
		("port,p", po::value<int>(&_port)->default_value(19400), "TCP port")																																														  //
		("priority,r", po::value<int>(&_priority)->default_value(120), "Hyperion priority")																																											  //
		("transform,t", po::value<std::string>(&_transformStr), "Picture transformation in format x1,y1:x2,y2:x3,y3:x4,y4")																																			  //
		("output-crop", po::value<std::string>(&_outputCropStr), "Output crop in format x,y,width,height (applied after transform)")																																  //
		("nv12", po::bool_switch(&_useNv12)->default_value(false), "Enable NV12 format output (default is BGR888)")																																					  //
		("hyperhdr-calib", po::bool_switch(&_hyperhdrCalib)->default_value(false), "HyperHDR calibration mode: force NV12, ensure 16:9 and >=1920x1080")("cpuonly", po::bool_switch(&_cpuOnly)->default_value(false), "Force CPU transform even if GPU is available") //
		("verbose,v", po::bool_switch(&_verbose)->default_value(false), "Verbose output")																																											  //
		("imgout", po::value<std::string>(&_imgOutPath), "Capture one image (RGB) to file (png/bmp/...) and exit")																																					  //
		("imgout-crophelp", po::value<std::string>(&_imgOutCropHelpPath), "Capture image showing sensor area used at requested resolution (helps determine --crop coordinates)")																					  //
		("autowb", po::bool_switch(&_autoWb)->default_value(false), "One-shot auto white balance calibration (gray 127) and exit")																																	  //
		("autogain", po::bool_switch(&_autoGain)->default_value(false), "One-shot auto gain calibration (gray 127) and exit")																																		  //
		("autocalib", po::bool_switch(&_autoCalib)->default_value(false), "One-shot auto white balance + gain calibration (gray 127) and exit")																														  //
		("autosat", po::value<std::string>(&_autoSatTargetSpec)->implicit_value(""), "One-shot auto saturation calibration and exit (optional arg: R,G,B target; default 230,50,50)")																				  //
		("autosat-iters", po::value<int>(&_autoSatMaxIters)->default_value(24), "Auto saturation max iterations")																																					  //
		("autocalib-iters", po::value<int>(&_autoCalibMaxIters)->default_value(8), "Auto calibration max iterations")																																				  //
		("autocalib-delay-ms", po::value<int>(&_autoCalibDelayMs)->default_value(500), "Delay between auto calibration updates (ms)")																																  //
		("autocalib-tol-wb", po::value<float>(&_autoCalibTolWb)->default_value(2.0), "Auto WB convergence tolerance (abs channel delta: |R-G| and |B-G|)")																											  //
		("autocalib-tol-grey", po::value<float>(&_autoCalibTolGrey)->default_value(3.0), "Auto gain convergence tolerance (abs avg RGB error from 127)")																											  //
		("autocalib-gain-samples", po::value<int>(&_autoGainSamples)->default_value(8), "Auto gain samples (curve-fit)")																																			  //
		("autocalib-gain-range", po::value<float>(&_autoGainRange)->default_value(4.0), "Auto gain range multiplier (min=gain/range, max=gain*range)")																												  //
		("autofocus-once", po::bool_switch(&_autoFocusOnce)->default_value(false), "Run autofocus once at startup, print focus, then lock manual focus")																											  //
		("focus,f", po::value<float>(&_focus)->default_value(-1.0f), "Manual focus lens position (-1 keeps default)")																																				  //
		("saturation", po::value<float>(&_saturation)->default_value(1.0f), "Camera saturation")																																									  //
		("contrast", po::value<float>(&_contrast)->default_value(1.0f), "Camera contrast")																																											  //
		("gamma", po::value<float>(&_gamma)->default_value(0.0f), "Camera gamma (0.0=use camera default)")																																							  //
		("hflip", po::bool_switch(&_hflip)->default_value(false), "Horizontal flip (native camera transform)")																																						  //
		("vflip", po::bool_switch(&_vflip)->default_value(false), "Vertical flip (native camera transform)")																																						  //
		("rotate", po::value<int>(&_rotate)->default_value(0), "Rotation in degrees: 0, 90, 180, 270 (native camera transform)")																																	  //
		("crop", po::value<std::string>(&_cropStr), "Crop region in format x,y,width,height (native camera ScalerCrop)")																																			  //
		("version", po::bool_switch(&_version)->default_value(false), "Print version and exit")																																										  //
		("help", po::bool_switch(&_help)->default_value(false), "Print help")																																														  //
		("max-pending-process", po::value<int>(&_maxPendingProcess)->default_value(1), "Max in-flight transform jobs (0=unlimited, default 1=lowest latency, drops frames when busy)")																			  //
		("max-pending-send", po::value<int>(&_maxPendingSend)->default_value(1), "Max in-flight send jobs (0=unlimited, default 1=lowest latency, drops frames when busy)")																					  //
		("connect-timeout-ms", po::value<int>(&_flatbufConnectTimeoutMs)->default_value(250), "Socket connect timeout in ms (lower = lower latency when server is down)")																							  //
		("reconnect-interval-ms", po::value<int>(&_flatbufReconnectIntervalMs)->default_value(250), "Min time between reconnect attempts in ms when disconnected")																								  //
		("send-timeout-ms", po::value<int>(&_sendTimeoutMs)->default_value(100), "Socket send timeout in ms (SO_SNDTIMEO); abandons a frame if HyperHDR stalls")																									  //
		("no-realtime", po::bool_switch(&_noRealtime)->default_value(false), "Disable SCHED_FIFO real-time scheduling (still pins threads to cores)")																											  //
		("rt-priority", po::value<int>(&_rtPriority)->default_value(10), "SCHED_FIFO priority for capture/worker threads (1-99, default 10)")																													  //
		("latency-stats", po::bool_switch(&_latencyStats)->default_value(false), "Print per-second sensor->send latency percentiles (p50/p95/p99) to stderr")																									  //
		("no-direct-dispatch", po::bool_switch(&_noDirectDispatch)->default_value(false), "Dispatch frames via the capture event loop instead of directly from the camera thread")														  //
		("ledcalib", po::value<std::string>(&_ledCalibPhase)->implicit_value("all"), "One-shot LED 3D-LUT calibration measurement and exit (optional phase: all|white|tv|led; see doc/led-lut-calibration.md)")									  //
		("calib-tv-roi", po::value<std::string>(&_calibTvRoiStr), "LED calib: ROI inside the TV picture, x,y,w,h in raw full-frame coords (pick from --imgout without --crop)")																	  //
		("calib-led-roi", po::value<std::string>(&_calibLedRoiStr), "LED calib: ROI on the LED-lit wall, x,y,w,h in raw full-frame coords")																										  //
		("calib-out", po::value<std::string>(&_ledCalibCfg.outDir)->default_value("ledcalib-out"), "LED calib: output directory for measurement files")																							  //
		("ledcalib-led-grid", po::value<int>(&_ledCalibCfg.ledGrid)->default_value(8), "LED calib: LED sweep cube grid points per axis (2-16)")																									  //
		("ledcalib-tv-patches", po::value<int>(&_ledCalibCfg.tvPatchCount)->default_value(216), "LED calib: patches in the calib video TV sweep (1-216, must match the generated video)")														  //
		("ledcalib-frames", po::value<int>(&_ledCalibCfg.measureFrames)->default_value(20), "LED calib: frames averaged per measurement")																										  //
		("ledcalib-settle-ms", po::value<int>(&_ledCalibCfg.settleMs)->default_value(700), "LED calib: settle time before each measurement (HyperHDR smoothing + LED response)");

	std::signal(SIGPIPE, SIG_IGN); // Ignore SIGPIPE to prevent crashes on broken pipes

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, options), vm);
	po::notify(vm);

	_flatbuf.setConnectTimeoutMs(_flatbufConnectTimeoutMs);
	_flatbuf.setSendTimeoutMs(_sendTimeoutMs);

	if (_rtPriority < 1)
		_rtPriority = 1;
	if (_rtPriority > 99)
		_rtPriority = 99;

	// Lock all current and future memory: page faults on the RT capture/transform
	// threads show up directly as frame latency spikes. Non-fatal without
	// CAP_IPC_LOCK (running as non-root) - we just keep going unlocked.
	if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
	{
		if (_verbose)
			std::cout << "WARNING: mlockall failed (" << std::strerror(errno) << "); page faults may add jitter" << std::endl;
	}

	if (_hyperhdrCalib)
	{
		_useNv12 = true;
		_xformInterp = cv::INTER_LINEAR;
		if (_width < 1920)
			_width = 1920;
		if (_height < 1080)
			_height = 1080;
		if ((_width & 1) != 0)
			_width += 1;
		if ((_height & 1) != 0)
			_height += 1;
		if (_autoWb || _autoGain || _autoCalib || vm.count("autosat"))
		{
			std::cout << "ERROR: --hyperhdr-calib cannot be combined with auto calibration options" << std::endl;
			return EXIT_FAILURE;
		}
		if (vm.count("output-crop"))
		{
			std::cout << "ERROR: --hyperhdr-calib cannot be combined with --output-crop" << std::endl;
			return EXIT_FAILURE;
		}
	}
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

	if (_gain < 1.0f)
	{
		std::cout << "ERROR: gain must be >= 1.0" << std::endl;
		return EXIT_FAILURE;
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
			_transformUserEn = true;
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
		if (_useNv12)
			_cropNv12Out.reset(_outputCropWidth, _outputCropHeight, ringSize);
		else
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
		_autoGainUseShutter = false;
		_autoShutterCurvePhase = 0;
		_autoShutterCurveShutters.clear();
		_autoShutterCurveYs.clear();
		_autoShutterCurveNextIdx = 0;
		_autoShutterClampMin = 0.0f;
		_autoShutterClampMax = 0.0f;
		_autoCalibStage.store(0, std::memory_order_release);
		_autoCalibIterCount.store(0, std::memory_order_release);
		_autoCalibFrameCount.store(0, std::memory_order_release);
		_autoCalibNextAllowedNs.store(0, std::memory_order_release);
	}

	_ledCalibMode = (vm.count("ledcalib") > 0);
	if (_ledCalibMode)
	{
		if (autoCalibEnabled() || _hyperhdrCalib || !_imgOutPath.empty())
		{
			std::cout << "ERROR: --ledcalib must be run alone (no --autowb/--autogain/--autocalib/--autosat/--hyperhdr-calib/--imgout)" << std::endl;
			return EXIT_FAILURE;
		}
		if (_ledCalibPhase != "all" && _ledCalibPhase != "white" && _ledCalibPhase != "tv" && _ledCalibPhase != "led")
		{
			std::cout << "ERROR: --ledcalib phase must be one of: all, white, tv, led" << std::endl;
			return EXIT_FAILURE;
		}
		auto parseRoi = [](const std::string &s, LedCalibRoi &roi)
		{
			return sscanf(s.c_str(), "%d,%d,%d,%d", &roi.x, &roi.y, &roi.w, &roi.h) == 4 && roi.w > 0 && roi.h > 0;
		};
		if (!parseRoi(_calibTvRoiStr, _ledCalibCfg.tvRoi) || !parseRoi(_calibLedRoiStr, _ledCalibCfg.ledRoi))
		{
			std::cout << "ERROR: --ledcalib requires --calib-tv-roi and --calib-led-roi as x,y,w,h" << std::endl;
			return EXIT_FAILURE;
		}
		if (_ledCalibCfg.tvPatchCount < 1 || _ledCalibCfg.tvPatchCount > 216)
		{
			std::cout << "ERROR: --ledcalib-tv-patches must be 1-216 (3 base-6 index digits)" << std::endl;
			return EXIT_FAILURE;
		}
		if (_ledCalibCfg.ledGrid < 2 || _ledCalibCfg.ledGrid > 16)
		{
			std::cout << "ERROR: --ledcalib-led-grid must be 2-16" << std::endl;
			return EXIT_FAILURE;
		}
		if (_ledCalibCfg.measureFrames < 1)
			_ledCalibCfg.measureFrames = 1;
		if (_ledCalibCfg.settleMs < 0)
			_ledCalibCfg.settleMs = 0;
		if (!isatty(STDIN_FILENO))
		{
			std::cout << "ERROR: --ledcalib is interactive and needs a terminal on stdin" << std::endl;
			return EXIT_FAILURE;
		}
		_ledCalibCfg.phase = _ledCalibPhase;
		_ledCalibCfg.verbose = _verbose;

		// Measurement uses the raw full frame: RGB capture, no sensor crop, no warp.
		if (_useNv12 && _verbose)
			std::cout << "INFO: --ledcalib forces RGB output; ignoring --nv12" << std::endl;
		_useNv12 = false;
		if (_transformEn)
		{
			std::cout << "INFO: --ledcalib ignores --transform (ROIs are raw-frame rectangles)" << std::endl;
			_transformEn = false;
		}
		if (_cropEn)
		{
			std::cout << "INFO: --ledcalib ignores --crop (the LED wall must stay in view)" << std::endl;
			_cropEn = false;
		}
		if (_outputCropEn)
		{
			std::cout << "INFO: --ledcalib ignores --output-crop" << std::endl;
			_outputCropEn = false;
		}

		_ledCalib = new LedCalib(
			_ledCalibCfg,
			[](uint8_t r, uint8_t g, uint8_t b)
			{ _loopFb.callLater([r, g, b]() { ledCalibSendSolid(r, g, b); }); },
			[](int code)
			{ imgOutExit(code); });
	}

	// Validate mutually exclusive options
	if (_transformEn && _outputCropEn)
	{
		std::cout << "ERROR: --transform and --output-crop are mutually exclusive" << std::endl;
		std::cout << "       If using transform, adjust the transform coordinates instead of cropping" << std::endl;
		return EXIT_FAILURE;
	}

	if (_useNv12 && _outputCropEn)
	{
		if ((_outputCropX & 1) || (_outputCropY & 1) || (_outputCropWidth & 1) || (_outputCropHeight & 1))
		{
			std::cout << "ERROR: --output-crop requires even x,y,width,height when using NV12" << std::endl;
			return EXIT_FAILURE;
		}
	}

	// Direct dispatch only in pure streaming mode (see declaration comment).
	_directDispatch = !_noDirectDispatch && _imgOutPath.empty() && !autoCalibEnabled() && !_ledCalibMode;

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

	if (_hyperhdrCalib)
	{
		int baseW = _imgWidthToSend;
		int baseH = _imgHeightToSend;
		bool autoEnableTransform = false;
		if (_transformEn && _transformUserEn)
		{
			calcTransformOutputSize(baseW, baseH, _useNv12);
		}
		else
		{
			const double targetRatio = 16.0 / 9.0;
			const int reqW = _width;
			const int reqH = _height;
			const double ratio = (reqH > 0) ? (static_cast<double>(reqW) / static_cast<double>(reqH)) : 0.0;
			const bool ratioOk = std::fabs(ratio - targetRatio) <= 1e-6;
			const bool sizeOk = (reqW >= 1920) && (reqH >= 1080);
			if (!ratioOk || !sizeOk)
				autoEnableTransform = true;

			if (autoEnableTransform)
			{
				// Auto-enable transform (full-frame) for calibration output sizing.
				_transformEn = true;
				_imgQuad[0] = cv::Point2f(0.f, 0.f);
				_imgQuad[1] = cv::Point2f(static_cast<float>(_imgWidth), 0.f);
				_imgQuad[2] = cv::Point2f(0.f, static_cast<float>(_imgHeight));
				_imgQuad[3] = cv::Point2f(static_cast<float>(_imgWidth), static_cast<float>(_imgHeight));
				baseW = _imgWidth;
				baseH = _imgHeight;
			}
		}

		computeCalibGeometry(baseW, baseH);
		if (_transformEn)
		{
			_imgWidthToSend = _calibOutW;
			_imgHeightToSend = _calibOutH;
		}
		else
		{
			_calibOutW = baseW;
			_calibOutH = baseH;
			_calibNeedScale = false;
		}
		_calibAdjustEn = false;
		if (_transformEn)
		{
			_transformForcedW = _calibOutW;
			_transformForcedH = _calibOutH;
		}
		else
		{
			_transformForcedW = 0;
			_transformForcedH = 0;
		}

		// Re-evaluate GPU usage if transform was auto-enabled.
#ifdef HAVE_GPU_TRANSFORM
		_useGpu = (_transformEn && !_cpuOnly);
#else
		_useGpu = false;
#endif
		if (!_transformEn)
		{
			const double actualRatio = (_imgHeightToSend > 0) ? (static_cast<double>(_imgWidthToSend) / static_cast<double>(_imgHeightToSend)) : 0.0;
			const bool actualRatioOk = std::fabs(actualRatio - (16.0 / 9.0)) <= 1e-6;
			const bool actualSizeOk = (_imgWidthToSend >= 1920) && (_imgHeightToSend >= 1080);
			if ((!actualRatioOk || !actualSizeOk) && _verbose)
			{
				std::cout << "WARNING: hyperhdr-calib without transform but stream is "
						  << _imgWidthToSend << "x" << _imgHeightToSend
						  << " (may not meet 16:9 >=1920x1080)" << std::endl;
			}
		}

		if (_verbose)
		{
			std::cout << "INFO: hyperhdr-calib base=" << baseW << "x" << baseH
					  << " out=" << _calibOutW << "x" << _calibOutH
					  << " interp=" << _xformInterp
					  << " xform=" << (_transformEn ? "on" : "off") << std::endl;
		}
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
			if (memory == MAP_FAILED)
			{
				std::cout << "ERROR: mmap failed: " << std::strerror(errno) << std::endl;
				return EXIT_FAILURE;
			}
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

	// Run thread for flatbuffer. In transform mode the send is fused onto the
	// transform thread (sent inline), so the dedicated send thread is only needed
	// for the no-transform streaming path.
	if (_imgOutPath.empty() && !_transformEn)
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

	// This (main) thread runs the camera capture/dispatch loop: pin + RT it.
	applyThreadRealtime("capture", 1);

	// Run an EventLoop for camera
	_loop.exec();

	// Clean Up
	_camera->stop();
	if (_ledCalib)
	{
		// Joins the keyboard thread and restores terminal raw mode.
		delete _ledCalib;
		_ledCalib = nullptr;
	}
	allocator->free(stream);
	delete allocator;
	if (_transformation)
	{
		delete _transformation;
		_transformation = nullptr;
	}
#ifdef HAVE_GPU_TRANSFORM
	if (_gpuNv12)
	{
		delete _gpuNv12;
		_gpuNv12 = nullptr;
	}
	if (_gpuRgb)
	{
		delete _gpuRgb;
		_gpuRgb = nullptr;
	}
#endif
	_camera->release();
	_camera.reset();
	cm->stop();

	return EXIT_SUCCESS;
}
