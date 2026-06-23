#ifndef __LED_CALIB_H__
#define __LED_CALIB_H__

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// One-shot LED 3D-LUT calibration measurement engine (--ledcalib).
// Design + procedure: doc/led-lut-calibration.md
//
// Runs entirely on the main event-loop thread (direct dispatch is disabled in
// this mode); the only cross-thread traffic is the keyboard reader pushing
// keys into a mutex-guarded queue and the send callback posting to the
// flatbuffer send loop.

struct LedCalibRoi
{
	int x = 0;
	int y = 0;
	int w = 0;
	int h = 0;
};

class LedCalibWeb;

struct LedCalibConfig
{
	LedCalibRoi tvRoi;	// inside the TV picture, raw full-frame coords
	LedCalibRoi ledRoi; // on the wall lit by the LEDs, raw full-frame coords
	std::string outDir = "ledcalib-out";
	std::string phase = "tv"; // tv | led | shading | gamma | white
	int ledGrid = 8;		   // LED sweep cube grid points per axis
	int ledGrayRamp = 17;	   // extra gray ramp steps in the LED sweep
	int tvPatchCount = 216;	   // patches in the TV sweep (set to tvGrid^3)
	int measureFrames = 20;	   // frames averaged per measurement
	int settleMs = 700;		   // LED + HyperHDR smoothing settle before measuring
	// Web only: after the user taps 'begin' (which puts the page fullscreen),
	// hold the first patch back this long so the browser's transient "press Esc
	// to exit fullscreen" prompt clears before it can cover the patch. 0 = off.
	int startDelayMs = 5000;
	// TV-patch display is web-only.
	bool tvWeb = false; // drive the TV via the browser page
	int tvGrid = 6;		// TV patch cube grid points per axis
	// Configured window-patch geometry for the measurement sweeps. The Phase 0
	// white field overrides this to fullscreen; the sweeps restore it.
	double patchPct = 10.0;
	double patchX = -1.0;
	double patchY = -1.0;
	// Shading (ALSC) calibration (phase "shading"): full-screen safe-gray field,
	// sample a grid across the TV quad to map lens + panel non-uniformity.
	int shadingLevel = 180;	 // safe gray code value (avoid clip/ABL)
	int shadingGridU = 24;	 // grid columns across the TV (output u)
	int shadingGridV = 14;	 // grid rows down the TV (output v)
	bool haveHomography = false;
	double invHomography[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // unit-square (u,v in [0,1]) -> input px
	// Shading gain at the TV ROI's output position (from --shading-map), applied
	// to TV-side measurements so the LUT-calibration input domain matches the
	// shaded production output. {1,1,1} = no shading. tv/gamma phases only.
	double tvShadeGain[3] = {1.0, 1.0, 1.0};
	// Gamma ramp probe (phase "gamma"): R/G/B/gray ramps, measure the camera's
	// response to spot crush/clip. Always web-driven.
	bool gammaRaw = false; // measure the raw Bayer stream instead of ISP RGB
	int gammaStep = 1;	   // ramp step in code values (1 = every level 0..255)
	// Resolution the TV ROI was picked at (processed --imgout); used to scale
	// the ROI into the raw stream's native dimensions for --gamma-raw.
	int gammaRefW = 0;
	int gammaRefH = 0;
	bool verbose = false;
};

class LedCalib
{
public:
	using SendLedFn = std::function<void(uint8_t r, uint8_t g, uint8_t b)>;
	using ExitFn = std::function<void(int code)>;

	// web may be null (calib-video path). When set, phases 0/1 are driven
	// through the page instead of stdin + camera index decoding.
	LedCalib(const LedCalibConfig &cfg, SendLedFn sendLed, ExitFn exitFn, LedCalibWeb *web = nullptr);
	~LedCalib();

	LedCalib(const LedCalib &) = delete;
	LedCalib &operator=(const LedCalib &) = delete;

	/// Feed one captured raw frame (event-loop thread). bpp 3 = RGB byte
	/// order, bpp 4 = RGBX, or BGRX when swapRb is set (matches the
	/// conventions used by the autocalib paths).
	void processFrame(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb);

	/// Feed one Bayer frame for the --gamma-raw run. fourcc is the libcamera
	/// pixel-format fourcc (e.g. SRGGB10_CSI2P) used to pick the unpack +
	/// Bayer phase. width/height are the raw stream dimensions.
	void processRawFrame(const uint8_t *img, int width, int height, int strideBytes, uint32_t fourcc);

	/// Unit-test the RAW10 CSI2P unpack + Bayer-plane averaging on a synthetic
	/// frame. Returns true on PASS. Static so it needs no camera/web.
	static bool gammaRawSelftest();

private:
	enum class Phase
	{
		WhiteWait,
		White,
		TvWait,
		Tv,
		LedWait,
		LedBaseline,
		LedSweep,
		GammaWait,
		Gamma,
		ShadingWait,
		Shading,
		Done
	};

	struct RoiSample
	{
		double mean[3] = {0.0, 0.0, 0.0}; // R,G,B
		double sstd[3] = {0.0, 0.0, 0.0}; // spatial std inside the ROI
	};

	// Averages per-frame ROI means over a fixed number of frames.
	// Index 0 = TV ROI, index 1 = LED ROI.
	struct Measurement
	{
		bool active = false;
		int target = 0;
		int n = 0;
		double sum[2][3] = {};
		double sumSq[2][3] = {};
		double sstdSum[2][3] = {};

		void start(int frames);
		void add(const RoiSample &tv, const RoiSample &led);
		bool done() const { return active && n >= target; }
		void finish(double mean[2][3], double tstd[2][3], double sstd[2][3]);
	};

	struct SweepRecord
	{
		int idx = -1;			   // TV sweep: patch index
		uint8_t drive[3] = {};	   // LED sweep: drive value sent (TV: commanded patch RGB)
		double mean[2][3] = {};	   // [tv|led][R,G,B]
		double tstd[2][3] = {};	   // temporal std of per-frame means
		double sstd[2][3] = {};	   // average spatial std
		int frames = 0;
	};

	RoiSample sampleRoi(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb,
						const LedCalibRoi &roi) const;

	void enterPhase(Phase p, int64_t nowMs);
	// Enter an active sweep phase after the 'begin' tap, deferred by
	// _cfg.startDelayMs the first time so the fullscreen prompt clears.
	void beginActivePhase(Phase active, int64_t nowMs);
	void tickStartDelay(int64_t nowMs);
	void handleKeys(int64_t nowMs);
	void whiteKey(char c, int64_t nowMs);
	void finishTvPhase(int64_t nowMs);
	// Web TV-patch path.
	void buildTvPlan();
	void showTvPatch(int64_t nowMs);
	void runTvWeb(const RoiSample &tv, const RoiSample &led, int64_t nowMs);
	void webRefreshWhite();

	// Gamma ramp probe.
	void buildGammaPlan();
	void showGammaPatch(int64_t nowMs);
	void runGamma(const RoiSample &meas, int64_t nowMs);
	void writeGammaCsv();

	// Shading (ALSC) calibration.
	void runShading(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb, int64_t nowMs);
	void writeShadingMap();
	RoiSample sampleRoiBayer(const uint8_t *img, int width, int height, int strideBytes,
							 uint32_t fourcc, const LedCalibRoi &roi) const;
	void runLed(const RoiSample &tv, const RoiSample &led, int64_t nowMs);
	void buildLedPlan();

	void setLed(uint8_t r, uint8_t g, uint8_t b, int64_t nowMs);

	bool ensureOutDir();
	void writeTvSweep();
	void writeLedSweep();
	void writeBaseline();
	void writeMeta();
	void finishAll(int code);

	void keyboardLoop();

	LedCalibConfig _cfg;
	SendLedFn _sendLed;
	ExitFn _exit;
	LedCalibWeb *_web = nullptr;

	Phase _phase = Phase::Done;
	int64_t _t0Ms = -1;
	bool _finished = false;
	bool _roiChecked = false;
	bool _webStarted = false;	// page connected + user pressed start
	uint32_t _webSeq = 0;		// monotonic display sequence for SSE/ack
	int64_t _lastUrlPrintMs = 0; // throttle the "open this URL" prompt
	// Deferred first-patch start (browser fullscreen-prompt grace). _pendingPhase
	// is the active phase to enter once _startDelayUntilMs passes; applied once.
	Phase _pendingPhase = Phase::Done;
	int64_t _startDelayUntilMs = -1;
	bool _startDelayDone = false;

	// keyboard reader
	std::thread _kbThread;
	std::atomic<bool> _kbStop{false};
	std::mutex _kbMutex;
	std::deque<char> _keys;

	// LED drive state
	uint8_t _drive[3] = {0, 0, 0};
	int64_t _lastSendMs = -1000000;

	Measurement _meas;

	// white phase (manual eye-match): LED-drive nudge step only
	int _whiteStep = 8;

	// TV sweep
	std::vector<SweepRecord> _tvRecords;
	int _tvDecoded = 0;
	std::vector<std::array<uint8_t, 3>> _tvPlan;
	size_t _tvPlanPos = 0;
	uint32_t _tvSeq = 0;		   // seq of the currently shown patch
	int64_t _tvAckWaitStartMs = -1;
	int64_t _tvSettleUntilMs = -1; // <0 = waiting for the page to ack the paint

	// Gamma ramp probe
	struct GammaRecord
	{
		int chan = 0;  // 0=R 1=G 2=B 3=gray
		int level = 0; // commanded code value 0..255
		double mean[3] = {};
		double tstd[3] = {};
		double sstd[3] = {};
		int frames = 0;
	};
	std::vector<std::array<int, 4>> _gammaPlan; // {chan, r, g, b}
	std::vector<GammaRecord> _gammaRecords;
	size_t _gammaPos = 0;
	LedCalibRoi _rawRoi; // tvRoi scaled into raw-stream coords (--gamma-raw)

	// Shading calibration: per-cell RGB accumulators over the TV-quad grid.
	std::vector<double> _shadeSum;	  // size gridU*gridV*3
	std::vector<uint32_t> _shadeCount; // size gridU*gridV
	int _shadeFrames = 0;
	int64_t _shadeSettleUntilMs = -1;

	// LED sweep
	std::vector<std::array<uint8_t, 3>> _ledPlan;
	size_t _ledPlanPos = 0;
	int64_t _settleUntilMs = -1;
	SweepRecord _baseline{};
	bool _haveBaseline = false;
	std::vector<SweepRecord> _ledRecords;
};

#endif /* __LED_CALIB_H__ */
