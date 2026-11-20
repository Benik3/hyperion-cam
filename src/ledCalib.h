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

// One RAW Bayer frame captured in the SAME request as the ISP frame (dual
// stream, production sensor mode). Linear sensor data: no tone curve, no CCM,
// no WB, no lens shading — the ground truth the ISP readings are checked
// against (doc/led-lut-calibration.md, "RAW measurement").
struct LedCalibRawFrame
{
	const uint8_t *data = nullptr;
	int width = 0;
	int height = 0;
	int stride = 0;
	uint32_t fourcc = 0;  // libcamera base fourcc ('BG10' = BGGR, ...); selects the 2x2 phase
	bool packed = true;	  // MIPI CSI2 packed (4 px / 5 bytes); else 16-bit little-endian samples
	int bits = 10;		  // sample depth
	double black[3] = {64.0, 64.0, 64.0}; // per R,G,B, in sample units (SensorBlackLevels)
	// ISP-stream pixel -> raw pixel (ScalerCrop + sensor-mode crop + binning):
	// rawX = ax + bx * ispX, rawY = ay + by * ispY.
	double ax = 0.0, bx = 1.0, ay = 0.0, by = 1.0;
	double fullScale() const { return static_cast<double>((1 << bits) - 1); }
};

// Per-frame camera metadata relevant to measurement (0 = not reported).
struct LedCalibFrameMeta
{
	int64_t exposureUs = 0;
	double analogueGain = 0.0;
	double digitalGain = 0.0;
	double colourGains[2] = {0.0, 0.0};
	double ccm[9] = {};
	bool haveCcm = false;
};

struct LedCalibConfig
{
	LedCalibRoi tvRoi;	// inside the TV picture, raw full-frame coords
	LedCalibRoi ledRoi; // on the wall lit by the LEDs, raw full-frame coords
	std::string outDir = "ledcalib-out";
	std::string phase = "tv"; // tv | led | shading | gamma | white
	int ledGrid = 8;		   // LED sweep cube grid points per axis
	int ledGrayRamp = 17;	   // extra gray ramp steps in the LED sweep
	int ledAxisSteps = 24;	   // extra per-channel axis ramp, edge-dense near 0/255 (0 = off)
	int measureFrames = 10;	   // frames averaged per measurement
	// Settle before each measurement. Short because nothing slow has to converge:
	// exposure/gain/AWB are locked, and HyperHDR LED smoothing must be DISABLED
	// for calibration (see doc/calibration-guide.md) so the LEDs jump instantly —
	// the only waits are the TV panel showing the patch + the camera pipeline
	// flushing stale frames. Every measurement additionally requires a minimum
	// of fresh frames after the patch ack / LED drive change (guards low
	// camera fps, where in-flight frames outlive the settle time).
	int settleMs = 100;
	// Web only: after the user taps 'begin' (which puts the page fullscreen),
	// hold the first patch back this long so the browser's transient "press Esc
	// to exit fullscreen" prompt clears before it can cover the patch. 0 = off.
	int startDelayMs = 5000;
	// TV-patch display is web-only.
	bool tvWeb = false; // drive the TV via the browser page
	int tvGrid = 6;		// TV patch cube grid points per axis
	// Reference white level for the whole calibration. In the manual "white"
	// phase it is the level driven to BOTH the full-screen TV reference and the
	// LEDs (then nudgeable). It is ALSO injected as a guaranteed neutral patch
	// into the tv/led sweeps and becomes the fit's von-Kries white anchor level
	// (see writeMeta + ledCalibFit deriveAnchorFromSweeps), so the metameric
	// white correction is computed where the eye white-balance was matched.
	// Default 127 (mid-gray) — where the WB eye-match is most trustworthy and
	// clipping least likely; 255 = full white.
	int whiteLevel = 127;
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
	// A RAW stream is configured next to the ISP stream (tv-raw, response, led,
	// gamma --gamma-raw): processFrame() then receives the matching raw frame.
	bool captureRaw = false;
	// Response phase (shutter sweep): base exposure (= production --shutter),
	// the TV patch level shown and the LED drive held while the shutter steps.
	int baseExposureUs = 0;
	int responseTvLevel = 180;
	int responseLedLevel = 255;
	// Provenance recorded in meta_<phase>.json so the fit can refuse to mix
	// sweeps taken with different camera settings / tuning files.
	std::string tuningFile;	   // LIBCAMERA_RPI_TUNING_FILE in effect ("" = system default)
	std::string tuningHash;	   // FNV-1a of its contents ("" = unknown)
	std::string cameraJson;	   // preformatted JSON object of the camera args (shutter, gains, ...)
	bool verbose = false;
};

class LedCalib
{
public:
	using SendLedFn = std::function<void(uint8_t r, uint8_t g, uint8_t b)>;
	using ExitFn = std::function<void(int code)>;
	// Request a per-frame exposure time (us; 0 = back to the base --shutter).
	using SetExposureFn = std::function<void(int us)>;

	// web may be null (calib-video path). When set, phases 0/1 are driven
	// through the page instead of stdin + camera index decoding.
	LedCalib(const LedCalibConfig &cfg, SendLedFn sendLed, ExitFn exitFn, LedCalibWeb *web = nullptr);
	~LedCalib();

	LedCalib(const LedCalib &) = delete;
	LedCalib &operator=(const LedCalib &) = delete;

	void setExposureFn(SetExposureFn fn) { _setExposure = std::move(fn); }

	/// Feed one captured ISP frame (event-loop thread). bpp 3 = RGB byte
	/// order, bpp 4 = RGBX, or BGRX when swapRb is set (matches the
	/// conventions used by the autocalib paths). raw = the Bayer frame from the
	/// same request (null when no RAW stream is configured); meta = that
	/// request's metadata (null = none).
	void processFrame(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb,
					  const LedCalibRawFrame *raw = nullptr, const LedCalibFrameMeta *meta = nullptr);

	/// Unit-test the RAW unpack (CSI2P + 16-bit), Bayer-plane averaging, black
	/// subtraction and ISP->raw ROI mapping on synthetic frames. Returns true on
	/// PASS. Static so it needs no camera/web.
	static bool gammaRawSelftest();

	/// Create the output directory (recursively) if missing. Returns false and
	/// prints an error on failure. Static so the CLI can fail-fast at startup,
	/// before the long sweep, rather than only when the first file is written.
	static bool ensureDir(const std::string &dir);

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
		GammaLedsWait,
		GammaLeds,
		ShadingWait,
		Shading,
		TvRawWait,
		TvRaw,
		ResponseWait,
		Response,
		Done
	};

	struct RoiSample
	{
		double mean[3] = {0.0, 0.0, 0.0}; // R,G,B
		double sstd[3] = {0.0, 0.0, 0.0}; // spatial std inside the ROI
		double sat[3] = {0.0, 0.0, 0.0};  // fraction of pixels at the ceiling (ISP 255 / sensor full scale)
	};

	// Measurement slots: ISP TV/LED ROI, then the same ROIs on the RAW frame.
	static constexpr int kSlots = 4;
	static constexpr int kTv = 0, kLed = 1, kTvRaw = 2, kLedRaw = 3;

	// Averages per-frame ROI means over a fixed number of frames.
	struct Measurement
	{
		bool active = false;
		int target = 0;
		int n = 0;
		double sum[kSlots][3] = {};
		double sumSq[kSlots][3] = {};
		double sstdSum[kSlots][3] = {};
		double satSum[kSlots][3] = {};
		double expSum = 0.0; // per-frame reported exposure (us), averaged
		double gainSum = 0.0;

		void start(int frames);
		void add(const RoiSample &tv, const RoiSample &led); // ISP slots only
		void add(const RoiSample &tv, const RoiSample &led, const RoiSample &tvRaw, const RoiSample &ledRaw,
				 const LedCalibFrameMeta *meta);
		bool done() const { return active && n >= target; }
		void finish(double mean[kSlots][3], double tstd[kSlots][3], double sstd[kSlots][3],
					double sat[kSlots][3] = nullptr);
		double exposureUs() const { return n ? expSum / n : 0.0; }
		double analogueGain() const { return n ? gainSum / n : 0.0; }
	};

	struct SweepRecord
	{
		int idx = -1;				 // TV sweep: patch index
		uint8_t drive[3] = {};		 // LED sweep: drive value sent (TV: commanded patch RGB)
		double mean[kSlots][3] = {}; // [slot][R,G,B]
		double tstd[kSlots][3] = {}; // temporal std of per-frame means
		double sstd[kSlots][3] = {}; // average spatial std
		double sat[kSlots][3] = {};	 // average ceiling fraction
		double windowPct = 0.0;		 // TV patch area % (tv-raw)
		double exposureUs = 0.0;	 // measured (metadata) exposure
		double analogueGain = 0.0;
		double targetExposureUs = 0.0; // response phase: commanded exposure
		int64_t tMs = 0;			   // response phase: time since phase start
		int frames = 0;
	};

	RoiSample sampleRoi(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb,
						const LedCalibRoi &roi) const;
	// ROI in ISP-stream coords -> the same scene area in raw-frame coords.
	static LedCalibRoi mapRoiToRaw(const LedCalibRoi &roi, const LedCalibRawFrame &raw);
	static RoiSample sampleRoiBayer(const LedCalibRawFrame &raw, const LedCalibRoi &rawRoi);

	void enterPhase(Phase p, int64_t nowMs);
	// Enter an active sweep phase after the 'begin' tap, deferred by
	// _cfg.startDelayMs the first time so the fullscreen prompt clears.
	void beginActivePhase(Phase active, int64_t nowMs);
	void tickStartDelay(int64_t nowMs);
	void handleKeys(int64_t nowMs);
	void whiteKey(char c, int64_t nowMs);
	void finishTvPhase();
	// Settle/measure gates shared by the sweeps. Both wait the settle time AND
	// a minimum of fresh camera frames (flushes stale in-flight frames at low
	// camera fps), then start the measurement; true = feed _meas this frame.
	// webPatchReady() additionally waits for the page's paint-ack first (with
	// timeout -> finishAll); ledPatchReady() keys off the LED drive change.
	bool webPatchReady(const char *what, std::size_t pos, std::size_t total, int64_t nowMs);
	bool ledPatchReady(int64_t nowMs);
	// (Re)start the LED-drive settle window; call right after setLed().
	void ledSettleStart(int64_t nowMs, int minSettleMs = 0);
	// Finish _meas into a GammaRecord for ROI slot `roi` and advance _gammaPos.
	void pushGammaRecord(int roi, const char *label);
	// Shared gamma/gamma_leds completion; true = plan done (files written / failed).
	bool gammaFinishIfDone();
	// Web TV-patch path.
	void buildTvPlan();
	void showTvPatch(int64_t nowMs);
	void runTvWeb(const RoiSample &tv, const RoiSample &led, int64_t nowMs);
	void webRefreshWhite();

	// Gamma ramp probe.
	void buildGammaPlan();
	void showGammaPatch(int64_t nowMs);
	void runGamma(const RoiSample &meas, int64_t nowMs);
	// Same R/G/B/gray ramp, but driven into the LEDs (flatbuffer) like the LED
	// sweep — page stays black, measures the LED ROI → gamma_leds.csv.
	void runGammaLeds(const RoiSample &led, int64_t nowMs);
	void writeGammaCsv();
	void reportGammaLimits(); // pre-flight: warn on per-channel camera clip/crush in the ramps

	// Shading (ALSC) calibration.
	void runShading(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb, int64_t nowMs);
	void writeShadingMap();
	void runLed(const RoiSample &tv, const RoiSample &led, const RoiSample &tvRaw, const RoiSample &ledRaw,
				const LedCalibFrameMeta *meta, int64_t nowMs);
	void buildLedPlan();

	// TV RAW probe (phase "tv-raw"): TV colour patches measured on the RAW frame
	// -> tv_raw.csv, the input of --make-tuning's TV-fitted colour matrix.
	void buildTvRawPlan();
	void showTvRawPatch(int64_t nowMs);
	void runTvRaw(const RoiSample &tv, const RoiSample &tvRaw, const LedCalibFrameMeta *meta, int64_t nowMs);
	void writeTvRaw();

	// Camera response (phase "response"): shutter sweep over a fixed TV patch +
	// LED white -> camera_response.csv (ISP vs RAW vs exposure).
	void buildResponsePlan();
	void startResponseStep(int64_t nowMs);
	void runResponse(const RoiSample &tv, const RoiSample &led, const RoiSample &tvRaw, const RoiSample &ledRaw,
					 const LedCalibFrameMeta *meta, int64_t nowMs);
	void writeResponse();

	// meta_<phase>.json: phase, ROIs, camera args, tuning file + hash, metadata.
	void writePhaseMeta(const LedCalibFrameMeta *meta);

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
	SetExposureFn _setExposure;
	LedCalibWeb *_web = nullptr;

	Phase _phase = Phase::Done;
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
	std::vector<std::array<uint8_t, 3>> _tvPlan;
	size_t _tvPlanPos = 0;
	uint32_t _tvSeq = 0;		   // seq of the currently shown patch
	int64_t _tvAckWaitStartMs = -1;
	int64_t _tvSettleUntilMs = -1; // <0 = waiting for the page to ack the paint
	int _tvSettleFrames = 0;	   // fresh camera frames seen since the ack (settle guard)

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

	bool _rawWarned = false;	   // "no RAW frame" warning printed once
	LedCalibFrameMeta _lastMeta{}; // latest frame metadata (for meta_<phase>.json)
	bool _haveMeta = false;

	// TV RAW probe
	struct TvRawPatch
	{
		std::array<uint8_t, 3> rgb;
		double windowPct; // patch area %
		bool peak;		  // small centered peak-white window (sampled on _peakRoi)
	};
	LedCalibRoi _peakRoi;	 // centre of the peak-white window, in ISP-stream coords
	bool _peakRoiOk = false; // false = fall back to the TV ROI (with a warning)
	bool computePeakRoi(int width, int height, const LedCalibRawFrame *raw);
	std::vector<TvRawPatch> _tvRawPlan;
	size_t _tvRawPos = 0;
	std::vector<SweepRecord> _tvRawRecords;

	// Camera response (shutter sweep)
	std::vector<int> _respPlan; // exposure per step (us)
	size_t _respPos = 0;
	int64_t _respStartMs = 0;
	int64_t _respStepStartMs = 0;
	int _respGoodFrames = 0; // fresh frames already at the target exposure
	std::vector<SweepRecord> _respRecords;

	// Shading calibration: per-cell RGB accumulators over the TV-quad grid.
	std::vector<double> _shadeSum;	  // size gridU*gridV*3
	std::vector<uint32_t> _shadeCount; // size gridU*gridV
	int _shadeFrames = 0;
	int64_t _shadeSettleUntilMs = -1;

	// LED sweep
	std::vector<std::array<uint8_t, 3>> _ledPlan;
	size_t _ledPlanPos = 0;
	int64_t _settleUntilMs = -1;
	int _ledSettleFrames = 0; // fresh frames since the last LED drive change (settle guard)
	SweepRecord _baseline{};
	bool _haveBaseline = false;
	std::vector<SweepRecord> _ledRecords;
};

#endif /* __LED_CALIB_H__ */
