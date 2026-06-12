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

struct LedCalibConfig
{
	LedCalibRoi tvRoi;	// inside the TV picture, raw full-frame coords
	LedCalibRoi ledRoi; // on the wall lit by the LEDs, raw full-frame coords
	std::string outDir = "ledcalib-out";
	std::string phase = "all"; // all | white | tv | led
	int ledGrid = 8;		   // LED sweep cube grid points per axis
	int ledGrayRamp = 17;	   // extra gray ramp steps in the LED sweep
	int tvPatchCount = 216;	   // patches in the calib video TV sweep (max 216 = 3 base-6 digits)
	int measureFrames = 20;	   // frames averaged per measurement
	int settleMs = 700;		   // LED + HyperHDR smoothing settle before measuring
	// Index-code symbol classification (tolerant of an uncalibrated camera).
	float symbolBlackThresh = 60.0f;   // max channel below this -> BLACK
	float symbolChannelThresh = 0.55f; // channel "on" when > thresh * max channel
	bool verbose = false;
};

class LedCalib
{
public:
	using SendLedFn = std::function<void(uint8_t r, uint8_t g, uint8_t b)>;
	using ExitFn = std::function<void(int code)>;

	LedCalib(const LedCalibConfig &cfg, SendLedFn sendLed, ExitFn exitFn);
	~LedCalib();

	LedCalib(const LedCalib &) = delete;
	LedCalib &operator=(const LedCalib &) = delete;

	/// Feed one captured raw frame (event-loop thread). bpp 3 = RGB byte
	/// order, bpp 4 = RGBX, or BGRX when swapRb is set (matches the
	/// conventions used by the autocalib paths).
	void processFrame(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb);

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
		Done
	};

	// Full-screen index-code symbols (values must match tools/gen-calib-video.py).
	enum class Sym
	{
		Black,
		White,
		Red,
		Green,
		Blue,
		Yellow,
		Cyan,
		Other
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
		int idx = -1;			   // TV sweep: decoded patch index
		uint8_t drive[3] = {};	   // LED sweep: drive value sent
		double mean[2][3] = {};	   // [tv|led][R,G,B]
		double tstd[2][3] = {};	   // temporal std of per-frame means
		double sstd[2][3] = {};	   // average spatial std
		int frames = 0;
	};

	enum class TvState
	{
		ScanSync,
		SyncBlack,
		ExpectD1,
		Gap1,
		ExpectD2,
		Gap2,
		ExpectD3,
		PrePatch,
		PatchWindow,
		WaitSync
	};

	RoiSample sampleRoi(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb,
						const LedCalibRoi &roi) const;
	Sym classify(const double mean[3]) const;
	static int digitValue(Sym s);
	static const char *symName(Sym s);

	void enterPhase(Phase p, int64_t nowMs);
	void handleKeys(int64_t nowMs);
	void whiteKey(char c, int64_t nowMs);
	void runWhite(const RoiSample &tv, const RoiSample &led, int64_t nowMs);
	void runTv(const RoiSample &tv, const RoiSample &led, int64_t nowMs);
	void onStableSym(Sym s, int64_t nowMs);
	void recordTvPatch(int64_t nowMs);
	void finishTvPhase(int64_t nowMs);
	void runLed(const RoiSample &tv, const RoiSample &led, int64_t nowMs);
	void buildLedPlan();

	void setLed(uint8_t r, uint8_t g, uint8_t b, int64_t nowMs);

	bool ensureOutDir();
	void writeWhiteAnchor();
	void writeTvSweep();
	void writeLedSweep();
	void writeBaseline();
	void writeMeta();
	void finishAll(int code);

	void keyboardLoop();

	LedCalibConfig _cfg;
	SendLedFn _sendLed;
	ExitFn _exit;

	Phase _phase = Phase::Done;
	int64_t _t0Ms = -1;
	bool _finished = false;
	bool _roiChecked = false;

	// keyboard reader
	std::thread _kbThread;
	std::atomic<bool> _kbStop{false};
	std::mutex _kbMutex;
	std::deque<char> _keys;

	// LED drive state
	uint8_t _drive[3] = {0, 0, 0};
	int64_t _lastSendMs = -1000000;

	Measurement _meas;

	// white anchor phase
	int _whiteStep = 8;
	bool _whiteAccepted = false;
	int64_t _whiteSettleUntilMs = -1;
	int64_t _lastWhitePrintMs = 0;
	SweepRecord _anchor{};
	double _anchorGain[3] = {1.0, 1.0, 1.0};
	bool _haveAnchor = false;

	// TV sweep decode
	Sym _runClass = Sym::Other;
	int _runLen = 0;
	Sym _stable = Sym::Other;
	TvState _tvState = TvState::ScanSync;
	int _digits[3] = {0, 0, 0};
	int64_t _patchWindowStartMs = 0;
	int _pendingIdx = -1;
	std::vector<SweepRecord> _tvRecords;
	std::vector<bool> _tvSeen;
	int _tvDecoded = 0;

	// LED sweep
	std::vector<std::array<uint8_t, 3>> _ledPlan;
	size_t _ledPlanPos = 0;
	int64_t _settleUntilMs = -1;
	SweepRecord _baseline{};
	bool _haveBaseline = false;
	std::vector<SweepRecord> _ledRecords;
};

#endif /* __LED_CALIB_H__ */
