#include "ledCalib.h"
#include "ledCalibWeb.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <termios.h>
#include <unistd.h>

namespace
{
	constexpr int64_t kResendIntervalMs = 100;	  // keep HyperHDR fed while a color is held
	constexpr int64_t kWebAckTimeoutMs = 8000;	  // page must ack a patch within this or we abort
	constexpr int kMinSettleFrames = 3;			  // fresh camera frames to discard after a web-patch ack

	// Edge-dense grid level: Chebyshev/cosine nodes -> samples clustered near 0
	// and 255 (where crush/clip live), sparse in the middle. i in [0,n-1].
	inline uint8_t edgeLevel(int i, int n)
	{
		if (n < 2)
			return 0;
		const double pi = std::acos(-1.0);
		const double t = 0.5 * (1.0 - std::cos(pi * i / (n - 1)));
		return static_cast<uint8_t>(std::lround(255.0 * t));
	}

	int64_t steadyMs()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(
				   std::chrono::steady_clock::now().time_since_epoch())
			.count();
	}

	void jsonTriple(std::ostream &os, const char *name, const double v[3], bool comma = true)
	{
		os << "  \"" << name << "\": [" << v[0] << ", " << v[1] << ", " << v[2] << "]";
		os << (comma ? ",\n" : "\n");
	}
}

void LedCalib::Measurement::start(int frames)
{
	active = true;
	target = std::max(1, frames);
	n = 0;
	std::memset(sum, 0, sizeof(sum));
	std::memset(sumSq, 0, sizeof(sumSq));
	std::memset(sstdSum, 0, sizeof(sstdSum));
}

void LedCalib::Measurement::add(const RoiSample &tv, const RoiSample &led)
{
	if (!active || n >= target)
		return;
	const RoiSample *rois[2] = {&tv, &led};
	for (int roi = 0; roi < 2; roi++)
	{
		for (int c = 0; c < 3; c++)
		{
			const double m = rois[roi]->mean[c];
			sum[roi][c] += m;
			sumSq[roi][c] += m * m;
			sstdSum[roi][c] += rois[roi]->sstd[c];
		}
	}
	n++;
}

void LedCalib::Measurement::finish(double mean[2][3], double tstd[2][3], double sstd[2][3])
{
	const double cnt = std::max(1, n);
	for (int roi = 0; roi < 2; roi++)
	{
		for (int c = 0; c < 3; c++)
		{
			const double m = sum[roi][c] / cnt;
			mean[roi][c] = m;
			const double var = sumSq[roi][c] / cnt - m * m;
			tstd[roi][c] = std::sqrt(std::max(0.0, var));
			sstd[roi][c] = sstdSum[roi][c] / cnt;
		}
	}
	active = false;
}

LedCalib::LedCalib(const LedCalibConfig &cfg, SendLedFn sendLed, ExitFn exitFn, LedCalibWeb *web)
	: _cfg(cfg), _sendLed(std::move(sendLed)), _exit(std::move(exitFn)), _web(web)
{
	if (_cfg.phase == "white")
		_phase = Phase::WhiteWait;
	else if (_cfg.phase == "tv")
		_phase = Phase::TvWait;
	else if (_cfg.phase == "led")
		_phase = Phase::LedWait;
	else if (_cfg.phase == "gamma")
		_phase = Phase::GammaWait;
	else if (_cfg.phase == "gamma_leds")
		_phase = Phase::GammaLedsWait;
	else if (_cfg.phase == "shading")
		_phase = Phase::ShadingWait;
	else
		_phase = Phase::Done; // validated upstream; never reached

	_kbThread = std::thread(&LedCalib::keyboardLoop, this);
}

LedCalib::~LedCalib()
{
	_kbStop.store(true, std::memory_order_release);
	if (_kbThread.joinable())
		_kbThread.join();
}

void LedCalib::keyboardLoop()
{
	termios orig{};
	bool rawOk = false;
	if (isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &orig) == 0)
	{
		termios raw = orig;
		raw.c_lflag &= ~(ICANON | ECHO);
		raw.c_cc[VMIN] = 0;
		raw.c_cc[VTIME] = 1; // 100 ms read timeout so the stop flag is honored
		if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0)
			rawOk = true;
	}

	// Without a tty (e.g. running under systemd in --calib-tv-web mode) reading
	// stdin returns EOF immediately and would busy-spin; idle instead and let
	// the web page provide keys.
	if (!rawOk)
	{
		while (!_kbStop.load(std::memory_order_acquire))
			usleep(100000);
		return;
	}

	while (!_kbStop.load(std::memory_order_acquire))
	{
		char c = 0;
		const ssize_t r = read(STDIN_FILENO, &c, 1);
		if (r == 1)
		{
			std::lock_guard<std::mutex> lock(_kbMutex);
			_keys.push_back(c);
		}
		else if (r < 0)
		{
			usleep(100000);
		}
		// r == 0: VTIME timeout, loop and re-check the stop flag
	}

	tcsetattr(STDIN_FILENO, TCSANOW, &orig);
}

void LedCalib::processFrame(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb)
{
	if (_finished)
		return;

	const int64_t nowMs = steadyMs();

	if (!_roiChecked)
	{
		_roiChecked = true;
		auto inFrame = [&](const LedCalibRoi &r)
		{
			return r.w > 0 && r.h > 0 && r.x >= 0 && r.y >= 0 && r.x + r.w <= width && r.y + r.h <= height;
		};
		// tv/gamma need the TV ROI; led/gamma_leds need the LED ROI (the TV ROI is
		// an optional sanity column); shading samples the whole TV quad and white is
		// a manual eye-match — neither uses a fixed ROI.
		const bool gamma = (_cfg.phase == "gamma");
		const bool tvOnly = gamma || _cfg.phase == "tv";
		const bool ledPhase = (_cfg.phase == "led" || _cfg.phase == "gamma_leds");
		bool roiBad = false;
		if (tvOnly)
			roiBad = !inFrame(_cfg.tvRoi);
		else if (ledPhase)
			roiBad = !inFrame(_cfg.ledRoi) || (_cfg.tvRoi.w > 0 && !inFrame(_cfg.tvRoi));
		if (roiBad)
		{
			std::cout << "ERROR: LedCalib ROI outside the " << width << "x" << height
					  << " frame (tv " << _cfg.tvRoi.x << "," << _cfg.tvRoi.y << "," << _cfg.tvRoi.w << "," << _cfg.tvRoi.h
					  << "; led " << _cfg.ledRoi.x << "," << _cfg.ledRoi.y << "," << _cfg.ledRoi.w << "," << _cfg.ledRoi.h
					  << ")" << std::endl;
			finishAll(EXIT_FAILURE);
			return;
		}
		enterPhase(_phase, nowMs);
	}

	// Release a first-patch start delay once its grace has elapsed (set by the
	// 'begin' tap in handleKeys on an earlier frame).
	tickStartDelay(nowMs);

	// Shading samples the whole TV quad from the raw frame; the fixed TV/LED
	// ROIs are not used.
	if (_phase == Phase::Shading || _phase == Phase::ShadingWait)
	{
		handleKeys(nowMs);
		if (_finished)
			return;
		if (_phase == Phase::Shading)
			runShading(img, width, height, strideBytes, bpp, swapRb, nowMs);
		return;
	}

	RoiSample tv = sampleRoi(img, width, height, strideBytes, bpp, swapRb, _cfg.tvRoi);
	// Match the shaded production output: scale the TV reading by the shading
	// gain at this ROI's position (identity unless --shading-map was given).
	for (int c = 0; c < 3; c++)
	{
		tv.mean[c] *= _cfg.tvShadeGain[c];
		tv.sstd[c] *= _cfg.tvShadeGain[c];
	}
	const RoiSample led = (_cfg.phase == "gamma" || _cfg.phase == "tv")
							  ? RoiSample{}
							  : sampleRoi(img, width, height, strideBytes, bpp, swapRb, _cfg.ledRoi);

	handleKeys(nowMs);
	if (_finished)
		return;

	if (nowMs - _lastSendMs >= kResendIntervalMs)
		setLed(_drive[0], _drive[1], _drive[2], nowMs);

	switch (_phase)
	{
	case Phase::White:
		// Manual eye-match: nothing to measure. The LED stays full white (re-sent
		// above) while the user tunes HyperHDR's white balance or nudges.
		break;
	case Phase::Gamma:
		runGamma(tv, nowMs);
		break;
	case Phase::GammaLeds:
		runGammaLeds(led, nowMs);
		break;
	case Phase::Tv:
		runTvWeb(tv, led, nowMs);
		break;
	case Phase::LedBaseline:
	case Phase::LedSweep:
		runLed(tv, led, nowMs);
		break;
	default:
		// Web wait phases: re-print the URL every few seconds until the user starts.
		if (_web && !_webStarted && nowMs - _lastUrlPrintMs >= 5000)
		{
			_lastUrlPrintMs = nowMs;
			std::cout << (_web->clientConnected() ? "  page connected — tap 'begin' on the TV"
												  : "  waiting for the page: open http://<this-pi>:")
					  << (_web->clientConnected() ? "" : std::to_string(_web->port()) + "/")
					  << std::endl;
		}
		break;
	}
}

LedCalib::RoiSample LedCalib::sampleRoi(const uint8_t *img, int width, int height, int strideBytes, int bpp,
										bool swapRb, const LedCalibRoi &roi) const
{
	RoiSample out;
	const int x0 = std::clamp(roi.x, 0, width - 1);
	const int y0 = std::clamp(roi.y, 0, height - 1);
	const int x1 = std::clamp(roi.x + roi.w, x0 + 1, width);
	const int y1 = std::clamp(roi.y + roi.h, y0 + 1, height);
	const bool swap = (bpp == 4) && swapRb;

	uint64_t sum[3] = {0, 0, 0};
	uint64_t sumSq[3] = {0, 0, 0};
	for (int y = y0; y < y1; y++)
	{
		const uint8_t *p = img + static_cast<size_t>(y) * static_cast<size_t>(strideBytes) +
						   static_cast<size_t>(x0) * static_cast<size_t>(bpp);
		for (int x = x0; x < x1; x++, p += bpp)
		{
			const uint32_t r = swap ? p[2] : p[0];
			const uint32_t g = p[1];
			const uint32_t b = swap ? p[0] : p[2];
			sum[0] += r;
			sum[1] += g;
			sum[2] += b;
			sumSq[0] += static_cast<uint64_t>(r) * r;
			sumSq[1] += static_cast<uint64_t>(g) * g;
			sumSq[2] += static_cast<uint64_t>(b) * b;
		}
	}

	const double cnt = static_cast<double>(x1 - x0) * static_cast<double>(y1 - y0);
	for (int c = 0; c < 3; c++)
	{
		const double m = static_cast<double>(sum[c]) / cnt;
		out.mean[c] = m;
		const double var = static_cast<double>(sumSq[c]) / cnt - m * m;
		out.sstd[c] = std::sqrt(std::max(0.0, var));
	}
	return out;
}

void LedCalib::setLed(uint8_t r, uint8_t g, uint8_t b, int64_t nowMs)
{
	_drive[0] = r;
	_drive[1] = g;
	_drive[2] = b;
	_lastSendMs = nowMs;
	if (_sendLed)
		_sendLed(r, g, b);
	// Keep the Phase 0 page overlay in sync with the LED drive (same seq: no
	// new ack needed, just a display refresh). The TV reference holds at the
	// configured white level; r,g,b is the (nudgeable) LED-drive preview.
	if (_web && _phase == Phase::White)
	{
		const uint8_t w = static_cast<uint8_t>(_cfg.whiteLevel);
		_web->setState(w, w, w, true, r, g, b, _webSeq);
	}
}

void LedCalib::webRefreshWhite()
{
	if (_web)
	{
		const uint8_t w = static_cast<uint8_t>(_cfg.whiteLevel);
		_web->setState(w, w, w, true, _drive[0], _drive[1], _drive[2], _webSeq);
	}
}

void LedCalib::beginActivePhase(Phase active, int64_t nowMs)
{
	// The 'begin' tap just put the page fullscreen; some browsers then flash a
	// "press Esc to exit fullscreen" toast that can sit over the patch. Hold the
	// first patch back by startDelayMs (once per run) so it clears first. The TV
	// shows the wait-phase field (black/last) during the grace, then the patch.
	if (_web && _cfg.startDelayMs > 0 && !_startDelayDone)
	{
		_startDelayDone = true;
		_pendingPhase = active;
		_startDelayUntilMs = nowMs + _cfg.startDelayMs;
		std::cout << "Holding the first patch " << (_cfg.startDelayMs / 1000.0)
				  << "s for the browser's fullscreen prompt to clear..." << std::endl;
		return;
	}
	enterPhase(active, nowMs);
}

void LedCalib::tickStartDelay(int64_t nowMs)
{
	if (_pendingPhase == Phase::Done || nowMs < _startDelayUntilMs)
		return;
	const Phase active = _pendingPhase;
	_pendingPhase = Phase::Done;
	enterPhase(active, nowMs);
}

void LedCalib::enterPhase(Phase p, int64_t nowMs)
{
	_phase = p;
	switch (p)
	{
	case Phase::WhiteWait:
		if (_web)
		{
			if (_webStarted)
			{
				enterPhase(Phase::White, nowMs);
				return;
			}
			std::cout << "\nLedCalib white (manual white balance): open  http://<this-pi>:" << _web->port()
					  << "/  in a FULLSCREEN browser on the PC connected to the TV (same HDMI input\n"
						 "you normally watch), then tap 'begin'. 'q' quits." << std::endl;
			_lastUrlPrintMs = nowMs;
		}
		else
		{
			std::cout << "\nLedCalib white (manual white balance): show a full-screen white image on\n"
						 "the TV and press Enter. 'q' quits." << std::endl;
		}
		break;
	case Phase::White:
	{
		// Drive both the LEDs and the TV reference to the configured white level
		// (also sends a solid frame to HyperHDR so the LEDs go to that level).
		const uint8_t w = static_cast<uint8_t>(_cfg.whiteLevel);
		setLed(w, w, w, nowMs);
		if (_web)
		{
			// Full-screen white field so the whole TV is the white reference.
			_web->setPatchGeometry(100.0, -1.0, -1.0);
			_webSeq++;
			webRefreshWhite();
		}
		std::cout << "\nManual white balance: the LEDs are driven white at level " << static_cast<int>(w)
				  << ". Adjust HyperHDR's\n"
					 "white balance until the LED glow matches the TV white (or nudge here/on the\n"
					 "page to preview) —\n"
					 "  1/2 = red -/+   3/4 = green -/+   5/6 = blue -/+\n"
					 "  0 = toggle step (1/8)   Enter/Accept = done   q = quit" << std::endl;
		break;
	}
	case Phase::TvWait:
		setLed(0, 0, 0, nowMs);
		if (_webStarted)
		{
			enterPhase(Phase::Tv, nowMs);
			return;
		}
		std::cout << "\nLedCalib phase 1 (TV sweep): open  http://<this-pi>:" << _web->port()
				  << "/  fullscreen on the TV PC and tap 'begin'." << std::endl;
		_lastUrlPrintMs = nowMs;
		break;
	case Phase::Tv:
		// Restore the windowed measurement geometry (Phase 0 forced fullscreen).
		_web->setPatchGeometry(_cfg.patchPct, _cfg.patchX, _cfg.patchY);
		buildTvPlan();
		_tvPlanPos = 0;
		std::cout << "TV sweep (web): " << _tvPlan.size() << " patches commanded directly." << std::endl;
		showTvPatch(nowMs);
		break;
	case Phase::LedWait:
		setLed(0, 0, 0, nowMs);
		// Drive the TV black through the page; no manual step needed. On a tty
		// run (no --calib-tv-web) the user blacks the TV themselves.
		if (_web)
		{
			_webSeq++;
			_web->setState(0, 0, 0, false, 0, 0, 0, _webSeq);
		}
		if (_webStarted)
		{
			enterPhase(Phase::LedBaseline, nowMs);
			return;
		}
		if (_web)
			std::cout << "\nLedCalib phase 2 (LED sweep): open the page fullscreen on the TV PC\n"
						 "and tap 'begin' (the page will black out the TV)." << std::endl;
		else
			std::cout << "\nLedCalib phase 2 (LED sweep): make sure the TV is black and press Enter\n"
						 "(the LEDs are driven; only they should light the LED-ROI wall)." << std::endl;
		_lastUrlPrintMs = nowMs;
		break;
	case Phase::LedBaseline:
		setLed(0, 0, 0, nowMs);
		ledSettleStart(nowMs, 1500);
		std::cout << "LED sweep: measuring baseline (LEDs off, TV black)..." << std::endl;
		break;
	case Phase::LedSweep:
		buildLedPlan();
		_ledPlanPos = 0;
		setLed((*_ledPlan.begin())[0], (*_ledPlan.begin())[1], (*_ledPlan.begin())[2], nowMs);
		ledSettleStart(nowMs);
		// Rough ETA: settle + measurement (~16 ms/frame at the calib fps) per patch.
		std::cout << "LED sweep: " << _ledPlan.size() << " drive values, ~"
				  << (_ledPlan.size() * (static_cast<size_t>(_cfg.settleMs) + _cfg.measureFrames * 16) / 1000 / 60 + 1)
				  << " min." << std::endl;
		break;
	case Phase::GammaWait:
		setLed(0, 0, 0, nowMs); // LEDs off: this measures the TV->camera response only
		if (_webStarted)
		{
			enterPhase(Phase::Gamma, nowMs);
			return;
		}
		std::cout << "\nLedCalib gamma probe (" << (_cfg.gammaRaw ? "RAW Bayer" : "ISP RGB")
				  << "): open  http://<this-pi>:" << (_web ? _web->port() : 0)
				  << "/  fullscreen on the TV PC and tap 'begin'." << std::endl;
		_lastUrlPrintMs = nowMs;
		break;
	case Phase::Gamma:
		buildGammaPlan();
		_gammaPos = 0;
		_gammaRecords.clear();
		std::cout << "Gamma probe: " << _gammaPlan.size() << " patches (R/G/B/gray ramps, step "
				  << _cfg.gammaStep << "), measuring " << (_cfg.gammaRaw ? "RAW Bayer planes" : "ISP RGB")
				  << "." << std::endl;
		showGammaPatch(nowMs);
		break;
	case Phase::GammaLedsWait:
		setLed(0, 0, 0, nowMs);
		// Black the TV through the page (like the LED sweep) so only the LEDs light
		// the wall the LED ROI samples.
		if (_web)
		{
			_webSeq++;
			_web->setState(0, 0, 0, false, 0, 0, 0, _webSeq);
		}
		if (_webStarted)
		{
			enterPhase(Phase::GammaLeds, nowMs);
			return;
		}
		if (_web)
			std::cout << "\nLedCalib LED gamma probe: open the page fullscreen on the TV PC\n"
						 "and tap 'begin' (the page will black out the TV; the LEDs are driven)." << std::endl;
		else
			std::cout << "\nLedCalib LED gamma probe: make sure the TV is black and press Enter\n"
						 "(the LEDs are driven; only they should light the LED-ROI wall)." << std::endl;
		_lastUrlPrintMs = nowMs;
		break;
	case Phase::GammaLeds:
		buildGammaPlan();
		_gammaPos = 0;
		_gammaRecords.clear();
		setLed(static_cast<uint8_t>(_gammaPlan[0][1]), static_cast<uint8_t>(_gammaPlan[0][2]),
			   static_cast<uint8_t>(_gammaPlan[0][3]), nowMs);
		ledSettleStart(nowMs);
		std::cout << "LED gamma probe: " << _gammaPlan.size() << " drive values (R/G/B/gray ramps, step "
				  << _cfg.gammaStep << "), measuring the LED ROI, ~"
				  << (_gammaPlan.size() * (static_cast<size_t>(_cfg.settleMs) + _cfg.measureFrames * 16) / 1000 / 60 + 1)
				  << " min." << std::endl;
		break;
	case Phase::ShadingWait:
		setLed(0, 0, 0, nowMs); // LEDs off; this maps the TV/lens, not the LEDs
		if (_webStarted)
		{
			enterPhase(Phase::Shading, nowMs);
			return;
		}
		std::cout << "\nLedCalib shading (ALSC) calibration: open  http://<this-pi>:"
				  << (_web ? _web->port() : 0) << "/  fullscreen on the TV PC and tap 'begin'.\n"
					 "A safe gray (" << _cfg.shadingLevel << ") fills the whole TV; keep the room dark." << std::endl;
		_lastUrlPrintMs = nowMs;
		break;
	case Phase::Shading:
	{
		// Full-screen safe gray; sample the TV quad once it has settled.
		if (_web)
		{
			_web->setCleanField(true); // hide keep-alive + status: full-frame sample
			_web->setPatchGeometry(100.0, -1.0, -1.0);
			_webSeq++;
			const uint8_t g = static_cast<uint8_t>(_cfg.shadingLevel);
			_web->setState(g, g, g, false, 0, 0, 0, _webSeq);
		}
		const std::size_t cells = static_cast<std::size_t>(_cfg.shadingGridU) * _cfg.shadingGridV;
		_shadeSum.assign(cells * 3, 0.0);
		_shadeCount.assign(cells, 0);
		_shadeFrames = 0;
		_shadeSettleUntilMs = nowMs + std::max(_cfg.settleMs, 500);
		std::cout << "Shading: " << _cfg.shadingGridU << "x" << _cfg.shadingGridV
				  << " grid across the TV, gray " << _cfg.shadingLevel << "; settling..." << std::endl;
		break;
	}
	case Phase::Done:
		break;
	}
}

void LedCalib::handleKeys(int64_t nowMs)
{
	std::deque<char> keys;
	{
		std::lock_guard<std::mutex> lock(_kbMutex);
		keys.swap(_keys);
	}
	// The web page is another key source (nudge/accept/start/quit buttons).
	if (_web)
	{
		char c;
		while (_web->popKey(c))
			keys.push_back(c);
	}

	for (char c : keys)
	{
		if (c == 'q' || c == 'Q')
		{
			std::cout << "LedCalib aborted by user." << std::endl;
			finishAll(EXIT_FAILURE);
			return;
		}
		// The page's "begin" button arrives as kCalibBeginKey (distinct from
		// Enter so a reload + begin during the White phase can't fake the
		// accept); a tty run begins with Enter. Both start a wait phase and
		// unlock the unattended auto-advance of later wait phases.
		const bool begin = (c == '\n' || c == '\r' || c == kCalibBeginKey);
		if (_web && begin && (_phase == Phase::WhiteWait || _phase == Phase::TvWait ||
							  _phase == Phase::LedWait || _phase == Phase::GammaWait ||
							  _phase == Phase::GammaLedsWait || _phase == Phase::ShadingWait))
			_webStarted = true;
		switch (_phase)
		{
		case Phase::WhiteWait:
			if (begin)
				beginActivePhase(Phase::White, nowMs);
			break;
		case Phase::ShadingWait:
			if (begin)
				beginActivePhase(Phase::Shading, nowMs);
			break;
		case Phase::TvWait:
			if (begin)
				beginActivePhase(Phase::Tv, nowMs);
			break;
		case Phase::LedWait:
			if (begin)
				beginActivePhase(Phase::LedBaseline, nowMs);
			break;
		case Phase::GammaWait:
			if (begin)
				beginActivePhase(Phase::Gamma, nowMs);
			break;
		case Phase::GammaLedsWait:
			if (begin)
				beginActivePhase(Phase::GammaLeds, nowMs);
			break;
		case Phase::White:
			whiteKey(c, nowMs);
			break;
		default:
			break;
		}
		if (_finished)
			return;
	}
}

void LedCalib::whiteKey(char c, int64_t nowMs)
{
	int chan = -1;
	int dir = 0;
	switch (c)
	{
	case '1':
		chan = 0;
		dir = -1;
		break;
	case '2':
		chan = 0;
		dir = 1;
		break;
	case '3':
		chan = 1;
		dir = -1;
		break;
	case '4':
		chan = 1;
		dir = 1;
		break;
	case '5':
		chan = 2;
		dir = -1;
		break;
	case '6':
		chan = 2;
		dir = 1;
		break;
	case '0':
		_whiteStep = (_whiteStep == 1) ? 8 : 1;
		std::cout << "step=" << _whiteStep << std::endl;
		return;
	case '\n':
	case '\r':
		std::cout << "White balance confirmed at LED drive " << static_cast<int>(_drive[0]) << ","
				  << static_cast<int>(_drive[1]) << "," << static_cast<int>(_drive[2])
				  << "; exiting." << std::endl;
		finishAll(EXIT_SUCCESS);
		return;
	default:
		return;
	}

	const int v = std::clamp(static_cast<int>(_drive[chan]) + dir * _whiteStep, 0, 255);
	uint8_t d[3] = {_drive[0], _drive[1], _drive[2]};
	d[chan] = static_cast<uint8_t>(v);
	setLed(d[0], d[1], d[2], nowMs);
	std::cout << "LED white drive " << static_cast<int>(d[0]) << "," << static_cast<int>(d[1])
			  << "," << static_cast<int>(d[2]) << " (step " << _whiteStep << ")" << std::endl;
}

void LedCalib::finishTvPhase()
{
	if (_tvRecords.empty())
	{
		std::cout << "ERROR: TV sweep measured no patches" << std::endl;
		finishAll(EXIT_FAILURE);
		return;
	}
	if (!ensureOutDir())
	{
		finishAll(EXIT_FAILURE);
		return;
	}
	writeTvSweep();

	finishAll(EXIT_SUCCESS);
}

void LedCalib::buildLedPlan()
{
	_ledPlan.clear();
	const int g = std::max(2, _cfg.ledGrid);
	for (int r = 0; r < g; r++)
		for (int gg = 0; gg < g; gg++)
			for (int b = 0; b < g; b++)
				_ledPlan.push_back({edgeLevel(r, g), edgeLevel(gg, g), edgeLevel(b, g)});
	const int n = std::max(0, _cfg.ledGrayRamp);
	for (int i = 0; i < n; i++)
	{
		const uint8_t v = (n > 1) ? edgeLevel(i, n) : 255;
		_ledPlan.push_back({v, v, v});
	}
	// Dedicated fine per-channel axis ramp, edge-clustered near 0 and 255, to
	// resolve the dark (crush) and bright (clip) transitions better than the
	// coarse cube edges. Cheap because it is 1D (3*ledAxisSteps), not the cube.
	const int a = _cfg.ledAxisSteps;
	if (a >= 2)
		for (int i = 1; i < a; i++) // skip i=0 (level 0 is the baseline, already covered)
			for (int c = 0; c < 3; c++)
			{
				std::array<uint8_t, 3> d = {0, 0, 0};
				d[c] = edgeLevel(i, a);
				_ledPlan.push_back(d);
			}
}

void LedCalib::buildTvPlan()
{
	_tvPlan.clear();
	const int g = std::max(2, _cfg.tvGrid);
	for (int r = 0; r < g; r++)
		for (int gg = 0; gg < g; gg++)
			for (int b = 0; b < g; b++)
				_tvPlan.push_back({edgeLevel(r, g), edgeLevel(gg, g), edgeLevel(b, g)});
}

void LedCalib::showTvPatch(int64_t nowMs)
{
	const auto &p = _tvPlan[_tvPlanPos];
	_tvSeq = ++_webSeq;
	_web->setState(p[0], p[1], p[2], false, 0, 0, 0, _tvSeq);
	_tvSettleUntilMs = -1; // wait for the page to ack the paint
	_tvAckWaitStartMs = nowMs;
}

// Web-patch gate: wait for the page to confirm it painted this patch (with a
// timeout -> abort), then measure once BOTH the settle time has elapsed AND a
// few fresh frames have arrived since the ack: the time covers TV-panel +
// camera-pipeline latency, the frame floor flushes the stale in-flight frames
// even at low camera fps. True = feed _meas this frame.
bool LedCalib::webPatchReady(const char *what, std::size_t pos, std::size_t total, int64_t nowMs)
{
	if (_tvSettleUntilMs < 0)
	{
		if (_web->ackedSeq() >= _tvSeq)
		{
			_tvSettleUntilMs = nowMs + _cfg.settleMs;
			_tvSettleFrames = 0;
		}
		else if (_tvAckWaitStartMs >= 0 && nowMs - _tvAckWaitStartMs > kWebAckTimeoutMs)
		{
			std::cout << "ERROR: web page did not acknowledge " << what << " " << pos << "/" << total
					  << " within " << (kWebAckTimeoutMs / 1000) << "s (page closed or left fullscreen?)" << std::endl;
			finishAll(EXIT_FAILURE);
		}
		return false;
	}
	if (!_meas.active)
	{
		_tvSettleFrames++;
		if (nowMs < _tvSettleUntilMs || _tvSettleFrames <= kMinSettleFrames)
			return false;
		_meas.start(_cfg.measureFrames);
	}
	return true;
}

// LED-drive gate: same dual settle (time + min fresh frames since the drive
// change) without a web ack — the LEDs switch on the flatbuffer send, but
// in-flight camera frames still show the previous drive at low fps.
bool LedCalib::ledPatchReady(int64_t nowMs)
{
	if (!_meas.active)
	{
		_ledSettleFrames++;
		if (nowMs < _settleUntilMs || _ledSettleFrames <= kMinSettleFrames)
			return false;
		_meas.start(_cfg.measureFrames);
	}
	return true;
}

void LedCalib::ledSettleStart(int64_t nowMs, int minSettleMs)
{
	_settleUntilMs = nowMs + std::max(_cfg.settleMs, minSettleMs);
	_ledSettleFrames = 0;
}

void LedCalib::runTvWeb(const RoiSample &tv, const RoiSample &led, int64_t nowMs)
{
	if (!webPatchReady("patch", _tvPlanPos, _tvPlan.size(), nowMs))
		return;
	_meas.add(tv, led);
	if (!_meas.done())
		return;

	SweepRecord rec;
	rec.idx = static_cast<int>(_tvPlanPos);
	rec.frames = _meas.target;
	rec.drive[0] = _tvPlan[_tvPlanPos][0];
	rec.drive[1] = _tvPlan[_tvPlanPos][1];
	rec.drive[2] = _tvPlan[_tvPlanPos][2];
	_meas.finish(rec.mean, rec.tstd, rec.sstd);
	_tvRecords.push_back(rec);
	const int done = static_cast<int>(_tvRecords.size());
	if ((done % 10) == 0 || done == static_cast<int>(_tvPlan.size()) || _cfg.verbose)
		std::cout << "TV sweep: " << done << "/" << _tvPlan.size() << " patches" << std::endl;

	_tvPlanPos++;
	if (_tvPlanPos >= _tvPlan.size())
	{
		finishTvPhase();
		return;
	}
	showTvPatch(nowMs);
}

// ---- Gamma ramp probe ----

void LedCalib::buildGammaPlan()
{
	_gammaPlan.clear();
	const int step = std::clamp(_cfg.gammaStep, 1, 64);
	for (int chan = 0; chan < 4; chan++)
	{
		int last = -1;
		auto push = [&](int v)
		{
			int r = 0, g = 0, b = 0;
			if (chan == 0)
				r = v;
			else if (chan == 1)
				g = v;
			else if (chan == 2)
				b = v;
			else
				r = g = b = v;
			_gammaPlan.push_back({chan, r, g, b});
			last = v;
		};
		for (int v = 0; v <= 255; v += step)
			push(v);
		if (last != 255)
			push(255); // always include the clipping endpoint
	}
}

void LedCalib::showGammaPatch(int64_t nowMs)
{
	const auto &p = _gammaPlan[_gammaPos];
	_tvSeq = ++_webSeq;
	_web->setState(static_cast<uint8_t>(p[1]), static_cast<uint8_t>(p[2]), static_cast<uint8_t>(p[3]),
				   false, 0, 0, 0, _tvSeq);
	_tvSettleUntilMs = -1;
	_tvAckWaitStartMs = nowMs;
}

// Finish the running measurement into a GammaRecord (ROI slot `roi`: 0 = TV,
// 1 = LED wall), advance the plan and print progress.
void LedCalib::pushGammaRecord(int roi, const char *label)
{
	double mean[2][3], tstd[2][3], sstd[2][3];
	_meas.finish(mean, tstd, sstd);
	GammaRecord rec;
	rec.chan = _gammaPlan[_gammaPos][0];
	rec.level = std::max({_gammaPlan[_gammaPos][1], _gammaPlan[_gammaPos][2], _gammaPlan[_gammaPos][3]});
	rec.frames = _meas.target;
	for (int c = 0; c < 3; c++)
	{
		rec.mean[c] = mean[roi][c];
		rec.tstd[c] = tstd[roi][c];
		rec.sstd[c] = sstd[roi][c];
	}
	_gammaRecords.push_back(rec);

	_gammaPos++;
	if ((_gammaPos % 50) == 0 || _gammaPos == _gammaPlan.size())
		std::cout << label << ": " << _gammaPos << "/" << _gammaPlan.size() << " patches" << std::endl;
}

bool LedCalib::gammaFinishIfDone()
{
	if (_gammaPos < _gammaPlan.size())
		return false;
	if (!ensureOutDir())
	{
		finishAll(EXIT_FAILURE);
		return true;
	}
	reportGammaLimits();
	writeGammaCsv();
	finishAll(EXIT_SUCCESS);
	return true;
}

void LedCalib::runGamma(const RoiSample &meas, int64_t nowMs)
{
	if (!webPatchReady("gamma patch", _gammaPos, _gammaPlan.size(), nowMs))
		return;
	_meas.add(meas, RoiSample{});
	if (!_meas.done())
		return;

	pushGammaRecord(0, "Gamma");
	if (gammaFinishIfDone())
		return;
	showGammaPatch(nowMs);
}

void LedCalib::runGammaLeds(const RoiSample &led, int64_t nowMs)
{
	// LED-driven ramp: like the LED sweep (no web ack — the page is black),
	// measure the LED ROI (index 1).
	if (!ledPatchReady(nowMs))
		return;
	_meas.add(RoiSample{}, led);
	if (!_meas.done())
		return;

	pushGammaRecord(1, "LED gamma");
	if (gammaFinishIfDone())
		return;
	const auto &p = _gammaPlan[_gammaPos];
	setLed(static_cast<uint8_t>(p[1]), static_cast<uint8_t>(p[2]), static_cast<uint8_t>(p[3]), nowMs);
	ledSettleStart(nowMs);
}

void LedCalib::writeGammaCsv()
{
	const bool leds = (_cfg.phase == "gamma_leds");
	const std::string path =
		_cfg.outDir + "/" + (leds ? "gamma_leds.csv" : (_cfg.gammaRaw ? "raw_gamma.csv" : "gamma.csv"));
	std::ofstream f(path);
	f << "# " << (leds ? "LED-driven (LED-ROI, ISP RGB)" : (_cfg.gammaRaw ? "RAW Bayer-plane" : "ISP RGB"))
	  << " gamma ramp; mean units " << ((!leds && _cfg.gammaRaw) ? "0..1023 (10-bit)" : "0..255 (8-bit)") << "\n";
	f << "channel,level,meas_r,meas_g,meas_b,tstd_r,tstd_g,tstd_b,sstd_r,sstd_g,sstd_b,frames\n";
	const char *names[4] = {"R", "G", "B", "gray"};
	for (const auto &r : _gammaRecords)
	{
		f << names[r.chan] << "," << r.level;
		for (int c = 0; c < 3; c++)
			f << "," << r.mean[c];
		for (int c = 0; c < 3; c++)
			f << "," << r.tstd[c];
		for (int c = 0; c < 3; c++)
			f << "," << r.sstd[c];
		f << "," << r.frames << "\n";
	}
	std::cout << "Wrote " << path << " (" << _gammaRecords.size() << " patches)" << std::endl;
}

void LedCalib::reportGammaLimits()
{
	// Pre-flight: from the just-measured R/G/B/gray ramps, flag per channel where
	// the camera clips (pins near full scale) or crushes (sits in the dark floor),
	// so a bad exposure / TV-brightness is caught HERE, before the long sweep.
	const char *names[4] = {"R", "G", "B", "gray"};
	// gamma_leds ignores --gamma-raw (always 8-bit ISP RGB), so its full scale
	// must not follow the flag — same rule as writeGammaCsv's units line.
	const bool ledsRun = (_cfg.phase == "gamma_leds");
	const double full = (!ledsRun && _cfg.gammaRaw) ? 1023.0 : 255.0;
	const double ceilNear = 0.96 * full; // a reading this high is almost certainly sensor clip
	bool anyWarn = false;
	for (int chan = 0; chan < 4; chan++)
	{
		std::vector<const GammaRecord *> rs;
		for (const auto &r : _gammaRecords)
			if (r.chan == chan)
				rs.push_back(&r);
		if (rs.size() < 3)
			continue;
		std::sort(rs.begin(), rs.end(),
				  [](const GammaRecord *a, const GammaRecord *b) { return a->level < b->level; });
		const int cLo = (chan < 3) ? chan : 0; // single-channel ramp probes its own channel; gray probes all
		const int cHi = (chan < 3) ? chan : 2;
		for (int c = cLo; c <= cHi; c++)
		{
			const double lo = rs.front()->mean[c];
			double mx = 0.0;
			for (const auto *r : rs)
				mx = std::max(mx, r->mean[c]);
			const double noise = std::max(2.0 / 255.0 * full, 3.0 * rs.back()->tstd[c]);
			int crushLvl = rs.front()->level; // first command whose reading lifts clear of the floor
			for (const auto *r : rs)
				if (r->level > 0 && r->mean[c] > lo + noise)
				{
					crushLvl = r->level;
					break;
				}
			const bool clipping = mx >= ceilNear;
			int clipLvl = rs.back()->level; // highest command still below the saturated plateau
			if (clipping)
			{
				clipLvl = rs.front()->level;
				for (const auto *r : rs)
					if (r->mean[c] < mx - noise)
						clipLvl = r->level;
			}
			const bool crushBad = crushLvl > std::max(8, rs.front()->level + 1);
			if (clipping || crushBad)
			{
				anyWarn = true;
				std::cout << "WARNING: gamma " << names[chan] << " (camera " << "rgb"[c]
						  << ") responds only over command [" << crushLvl << ".." << clipLvl
						  << "] (reads " << lo << ".." << mx << ")";
				if (clipping)
					std::cout << "; CLIPS above " << clipLvl << " - lower exposure/brightness";
				if (crushBad)
					std::cout << "; CRUSHES below " << crushLvl;
				std::cout << std::endl;
			}
		}
	}
	if (!anyWarn)
		std::cout << "Gamma limits: all channels respond across the full command range (no clip/crush)."
				  << std::endl;
}

// RAW10 CSI2P Bayer-plane ROI averaging. fourcc selects the 2x2 phase.
namespace
{
	// color at (row,col): 0=R 1=G 2=B. libcamera PixelFormat::fourcc() returns
	// the V4L2 base fourcc (the CSI2P packing is a modifier, not in the fourcc),
	// so we key on 'RG10'/'BG10'/'GB10'/'BA10' rather than the packed names.
	// This sensor negotiates SBGGR10_CSI2P ('BG10', BGGR).
	inline int bayerColor(uint32_t fourcc, int y, int x)
	{
		const int yo = y & 1, xo = x & 1;
		switch (fourcc)
		{
		case 0x30314752: // 'RG10' SRGGB10: RGGB
			return (yo == 0) ? (xo == 0 ? 0 : 1) : (xo == 0 ? 1 : 2);
		case 0x30314742: // 'BG10' SBGGR10: BGGR
			return (yo == 0) ? (xo == 0 ? 2 : 1) : (xo == 0 ? 1 : 0);
		case 0x30314142: // 'BA10' SGRBG10: GRBG
			return (yo == 0) ? (xo == 0 ? 1 : 0) : (xo == 0 ? 2 : 1);
		case 0x30314247: // 'GB10' SGBRG10: GBRG
			return (yo == 0) ? (xo == 0 ? 1 : 2) : (xo == 0 ? 0 : 1);
		default:
			return (yo == 0) ? (xo == 0 ? 2 : 1) : (xo == 0 ? 1 : 0); // assume BGGR (this sensor)
		}
	}

	inline uint16_t raw10p(const uint8_t *row, int x)
	{
		const int grp = (x >> 2) * 5;
		const int within = x & 3;
		const uint8_t lo = row[grp + 4];
		return static_cast<uint16_t>((static_cast<uint16_t>(row[grp + within]) << 2) |
									 ((lo >> (within * 2)) & 0x3));
	}
}

LedCalib::RoiSample LedCalib::sampleRoiBayer(const uint8_t *img, int width, int height, int strideBytes,
											 uint32_t fourcc, const LedCalibRoi &roi) const
{
	RoiSample out;
	// raw10p reads up to byte (x>>2)*5+4; cap x at the last pixel whose 5-byte
	// group fits the row stride, so a width that is not a multiple of 4 with a
	// tight (unpadded) stride can't over-read past the row.
	const int safeW = std::min(width, (strideBytes / 5) * 4);
	const int x0 = std::clamp(roi.x, 0, std::max(0, safeW - 1));
	const int y0 = std::clamp(roi.y, 0, height - 1);
	const int x1 = std::clamp(roi.x + roi.w, x0 + 1, std::max(x0 + 1, safeW));
	const int y1 = std::clamp(roi.y + roi.h, y0 + 1, height);

	double sum[3] = {0, 0, 0}, sumSq[3] = {0, 0, 0};
	uint64_t cnt[3] = {0, 0, 0};
	for (int y = y0; y < y1; y++)
	{
		const uint8_t *row = img + static_cast<size_t>(y) * static_cast<size_t>(strideBytes);
		for (int x = x0; x < x1; x++)
		{
			const int c = bayerColor(fourcc, y, x);
			const double v = raw10p(row, x);
			sum[c] += v;
			sumSq[c] += v * v;
			cnt[c]++;
		}
	}
	for (int c = 0; c < 3; c++)
	{
		const double n = cnt[c] ? static_cast<double>(cnt[c]) : 1.0;
		const double m = sum[c] / n;
		out.mean[c] = m;
		out.sstd[c] = std::sqrt(std::max(0.0, sumSq[c] / n - m * m));
	}
	return out;
}

void LedCalib::processRawFrame(const uint8_t *img, int width, int height, int strideBytes, uint32_t fourcc)
{
	if (_finished)
		return;
	const int64_t nowMs = steadyMs();

	if (!_roiChecked)
	{
		_roiChecked = true;
		// Scale the (processed-resolution) TV ROI into raw-stream coords. The TV
		// patch is uniform, so landing inside the TV region is all that matters.
		const int refW = _cfg.gammaRefW > 0 ? _cfg.gammaRefW : width;
		const int refH = _cfg.gammaRefH > 0 ? _cfg.gammaRefH : height;
		_rawRoi.x = static_cast<int>(static_cast<int64_t>(_cfg.tvRoi.x) * width / refW) & ~1;
		_rawRoi.y = static_cast<int>(static_cast<int64_t>(_cfg.tvRoi.y) * height / refH) & ~1;
		_rawRoi.w = std::max(2, static_cast<int>(static_cast<int64_t>(_cfg.tvRoi.w) * width / refW));
		_rawRoi.h = std::max(2, static_cast<int>(static_cast<int64_t>(_cfg.tvRoi.h) * height / refH));
		std::cout << "Gamma RAW: stream " << width << "x" << height << " fourcc=0x" << std::hex << fourcc
				  << std::dec << "; TV ROI scaled to raw " << _rawRoi.x << "," << _rawRoi.y << ","
				  << _rawRoi.w << "," << _rawRoi.h << " (verify it sits on the TV)" << std::endl;
		if (_rawRoi.x + _rawRoi.w > width || _rawRoi.y + _rawRoi.h > height)
		{
			std::cout << "ERROR: scaled raw ROI outside the frame" << std::endl;
			finishAll(EXIT_FAILURE);
			return;
		}
		enterPhase(_phase, nowMs);
	}

	// Release a deferred first-patch start (begin-tap fullscreen grace), same as
	// processFrame — the raw stream is the only frame source in --gamma-raw, so
	// without this tick the deferred Gamma phase would never fire.
	tickStartDelay(nowMs);

	const RoiSample meas = sampleRoiBayer(img, width, height, strideBytes, fourcc, _rawRoi);

	handleKeys(nowMs);
	if (_finished)
		return;

	if (_phase == Phase::Gamma)
		runGamma(meas, nowMs);
}

bool LedCalib::gammaRawSelftest()
{
	// Synthesize a RAW10 CSI2P frame with known per-plane values for a given Bayer
	// layout, pack it, then verify sampleRoiBayer recovers the plane means. Covers
	// RGGB and the production BGGR ('BG10', this sensor) so an R/B swap in either
	// mapping is caught.
	const int w = 64, h = 48;
	const int stride = (w / 4) * 5; // packed, no extra padding
	const uint16_t valR = 900, valG = 500, valB = 120; // 10-bit
	auto run = [&](const char *name, uint32_t fourcc, auto colorAt) -> bool
	{
		std::vector<uint8_t> buf(static_cast<size_t>(stride) * h, 0);
		for (int y = 0; y < h; y++)
		{
			uint8_t *row = buf.data() + static_cast<size_t>(y) * stride;
			for (int x = 0; x < w; x++)
			{
				const int c = colorAt(y, x);
				const uint16_t v = (c == 0) ? valR : (c == 1) ? valG : valB;
				const int grp = (x >> 2) * 5, within = x & 3;
				row[grp + within] = static_cast<uint8_t>(v >> 2);
				row[grp + 4] |= static_cast<uint8_t>((v & 0x3) << (within * 2));
			}
		}
		LedCalibConfig cfg;
		LedCalib probe(cfg, nullptr, nullptr, nullptr);
		LedCalibRoi roi{4, 4, 40, 32};
		const RoiSample s = probe.sampleRoiBayer(buf.data(), w, h, stride, fourcc, roi);
		const bool ok = std::fabs(s.mean[0] - valR) < 0.5 && std::fabs(s.mean[1] - valG) < 0.5 &&
						std::fabs(s.mean[2] - valB) < 0.5;
		std::cout << "gammaRawSelftest " << name << ": R=" << s.mean[0] << " G=" << s.mean[1]
				  << " B=" << s.mean[2] << " (expect " << valR << "," << valG << "," << valB << ") -> "
				  << (ok ? "PASS" : "FAIL") << std::endl;
		return ok;
	};
	// RGGB: row0 = R G, row1 = G B
	const bool okRggb = run("RGGB", 0x30314752, [](int y, int x)
							{ return (y & 1) == 0 ? ((x & 1) == 0 ? 0 : 1) : ((x & 1) == 0 ? 1 : 2); });
	// BGGR ('BG10', this sensor): row0 = B G, row1 = G R
	const bool okBggr = run("BGGR", 0x30314742, [](int y, int x)
							{ return (y & 1) == 0 ? ((x & 1) == 0 ? 2 : 1) : ((x & 1) == 0 ? 1 : 0); });
	return okRggb && okBggr;
}

// ---- Shading (ALSC) calibration ----

void LedCalib::runShading(const uint8_t *img, int width, int height, int strideBytes, int bpp, bool swapRb, int64_t nowMs)
{
	if (nowMs < _shadeSettleUntilMs)
		return;

	const int U = _cfg.shadingGridU, V = _cfg.shadingGridV;
	const double *H = _cfg.invHomography;
	const bool useH = _cfg.haveHomography;
	auto mapH = [&](double u, double v, double &x, double &y)
	{
		if (!useH)
		{
			// No --transform: unit square -> full frame (clean-TV capture).
			x = u * width;
			y = v * height;
			return;
		}
		double xp = H[0] * u + H[1] * v + H[2];
		double yp = H[3] * u + H[4] * v + H[5];
		double wp = H[6] * u + H[7] * v + H[8];
		if (std::fabs(wp) < 1e-9)
			wp = (wp < 0 ? -1e-9 : 1e-9);
		x = xp / wp;
		y = yp / wp;
	};

	for (int j = 0; j < V; j++)
	{
		for (int i = 0; i < U; i++)
		{
			double xs[4], ys[4];
			mapH(static_cast<double>(i) / U, static_cast<double>(j) / V, xs[0], ys[0]);
			mapH(static_cast<double>(i + 1) / U, static_cast<double>(j) / V, xs[1], ys[1]);
			mapH(static_cast<double>(i) / U, static_cast<double>(j + 1) / V, xs[2], ys[2]);
			mapH(static_cast<double>(i + 1) / U, static_cast<double>(j + 1) / V, xs[3], ys[3]);
			const double xmin = std::min({xs[0], xs[1], xs[2], xs[3]});
			const double xmax = std::max({xs[0], xs[1], xs[2], xs[3]});
			const double ymin = std::min({ys[0], ys[1], ys[2], ys[3]});
			const double ymax = std::max({ys[0], ys[1], ys[2], ys[3]});
			LedCalibRoi roi;
			roi.x = static_cast<int>(std::floor(xmin));
			roi.y = static_cast<int>(std::floor(ymin));
			roi.w = std::max(1, static_cast<int>(std::ceil(xmax - xmin)));
			roi.h = std::max(1, static_cast<int>(std::ceil(ymax - ymin)));
			const RoiSample s = sampleRoi(img, width, height, strideBytes, bpp, swapRb, roi);
			const std::size_t idx = static_cast<std::size_t>(j) * U + i;
			for (int c = 0; c < 3; c++)
				_shadeSum[idx * 3 + c] += s.mean[c];
			_shadeCount[idx]++;
		}
	}

	if (++_shadeFrames >= _cfg.measureFrames)
	{
		if (!ensureOutDir())
		{
			finishAll(EXIT_FAILURE);
			return;
		}
		writeShadingMap();
		finishAll(EXIT_SUCCESS);
	}
}

void LedCalib::writeShadingMap()
{
	const int U = _cfg.shadingGridU, V = _cfg.shadingGridV;
	std::vector<double> m(static_cast<std::size_t>(U) * V * 3, 0.0);
	double mn[3] = {1e18, 1e18, 1e18};
	double mx[3] = {0, 0, 0};
	for (int idx = 0; idx < U * V; idx++)
	{
		const double cnt = _shadeCount[idx] ? static_cast<double>(_shadeCount[idx]) : 1.0;
		for (int c = 0; c < 3; c++)
		{
			const double v = _shadeSum[idx * 3 + c] / cnt;
			m[idx * 3 + c] = v;
			mn[c] = std::min(mn[c], v);
			mx[c] = std::max(mx[c], v);
		}
	}

	// A single near-black cell - e.g. the --transform quad overhanging the lit TV
	// onto the bezel/letterbox - becomes mn[c] and collapses every gain_c (=
	// mn/mean) toward 0, blacking that channel out across the whole frame. The
	// strict-min "darken-to-dimmest" target is only valid if every cell sits on
	// the lit TV. Deferred robust-target fix: see CLAUDE.md ("shading min").
	for (int c = 0; c < 3; c++)
		if (mx[c] > 1e-6 && mn[c] < 0.33 * mx[c])
			std::cout << "WARNING: shading channel " << "rgb"[c] << " has a near-black cell (min "
					  << mn[c] << " vs brightest " << mx[c] << "): darken-to-dimmest would drive gain_"
					  << "rgb"[c] << " to ~" << (mn[c] / mx[c]) << " across the frame (channel blackout). "
					  << "Check the --transform quad sits inside the lit TV, then re-probe." << std::endl;

	const std::string path = _cfg.outDir + "/shading.csv";
	std::ofstream f(path);
	f << "# ALSC shading map; gray " << _cfg.shadingLevel << ", grid " << U << "x" << V
	  << "; gain = min/mean (<=1, darken-to-dimmest, never clips)\n";
	f << "i,j,mean_r,mean_g,mean_b,gain_r,gain_g,gain_b\n";
	for (int j = 0; j < V; j++)
	{
		for (int i = 0; i < U; i++)
		{
			const int idx = j * U + i;
			f << i << "," << j;
			for (int c = 0; c < 3; c++)
				f << "," << m[idx * 3 + c];
			for (int c = 0; c < 3; c++)
				f << "," << (m[idx * 3 + c] > 1e-6 ? mn[c] / m[idx * 3 + c] : 1.0);
			f << "\n";
		}
	}
	std::cout << "Wrote " << path << " (" << (U * V) << " cells)" << std::endl;

	// Magnitude report: luminance falloff + center->corner colour drift.
	auto luma = [&](int idx)
	{ return 0.299 * m[idx * 3] + 0.587 * m[idx * 3 + 1] + 0.114 * m[idx * 3 + 2]; };
	const int cIdx = (V / 2) * U + (U / 2);
	const double cl = std::max(1e-6, luma(cIdx));
	if (cl < 16.0)
		std::cout << "WARNING: TV-center reads near black (" << cl << "); is the --transform quad on the\n"
					 "         TV and is it showing the gray field? The numbers below are unreliable." << std::endl;
	const double crg = m[cIdx * 3] / std::max(1e-6, m[cIdx * 3 + 1]);
	const double cbg = m[cIdx * 3 + 2] / std::max(1e-6, m[cIdx * 3 + 1]);
	double lumaMin = 1e18, lumaMax = 0.0, colorMax = 0.0;
	for (int idx = 0; idx < U * V; idx++)
	{
		lumaMin = std::min(lumaMin, luma(idx));
		lumaMax = std::max(lumaMax, luma(idx));
		const double rg = m[idx * 3] / std::max(1e-6, m[idx * 3 + 1]);
		const double bg = m[idx * 3 + 2] / std::max(1e-6, m[idx * 3 + 1]);
		colorMax = std::max(colorMax, std::max(std::fabs(rg / crg - 1.0), std::fabs(bg / cbg - 1.0)));
	}
	const int corners[4] = {0, U - 1, (V - 1) * U, V * U - 1};
	std::cout << "Shading magnitude (gray " << _cfg.shadingLevel << ", center luma " << cl << "):\n"
			  << "  luminance falloff: min " << (100.0 * lumaMin / cl - 100.0) << "% .. max +"
			  << (100.0 * lumaMax / cl - 100.0) << "% vs center\n"
			  << "  corners vs center luma:";
	for (int k = 0; k < 4; k++)
		std::cout << " " << (100.0 * luma(corners[k]) / cl - 100.0) << "%";
	std::cout << "\n  max colour (R/G,B/G) drift center->edge: " << (100.0 * colorMax) << "%\n"
			  << "  -> color drift is the number that matters for edge-LED accuracy; "
				 "<~2% = skip, larger = worth applying." << std::endl;
}

void LedCalib::runLed(const RoiSample &tv, const RoiSample &led, int64_t nowMs)
{
	if (!ledPatchReady(nowMs))
		return;
	_meas.add(tv, led);
	if (!_meas.done())
		return;

	SweepRecord rec;
	rec.frames = _meas.target;
	_meas.finish(rec.mean, rec.tstd, rec.sstd);

	if (_phase == Phase::LedBaseline)
	{
		_baseline = rec;
		_haveBaseline = true;
		std::cout << "Baseline: LED-wall " << rec.mean[1][0] << "," << rec.mean[1][1] << "," << rec.mean[1][2]
				  << "  TV " << rec.mean[0][0] << "," << rec.mean[0][1] << "," << rec.mean[0][2] << std::endl;
		enterPhase(Phase::LedSweep, nowMs);
		return;
	}

	rec.idx = static_cast<int>(_ledPlanPos);
	rec.drive[0] = _ledPlan[_ledPlanPos][0];
	rec.drive[1] = _ledPlan[_ledPlanPos][1];
	rec.drive[2] = _ledPlan[_ledPlanPos][2];
	_ledRecords.push_back(rec);

	_ledPlanPos++;
	if (_ledPlanPos >= _ledPlan.size())
	{
		if (!ensureOutDir())
		{
			finishAll(EXIT_FAILURE);
			return;
		}
		writeBaseline();
		writeLedSweep();
		writeMeta();
		std::cout << "LED sweep complete: " << _ledRecords.size() << " measurements." << std::endl;
		finishAll(EXIT_SUCCESS);
		return;
	}

	if ((_ledPlanPos % 25) == 0)
		std::cout << "LED sweep: " << _ledPlanPos << "/" << _ledPlan.size() << std::endl;
	setLed(_ledPlan[_ledPlanPos][0], _ledPlan[_ledPlanPos][1], _ledPlan[_ledPlanPos][2], nowMs);
	ledSettleStart(nowMs);
}

bool LedCalib::ensureDir(const std::string &dir)
{
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	if (ec)
	{
		std::cout << "ERROR: cannot create output dir '" << dir << "': " << ec.message() << std::endl;
		return false;
	}
	return true;
}

bool LedCalib::ensureOutDir() { return ensureDir(_cfg.outDir); }

void LedCalib::writeTvSweep()
{
	const std::string path = _cfg.outDir + "/tv_sweep.csv";
	std::ofstream f(path);
	f << "index,patch_r,patch_g,patch_b,tv_r,tv_g,tv_b,tv_r_tstd,tv_g_tstd,tv_b_tstd,"
		 "tv_r_sstd,tv_g_sstd,tv_b_sstd,frames\n";
	std::vector<SweepRecord> sorted = _tvRecords;
	std::sort(sorted.begin(), sorted.end(), [](const SweepRecord &a, const SweepRecord &b)
			  { return a.idx < b.idx; });
	for (const auto &r : sorted)
	{
		f << r.idx << "," << static_cast<int>(r.drive[0]) << "," << static_cast<int>(r.drive[1])
		  << "," << static_cast<int>(r.drive[2]);
		// Only the TV ROI (roi 0) is measured in the TV sweep; the LED columns
		// were always zero, so they are not written.
		for (int c = 0; c < 3; c++)
			f << "," << r.mean[0][c];
		for (int c = 0; c < 3; c++)
			f << "," << r.tstd[0][c];
		for (int c = 0; c < 3; c++)
			f << "," << r.sstd[0][c];
		f << "," << r.frames << "\n";
	}
	std::cout << "Wrote " << path << " (" << sorted.size() << " patches)" << std::endl;
}

void LedCalib::writeLedSweep()
{
	const std::string path = _cfg.outDir + "/led_sweep.csv";
	std::ofstream f(path);
	f << "drive_r,drive_g,drive_b,led_r,led_g,led_b,led_r_tstd,led_g_tstd,led_b_tstd,"
		 "led_r_sstd,led_g_sstd,led_b_sstd,frames\n";
	for (const auto &r : _ledRecords)
	{
		f << static_cast<int>(r.drive[0]) << "," << static_cast<int>(r.drive[1]) << ","
		  << static_cast<int>(r.drive[2]);
		for (int c = 0; c < 3; c++)
			f << "," << r.mean[1][c];
		for (int c = 0; c < 3; c++)
			f << "," << r.tstd[1][c];
		for (int c = 0; c < 3; c++)
			f << "," << r.sstd[1][c];
		f << "," << r.frames << "\n";
	}
	std::cout << "Wrote " << path << " (" << _ledRecords.size() << " drive values)" << std::endl;
}

void LedCalib::writeBaseline()
{
	if (!_haveBaseline)
		return;
	std::ofstream f(_cfg.outDir + "/baseline.json");
	f << "{\n";
	jsonTriple(f, "tv_mean", _baseline.mean[0]);
	jsonTriple(f, "tv_tstd", _baseline.tstd[0]);
	jsonTriple(f, "led_mean", _baseline.mean[1]);
	jsonTriple(f, "led_tstd", _baseline.tstd[1]);
	jsonTriple(f, "led_sstd", _baseline.sstd[1]);
	f << "  \"frames\": " << _baseline.frames << "\n";
	f << "}\n";
	std::cout << "Wrote " << _cfg.outDir << "/baseline.json" << std::endl;
}

void LedCalib::writeMeta()
{
	std::ofstream f(_cfg.outDir + "/meta.json");
	f << "{\n";
	f << "  \"phase\": \"" << _cfg.phase << "\",\n";
	f << "  \"tv_roi\": [" << _cfg.tvRoi.x << ", " << _cfg.tvRoi.y << ", " << _cfg.tvRoi.w << ", " << _cfg.tvRoi.h << "],\n";
	f << "  \"led_roi\": [" << _cfg.ledRoi.x << ", " << _cfg.ledRoi.y << ", " << _cfg.ledRoi.w << ", " << _cfg.ledRoi.h << "],\n";
	f << "  \"led_grid\": " << _cfg.ledGrid << ",\n";
	f << "  \"led_gray_ramp\": " << _cfg.ledGrayRamp << ",\n";
	f << "  \"tv_patch_count\": " << (_cfg.tvGrid * _cfg.tvGrid * _cfg.tvGrid) << ",\n";
	f << "  \"tv_patches_measured\": " << _tvRecords.size() << ",\n";
	f << "  \"measure_frames\": " << _cfg.measureFrames << ",\n";
	f << "  \"settle_ms\": " << _cfg.settleMs << ",\n";
	f << "  \"timestamp\": " << static_cast<long long>(::time(nullptr)) << "\n";
	f << "}\n";
	std::cout << "Wrote " << _cfg.outDir << "/meta.json" << std::endl;
}

void LedCalib::finishAll(int code)
{
	if (_finished)
		return;
	_finished = true;
	// Leave the LEDs dark; ignore the resend throttle so this final frame goes out.
	if (_sendLed)
		_sendLed(0, 0, 0);
	// Blank the TV page so it doesn't hold the last (e.g. white) patch on screen.
	// The web server runs on its own thread; give its SSE tick a moment to push
	// before we exit and tear it down.
	if (_web)
	{
		_web->setState(0, 0, 0, false, 0, 0, 0, ++_webSeq);
		usleep(200000);
	}
	std::cout << "LedCalib finished (" << (code == EXIT_SUCCESS ? "ok" : "failed") << ")" << std::endl;
	if (_exit)
		_exit(code);
}
