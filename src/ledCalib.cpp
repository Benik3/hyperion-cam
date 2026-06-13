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

// Index-code sequence nominal timings; must match tools/gen-calib-video.py:
//   WHITE .4s | BLACK .3s | D1 .4s | BLACK .2s | D2 .4s | BLACK .2s | D3 .4s
//   | BLACK .3s | PATCH 1.5s | BLACK .3s
namespace
{
	constexpr int kMinRun = 3; // frames a class must hold to count as a stable token
	// From detecting the post-D3 black (detection lags ~kMinRun frames) to the start
	// of the patch measurement window: 300 ms nominal gap + 350 ms margin - ~100 ms lag.
	constexpr int64_t kPatchWindowDelayMs = 550;
	constexpr int64_t kResendIntervalMs = 100;	  // keep HyperHDR fed while a color is held
	constexpr int64_t kWebAckTimeoutMs = 8000;	  // page must ack a patch within this or we abort

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
	if (_cfg.tvWeb)
		_cfg.tvPatchCount = _cfg.tvGrid * _cfg.tvGrid * _cfg.tvGrid;
	_tvSeen.assign(static_cast<size_t>(std::max(1, _cfg.tvPatchCount)), false);

	if (_cfg.phase == "white")
		_phase = Phase::WhiteWait;
	else if (_cfg.phase == "tv")
		_phase = Phase::TvWait;
	else if (_cfg.phase == "led")
		_phase = Phase::LedWait;
	else
		_phase = Phase::WhiteWait;

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
		_t0Ms = nowMs;
		auto inFrame = [&](const LedCalibRoi &r)
		{
			return r.w > 0 && r.h > 0 && r.x >= 0 && r.y >= 0 && r.x + r.w <= width && r.y + r.h <= height;
		};
		if (!inFrame(_cfg.tvRoi) || !inFrame(_cfg.ledRoi))
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

	const RoiSample tv = sampleRoi(img, width, height, strideBytes, bpp, swapRb, _cfg.tvRoi);
	const RoiSample led = sampleRoi(img, width, height, strideBytes, bpp, swapRb, _cfg.ledRoi);

	handleKeys(nowMs);
	if (_finished)
		return;

	if (nowMs - _lastSendMs >= kResendIntervalMs)
		setLed(_drive[0], _drive[1], _drive[2], nowMs);

	switch (_phase)
	{
	case Phase::White:
		runWhite(tv, led, nowMs);
		break;
	case Phase::Tv:
		if (_web)
			runTvWeb(tv, led, nowMs);
		else
			runTv(tv, led, nowMs);
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

LedCalib::Sym LedCalib::classify(const double mean[3]) const
{
	const double maxc = std::max({mean[0], mean[1], mean[2]});
	if (maxc < _cfg.symbolBlackThresh)
		return Sym::Black;
	const double th = _cfg.symbolChannelThresh * maxc;
	const bool r = mean[0] > th;
	const bool g = mean[1] > th;
	const bool b = mean[2] > th;
	if (r && g && b)
		return Sym::White;
	if (r && g)
		return Sym::Yellow;
	if (g && b)
		return Sym::Cyan;
	if (r && b)
		return Sym::Other; // magenta: not used as a digit
	if (r)
		return Sym::Red;
	if (g)
		return Sym::Green;
	if (b)
		return Sym::Blue;
	return Sym::Other;
}

int LedCalib::digitValue(Sym s)
{
	switch (s)
	{
	case Sym::White:
		return 0;
	case Sym::Red:
		return 1;
	case Sym::Green:
		return 2;
	case Sym::Blue:
		return 3;
	case Sym::Yellow:
		return 4;
	case Sym::Cyan:
		return 5;
	default:
		return -1;
	}
}

const char *LedCalib::symName(Sym s)
{
	switch (s)
	{
	case Sym::Black:
		return "BLACK";
	case Sym::White:
		return "WHITE";
	case Sym::Red:
		return "RED";
	case Sym::Green:
		return "GREEN";
	case Sym::Blue:
		return "BLUE";
	case Sym::Yellow:
		return "YELLOW";
	case Sym::Cyan:
		return "CYAN";
	default:
		return "OTHER";
	}
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
	// new ack needed, just a display refresh).
	if (_web && _phase == Phase::White)
		_web->setState(255, 255, 255, true, r, g, b, _webSeq);
}

void LedCalib::webRefreshWhite()
{
	if (_web)
		_web->setState(255, 255, 255, true, _drive[0], _drive[1], _drive[2], _webSeq);
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
			std::cout << "\nLedCalib phase 0 (eye anchor): open  http://<this-pi>:" << _web->port()
					  << "/  in a FULLSCREEN browser on the PC connected to the TV (same HDMI input\n"
						 "you normally watch), then tap 'begin'. 'q' quits." << std::endl;
			_lastUrlPrintMs = nowMs;
		}
		else
		{
			std::cout << "\nLedCalib phase 0 (eye anchor): put the calib video on its WHITE section\n"
						 "(or any full-screen white) and press Enter. 'q' quits." << std::endl;
		}
		break;
	case Phase::White:
		setLed(255, 255, 255, nowMs);
		_whiteAccepted = false;
		if (_web)
		{
			_webSeq++;
			webRefreshWhite();
		}
		std::cout << "\nAdjust the LED white until the wall glow matches the TV white BY EYE"
					 << (_web ? " (buttons on the page, or keys here):\n" : ":\n")
					 << "  1/2 = red -/+   3/4 = green -/+   5/6 = blue -/+\n"
					 "  0 = toggle step (1/8)   Enter = accept   q = quit" << std::endl;
		break;
	case Phase::TvWait:
		setLed(0, 0, 0, nowMs);
		if (_web)
		{
			if (_webStarted)
			{
				enterPhase(Phase::Tv, nowMs);
				return;
			}
			std::cout << "\nLedCalib phase 1 (TV sweep): open  http://<this-pi>:" << _web->port()
					  << "/  fullscreen on the TV PC and tap 'begin'." << std::endl;
			_lastUrlPrintMs = nowMs;
		}
		else
		{
			std::cout << "\nLedCalib phase 1 (TV sweep): LEDs are now off. Start the calib video\n"
						 "TV-sweep section, then press Enter. Do not pause/seek during the sweep\n"
						 "('f' finishes the phase early with the patches decoded so far)." << std::endl;
		}
		break;
	case Phase::Tv:
		if (_web)
		{
			buildTvPlan();
			_tvPlanPos = 0;
			std::cout << "TV sweep (web): " << _tvPlan.size() << " patches commanded directly." << std::endl;
			showTvPatch(nowMs);
			break;
		}
		_tvState = TvState::ScanSync;
		_runClass = Sym::Other;
		_runLen = 0;
		_stable = Sym::Other;
		std::cout << "TV sweep: decoding patch index codes (" << _cfg.tvPatchCount << " patches expected)..." << std::endl;
		break;
	case Phase::LedWait:
		setLed(0, 0, 0, nowMs);
		if (_web)
		{
			// Drive the TV black through the page; no manual step needed.
			_webSeq++;
			_web->setState(0, 0, 0, false, 0, 0, 0, _webSeq);
			if (_webStarted)
			{
				enterPhase(Phase::LedBaseline, nowMs);
				return;
			}
			std::cout << "\nLedCalib phase 2 (LED sweep): open the page fullscreen on the TV PC\n"
						 "and tap 'begin' (the page will black out the TV)." << std::endl;
			_lastUrlPrintMs = nowMs;
		}
		else
		{
			std::cout << "\nLedCalib phase 2 (LED sweep): stop the video / leave the TV showing\n"
						 "black, keep the room dark, then press Enter." << std::endl;
		}
		break;
	case Phase::LedBaseline:
		setLed(0, 0, 0, nowMs);
		_settleUntilMs = nowMs + std::max(_cfg.settleMs, 1500);
		std::cout << "LED sweep: measuring baseline (LEDs off, TV black)..." << std::endl;
		break;
	case Phase::LedSweep:
		buildLedPlan();
		_ledPlanPos = 0;
		setLed((*_ledPlan.begin())[0], (*_ledPlan.begin())[1], (*_ledPlan.begin())[2], nowMs);
		_settleUntilMs = nowMs + _cfg.settleMs;
		std::cout << "LED sweep: " << _ledPlan.size() << " drive values, ~"
				  << (_ledPlan.size() * (static_cast<size_t>(_cfg.settleMs) + 700) / 1000 / 60 + 1)
				  << " min." << std::endl;
		break;
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
		const bool enter = (c == '\n' || c == '\r');
		// In web mode the page's "begin" button arrives as enter and unlocks
		// the unattended auto-advance of later wait phases.
		if (_web && enter && (_phase == Phase::WhiteWait || _phase == Phase::TvWait || _phase == Phase::LedWait))
			_webStarted = true;
		switch (_phase)
		{
		case Phase::WhiteWait:
			if (enter)
				enterPhase(Phase::White, nowMs);
			break;
		case Phase::TvWait:
			if (enter)
				enterPhase(Phase::Tv, nowMs);
			break;
		case Phase::LedWait:
			if (enter)
				enterPhase(Phase::LedBaseline, nowMs);
			break;
		case Phase::White:
			whiteKey(c, nowMs);
			break;
		case Phase::Tv:
			if (c == 'f' || c == 'F')
			{
				std::cout << "TV sweep finished early by user (" << _tvDecoded << "/"
						  << _cfg.tvPatchCount << " patches)." << std::endl;
				finishTvPhase(nowMs);
			}
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
	if (_whiteAccepted)
		return;

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
		_whiteAccepted = true;
		_whiteSettleUntilMs = nowMs + _cfg.settleMs;
		std::cout << "White anchor accepted at drive " << static_cast<int>(_drive[0]) << ","
				  << static_cast<int>(_drive[1]) << "," << static_cast<int>(_drive[2])
				  << "; measuring..." << std::endl;
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

void LedCalib::runWhite(const RoiSample &tv, const RoiSample &led, int64_t nowMs)
{
	if (!_whiteAccepted)
	{
		// 1 Hz feedback so the user sees what the camera sees while matching.
		if (nowMs - _lastWhitePrintMs >= 1000)
		{
			_lastWhitePrintMs = nowMs;
			std::cout << "  cam: TV " << tv.mean[0] << "," << tv.mean[1] << "," << tv.mean[2]
					  << "  LED-wall " << led.mean[0] << "," << led.mean[1] << "," << led.mean[2] << std::endl;
		}
		return;
	}

	if (nowMs < _whiteSettleUntilMs)
		return;
	if (!_meas.active)
		_meas.start(_cfg.measureFrames);
	_meas.add(tv, led);
	if (!_meas.done())
		return;

	_meas.finish(_anchor.mean, _anchor.tstd, _anchor.sstd);
	_anchor.frames = _meas.target;
	_anchor.drive[0] = _drive[0];
	_anchor.drive[1] = _drive[1];
	_anchor.drive[2] = _drive[2];

	for (int c = 0; c < 3; c++)
	{
		const double ledMean = _anchor.mean[1][c];
		if (ledMean < 1.0)
		{
			std::cout << "WARNING: LED ROI channel " << c << " too dark (" << ledMean
					  << ") for a white anchor gain; using 1.0" << std::endl;
			_anchorGain[c] = 1.0;
		}
		else
		{
			_anchorGain[c] = _anchor.mean[0][c] / ledMean;
		}
	}
	_haveAnchor = true;

	std::cout << "White anchor: TV " << _anchor.mean[0][0] << "," << _anchor.mean[0][1] << "," << _anchor.mean[0][2]
			  << "  LED-wall " << _anchor.mean[1][0] << "," << _anchor.mean[1][1] << "," << _anchor.mean[1][2]
			  << "  gains " << _anchorGain[0] << "," << _anchorGain[1] << "," << _anchorGain[2] << std::endl;

	if (!ensureOutDir())
	{
		finishAll(EXIT_FAILURE);
		return;
	}
	writeWhiteAnchor();

	if (_cfg.phase == "white")
		finishAll(EXIT_SUCCESS);
	else
		enterPhase(Phase::TvWait, nowMs);
}

void LedCalib::runTv(const RoiSample &tv, const RoiSample &led, int64_t nowMs)
{
	// Stable-token tracking on the TV ROI classification.
	const Sym s = classify(tv.mean);
	if (s == _runClass)
	{
		_runLen++;
	}
	else
	{
		_runClass = s;
		_runLen = 1;
	}
	if (_runLen == kMinRun && s != _stable)
	{
		_stable = s;
		onStableSym(s, nowMs);
	}

	// The patch itself is measured in a timed window (it may be any color,
	// including black, so it cannot be detected by classification).
	if (_tvState == TvState::PatchWindow)
	{
		if (!_meas.active && nowMs >= _patchWindowStartMs)
			_meas.start(_cfg.measureFrames);
		if (_meas.active)
		{
			_meas.add(tv, led);
			if (_meas.done())
			{
				recordTvPatch(nowMs);
				_tvState = TvState::WaitSync;
			}
		}
	}
}

void LedCalib::onStableSym(Sym s, int64_t nowMs)
{
	auto resync = [&](Sym sym)
	{
		if (_cfg.verbose)
			std::cout << "TV sweep: unexpected " << symName(sym) << ", resyncing" << std::endl;
		return (sym == Sym::White) ? TvState::SyncBlack : TvState::ScanSync;
	};

	switch (_tvState)
	{
	case TvState::ScanSync:
		if (s == Sym::White)
			_tvState = TvState::SyncBlack;
		break;
	case TvState::SyncBlack:
		if (s == Sym::Black)
			_tvState = TvState::ExpectD1;
		else
			_tvState = resync(s);
		break;
	case TvState::ExpectD1:
	case TvState::ExpectD2:
	case TvState::ExpectD3:
	{
		const int d = digitValue(s);
		if (d < 0)
		{
			_tvState = resync(s);
			break;
		}
		if (_tvState == TvState::ExpectD1)
		{
			_digits[0] = d;
			_tvState = TvState::Gap1;
		}
		else if (_tvState == TvState::ExpectD2)
		{
			_digits[1] = d;
			_tvState = TvState::Gap2;
		}
		else
		{
			_digits[2] = d;
			_pendingIdx = _digits[0] * 36 + _digits[1] * 6 + _digits[2];
			_tvState = TvState::PrePatch;
		}
		break;
	}
	case TvState::Gap1:
		_tvState = (s == Sym::Black) ? TvState::ExpectD2 : resync(s);
		break;
	case TvState::Gap2:
		_tvState = (s == Sym::Black) ? TvState::ExpectD3 : resync(s);
		break;
	case TvState::PrePatch:
		if (s == Sym::Black)
		{
			_patchWindowStartMs = nowMs + kPatchWindowDelayMs;
			_tvState = TvState::PatchWindow;
		}
		else
		{
			_tvState = resync(s);
		}
		break;
	case TvState::PatchWindow:
		// Patch (or trailing black) stabilizing: expected, the timed window handles it.
		break;
	case TvState::WaitSync:
		if (s == Sym::White)
			_tvState = TvState::SyncBlack;
		break;
	}
}

void LedCalib::recordTvPatch(int64_t nowMs)
{
	SweepRecord rec;
	rec.idx = _pendingIdx;
	rec.frames = _meas.target;
	_meas.finish(rec.mean, rec.tstd, rec.sstd);

	if (rec.idx < 0 || rec.idx >= _cfg.tvPatchCount)
	{
		std::cout << "WARNING: decoded out-of-range patch index " << rec.idx << "; dropped" << std::endl;
		return;
	}

	// Commanded patch RGB from the base-6 index grid (matches gen-calib-video.py).
	rec.drive[0] = static_cast<uint8_t>((rec.idx / 36) * 255 / 5);
	rec.drive[1] = static_cast<uint8_t>(((rec.idx / 6) % 6) * 255 / 5);
	rec.drive[2] = static_cast<uint8_t>((rec.idx % 6) * 255 / 5);

	if (_tvSeen[static_cast<size_t>(rec.idx)])
	{
		std::cout << "TV sweep: patch " << rec.idx << " re-measured; keeping the new reading" << std::endl;
		for (auto &r : _tvRecords)
		{
			if (r.idx == rec.idx)
			{
				r = rec;
				break;
			}
		}
	}
	else
	{
		_tvSeen[static_cast<size_t>(rec.idx)] = true;
		_tvRecords.push_back(rec);
		_tvDecoded++;
		if ((_tvDecoded % 10) == 0 || _tvDecoded == _cfg.tvPatchCount || _cfg.verbose)
			std::cout << "TV sweep: " << _tvDecoded << "/" << _cfg.tvPatchCount << " patches" << std::endl;
	}

	if (_tvDecoded >= _cfg.tvPatchCount)
		finishTvPhase(nowMs);
}

void LedCalib::finishTvPhase(int64_t nowMs)
{
	if (_tvRecords.empty())
	{
		std::cout << "ERROR: TV sweep decoded no patches" << std::endl;
		finishAll(EXIT_FAILURE);
		return;
	}
	if (!ensureOutDir())
	{
		finishAll(EXIT_FAILURE);
		return;
	}
	writeTvSweep();

	if (_cfg.phase == "tv")
		finishAll(EXIT_SUCCESS);
	else
		enterPhase(Phase::LedWait, nowMs);
}

void LedCalib::buildLedPlan()
{
	_ledPlan.clear();
	const int g = std::max(2, _cfg.ledGrid);
	for (int r = 0; r < g; r++)
	{
		for (int gg = 0; gg < g; gg++)
		{
			for (int b = 0; b < g; b++)
			{
				_ledPlan.push_back({static_cast<uint8_t>(r * 255 / (g - 1)),
									static_cast<uint8_t>(gg * 255 / (g - 1)),
									static_cast<uint8_t>(b * 255 / (g - 1))});
			}
		}
	}
	const int n = std::max(0, _cfg.ledGrayRamp);
	for (int i = 0; i < n; i++)
	{
		const uint8_t v = static_cast<uint8_t>(n > 1 ? i * 255 / (n - 1) : 255);
		_ledPlan.push_back({v, v, v});
	}
}

void LedCalib::buildTvPlan()
{
	_tvPlan.clear();
	const int g = std::max(2, _cfg.tvGrid);
	for (int r = 0; r < g; r++)
		for (int gg = 0; gg < g; gg++)
			for (int b = 0; b < g; b++)
				_tvPlan.push_back({static_cast<uint8_t>(r * 255 / (g - 1)),
								   static_cast<uint8_t>(gg * 255 / (g - 1)),
								   static_cast<uint8_t>(b * 255 / (g - 1))});
}

void LedCalib::showTvPatch(int64_t nowMs)
{
	const auto &p = _tvPlan[_tvPlanPos];
	_tvSeq = ++_webSeq;
	_web->setState(p[0], p[1], p[2], false, 0, 0, 0, _tvSeq);
	_tvSettleUntilMs = -1; // wait for the page to ack the paint
	_tvAckWaitStartMs = nowMs;
}

void LedCalib::runTvWeb(const RoiSample &tv, const RoiSample &led, int64_t nowMs)
{
	// Wait for the page to confirm it painted this patch, then settle.
	if (_tvSettleUntilMs < 0)
	{
		if (_web->ackedSeq() >= _tvSeq)
		{
			_tvSettleUntilMs = nowMs + _cfg.settleMs;
		}
		else if (_tvAckWaitStartMs >= 0 && nowMs - _tvAckWaitStartMs > kWebAckTimeoutMs)
		{
			std::cout << "ERROR: web page did not acknowledge patch " << _tvPlanPos << "/" << _tvPlan.size()
					  << " within " << (kWebAckTimeoutMs / 1000) << "s (page closed or left fullscreen?)" << std::endl;
			finishAll(EXIT_FAILURE);
		}
		return;
	}

	if (nowMs < _tvSettleUntilMs)
		return;
	if (!_meas.active)
		_meas.start(_cfg.measureFrames);
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
	_tvDecoded++;
	if ((_tvDecoded % 10) == 0 || _tvDecoded == static_cast<int>(_tvPlan.size()) || _cfg.verbose)
		std::cout << "TV sweep: " << _tvDecoded << "/" << _tvPlan.size() << " patches" << std::endl;

	_tvPlanPos++;
	if (_tvPlanPos >= _tvPlan.size())
	{
		finishTvPhase(nowMs);
		return;
	}
	showTvPatch(nowMs);
}

void LedCalib::runLed(const RoiSample &tv, const RoiSample &led, int64_t nowMs)
{
	if (nowMs < _settleUntilMs)
		return;
	if (!_meas.active)
		_meas.start(_cfg.measureFrames);
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
	_settleUntilMs = nowMs + _cfg.settleMs;
}

bool LedCalib::ensureOutDir()
{
	std::error_code ec;
	std::filesystem::create_directories(_cfg.outDir, ec);
	if (ec)
	{
		std::cout << "ERROR: cannot create output dir '" << _cfg.outDir << "': " << ec.message() << std::endl;
		return false;
	}
	return true;
}

void LedCalib::writeWhiteAnchor()
{
	std::ofstream f(_cfg.outDir + "/white_anchor.json");
	f << "{\n";
	f << "  \"drive\": [" << static_cast<int>(_anchor.drive[0]) << ", " << static_cast<int>(_anchor.drive[1])
	  << ", " << static_cast<int>(_anchor.drive[2]) << "],\n";
	jsonTriple(f, "tv_mean", _anchor.mean[0]);
	jsonTriple(f, "tv_tstd", _anchor.tstd[0]);
	jsonTriple(f, "tv_sstd", _anchor.sstd[0]);
	jsonTriple(f, "led_mean", _anchor.mean[1]);
	jsonTriple(f, "led_tstd", _anchor.tstd[1]);
	jsonTriple(f, "led_sstd", _anchor.sstd[1]);
	jsonTriple(f, "gains", _anchorGain, false);
	f << "}\n";
	std::cout << "Wrote " << _cfg.outDir << "/white_anchor.json" << std::endl;
}

void LedCalib::writeTvSweep()
{
	const std::string path = _cfg.outDir + "/tv_sweep.csv";
	std::ofstream f(path);
	f << "index,patch_r,patch_g,patch_b,tv_r,tv_g,tv_b,tv_r_tstd,tv_g_tstd,tv_b_tstd,tv_r_sstd,tv_g_sstd,tv_b_sstd,"
		 "led_r,led_g,led_b,led_r_tstd,led_g_tstd,led_b_tstd,led_r_sstd,led_g_sstd,led_b_sstd,frames\n";
	std::vector<SweepRecord> sorted = _tvRecords;
	std::sort(sorted.begin(), sorted.end(), [](const SweepRecord &a, const SweepRecord &b)
			  { return a.idx < b.idx; });
	for (const auto &r : sorted)
	{
		f << r.idx << "," << static_cast<int>(r.drive[0]) << "," << static_cast<int>(r.drive[1])
		  << "," << static_cast<int>(r.drive[2]);
		for (int roi = 0; roi < 2; roi++)
		{
			for (int c = 0; c < 3; c++)
				f << "," << r.mean[roi][c];
			for (int c = 0; c < 3; c++)
				f << "," << r.tstd[roi][c];
			for (int c = 0; c < 3; c++)
				f << "," << r.sstd[roi][c];
		}
		f << "," << r.frames << "\n";
	}
	std::cout << "Wrote " << path << " (" << sorted.size() << " patches)" << std::endl;
}

void LedCalib::writeLedSweep()
{
	const std::string path = _cfg.outDir + "/led_sweep.csv";
	std::ofstream f(path);
	f << "drive_r,drive_g,drive_b,led_r,led_g,led_b,led_r_tstd,led_g_tstd,led_b_tstd,"
		 "led_r_sstd,led_g_sstd,led_b_sstd,tv_r,tv_g,tv_b,frames\n";
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
		for (int c = 0; c < 3; c++)
			f << "," << r.mean[0][c];
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
	f << "  \"tv_patch_count\": " << _cfg.tvPatchCount << ",\n";
	f << "  \"tv_patches_decoded\": " << _tvDecoded << ",\n";
	f << "  \"measure_frames\": " << _cfg.measureFrames << ",\n";
	f << "  \"settle_ms\": " << _cfg.settleMs << ",\n";
	f << "  \"have_white_anchor\": " << (_haveAnchor ? "true" : "false") << ",\n";
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
	std::cout << "LedCalib finished (" << (code == EXIT_SUCCESS ? "ok" : "failed") << ")" << std::endl;
	if (_exit)
		_exit(code);
}
