#ifndef __LED_CALIB_WEB_H__
#define __LED_CALIB_WEB_H__

#include <cstdint>
#include <memory>

// Key the page's 'begin' tap is delivered as (via popKey). Distinct from
// Enter/'\n' ('accept') so that reloading the page mid-phase and tapping
// begin again cannot fake an accept (e.g. instantly "confirming" the White
// phase). Consumers treat it like Enter in the wait phases only.
constexpr char kCalibBeginKey = 'B';

// Lightweight web display + control surface for --ledcalib (doc/led-lut-calibration.md,
// "Web TV-patch path"). A PC connected to the TV opens this page in a fullscreen
// browser; the grabber pushes the fullscreen color over SSE and the page acks
// once it has painted, so the settle timer covers only TV-internal latency.
// In Phase 0 the page also hosts the LED white-balance nudge buttons.
//
// Runs its own libevent event_base on a dedicated thread. All evhttp work
// happens on that thread; cross-thread state is mutex/atomic guarded. The
// engine (LedCalib) calls setState()/ackedSeq()/popKey() from the camera loop.

class LedCalibWeb
{
public:
	explicit LedCalibWeb(int port);
	~LedCalibWeb();

	LedCalibWeb(const LedCalibWeb &) = delete;
	LedCalibWeb &operator=(const LedCalibWeb &) = delete;

	/// Bind the socket and start the server thread. Returns false (and prints)
	/// if the port can't be bound.
	bool start();

	int port() const;

	/// Hide the corner keep-alive square + status text (they'd corrupt a
	/// full-frame measurement like shading). Off by default.
	void setCleanField(bool clean);

	/// Window-patch geometry: pct = patch area as a percentage of the screen
	/// (100 = fullscreen), x/y = top-left position in screen-percent (<0 =
	/// centered). Constant for a run; call once before/after start().
	void setPatchGeometry(double pct, double x, double y);

	/// Set what the page shows. seq must strictly increase whenever a new
	/// patch needs a fresh ack; pass the same seq to merely update the overlay
	/// (e.g. nudge feedback) without expecting a new ack. showControls draws
	/// the Phase 0 nudge UI and the led* values.
	void setState(uint8_t r, uint8_t g, uint8_t b, bool showControls,
				  uint8_t ledR, uint8_t ledG, uint8_t ledB, uint32_t seq);

	/// Highest seq the page has confirmed painting (0 = none yet).
	uint32_t ackedSeq() const;

	/// True once a browser has an SSE stream open.
	bool clientConnected() const;

	/// Drain one queued key from the page (nudge/accept/start/quit), mapped to
	/// the same characters the stdin reader produces. Returns false if empty.
	bool popKey(char &out);

	// Opaque implementation; public only so the file-scope evhttp callbacks in
	// the .cpp can name it. Not part of the API.
	struct Impl;

private:
	std::unique_ptr<Impl> _d;
};

#endif /* __LED_CALIB_WEB_H__ */
