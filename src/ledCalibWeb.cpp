#include "ledCalibWeb.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>

namespace
{
	// Fullscreen color display + Phase 0 nudge UI. Connects SSE immediately;
	// fullscreen is attempted at load (works when the browser was launched
	// fullscreen/kiosk or has the permission) and re-attempted on the first
	// gesture / the begin tap as a fallback (most browsers reject the
	// Fullscreen API without a user gesture).
	// After each painted frame it double-rAFs, then acks the seq so the grabber
	// knows the TV actually shows the patch before it starts the settle timer.
	const char *kPageHtml = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hyperion-cam LED calib</title>
<style>
  html,body{margin:0;height:100%;background:#000;overflow:hidden;font-family:sans-serif}
  /* #fill is the black surround; #patch is the (optionally windowed) colour. */
  #fill{position:fixed;inset:0;background:#000}
  #patch{position:fixed;left:0;top:0;width:100%;height:100%;background:#000}
  #start{position:fixed;inset:0;background:#111;cursor:pointer;z-index:3}
  /* example patch outline so the user can screenshot (--imgout) and place the ROI */
  #example{position:absolute;box-sizing:border-box;background:#555;border:4px solid #e22}
  #starttext{position:absolute;left:0;right:0;top:3vh;text-align:center;color:#ccc;
             font-size:3vh;padding:0 4vh;text-shadow:0 0 4px #000}
  #starttext small{font-size:1.9vh;color:#bbb}
  #ui{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:2;
      background:rgba(0,0,0,.55);color:#fff;padding:2vh 3vh;border-radius:1vh;display:none}
  #ui.on{display:block}
  #ui h2{margin:.2em 0;font-size:2.4vh;font-weight:normal}
  #ui .drv{font-size:3vh;margin:.4em 0;font-variant-numeric:tabular-nums}
  #ui button{font-size:2.6vh;margin:.2vh;padding:.4em .7em;min-width:2.6em}
  #ui .accept{background:#2a7;color:#fff;border:none}
  /* Dim gray, low luminance, to limit OLED retention of the static label. */
  #st{position:fixed;left:1vh;bottom:1vh;color:#555;font-size:1.6vh;z-index:2;
      font-variant-numeric:tabular-nums}
  /* keep-alive: a muted <video> playing a tiny canvas-capture MediaStream. A
     playing video makes the browser/OS treat the page like a media player (as
     YouTube does) and inhibit the screensaver/display sleep over plain HTTP,
     where the Screen Wake Lock API (HTTPS-only) is unavailable. opacity:0 keeps
     it out of the camera's view of the TV (and harmless even if fullscreen
     stretches it), so it can never cover or corrupt a patch. Firefox grants the
     blocker to an invisible *MediaStream* video (a file would have to be
     visible); the stream is driven at a low frame rate so it can't bog down a
     weak TV browser the way a per-frame canvas loop did. */
  #keep{position:fixed;left:0;top:0;width:32px;height:32px;z-index:0;
        pointer-events:none;opacity:0}
</style></head>
<body>
<div id="fill"></div>
<div id="patch"></div>
<video id="keep" muted playsinline></video>
<div id="ui">
  <h2>Make the LED glow match the TV white — tune HyperHDR white balance, or nudge:</h2>
  <div class="drv">R <span id="dr">0</span>&nbsp; G <span id="dg">0</span>&nbsp; B <span id="db">0</span></div>
  <div>
    <button onclick="k('r-')">R&minus;</button><button onclick="k('r+')">R+</button>
    <button onclick="k('g-')">G&minus;</button><button onclick="k('g+')">G+</button>
    <button onclick="k('b-')">B&minus;</button><button onclick="k('b+')">B+</button>
  </div>
  <div style="margin-top:1vh">
    <button onclick="k('step')">step</button>
    <button class="accept" onclick="k('accept')">Accept</button>
  </div>
  <div style="margin-top:.6em;font-size:1.7vh;color:#aaa">keys: 1/2 3/4 5/6 = R/G/B &minus;/+ , 0 step, Enter accept</div>
</div>
<div id="st">connecting&hellip;</div>
<div id="start" onclick="go()">
  <div id="example"></div>
  <div id="starttext">Tap / click to begin (goes fullscreen)<br>
    <small id="starthint">The red rectangle marks the patch area — screenshot it now
    (run <b>--imgout</b> while this is shown) and set <b>--calib-tv-roi</b> inside it.</small>
  </div>
</div>
<script>
let lastSeq=-1, last=null;
const fill=document.getElementById('fill'), patch=document.getElementById('patch'),
      ui=document.getElementById('ui'), st=document.getElementById('st'),
      example=document.getElementById('example'), hint=document.getElementById('starthint');
const keep=document.getElementById('keep');
function k(c){fetch('/key?c='+encodeURIComponent(c),{method:'POST'}).catch(()=>{});}
// Enter fullscreen from the begin tap only. Browsers (Firefox especially)
// reject requestFullscreen() unless it is called directly from a single user
// gesture, and even a caught rejection logs a console warning — so we must NOT
// call it on load or from extra listeners (those just error and, worse, a
// duplicate request racing the tap makes the real one fail). To have the page
// already fullscreen on the first screen, launch the browser fullscreen/kiosk
// (F11 / --kiosk): it is then fullscreen at the browser level, no API needed.
function tryFs(){ const el=document.documentElement;
  if(!document.fullscreenElement && el.requestFullscreen) el.requestFullscreen().catch(()=>{}); }
// Keep the display awake by playing a tiny canvas-capture MediaStream. A live
// video stream earns the browser/OS media power-save blocker over plain HTTP
// even while invisible (Firefox grants it to a MediaStream regardless of
// visibility, unlike a file). Kept deliberately cheap — a 16x16 canvas redrawn
// a few times a second via setInterval, NOT a 60fps requestAnimationFrame loop
// — so it can't bog down / hang a weak TV browser over a long sweep. Started
// from the begin tap (browsers block play() before a gesture); retried on
// refocus in case the browser paused background media.
let keepWanted=false, keepInit=false;
function keepAwake(){
  keepWanted=true;
  try{
    if(!keepInit){ keepInit=true;
      const c=document.createElement('canvas'); c.width=c.height=16;
      const g=c.getContext('2d'); let n=0;
      const draw=()=>{ const v=8+(n++&3); g.fillStyle='rgb('+v+','+v+','+v+')'; g.fillRect(0,0,16,16); };
      draw();
      keep.srcObject=c.captureStream(4);   // ~4 fps: live enough to hold the lock, cheap
      keep.muted=true; keep.playsInline=true;
      setInterval(draw, 250);              // tiny periodic change keeps the stream live
    }
    keep.play().catch(()=>{});
  }catch(e){}
}
document.addEventListener('visibilitychange',()=>{
  if(keepWanted && document.visibilityState==='visible') keep.play().catch(()=>{}); });
function go(){
  const s=document.getElementById('start'); s.style.display='none';
  tryFs();
  keepAwake();
  k('start');
}
function layout(s){
  const W=window.innerWidth, H=window.innerHeight;
  let bw=W, bh=H, lx=0, ly=0;
  if(s.wp<100){
    const f=Math.sqrt(Math.max(s.wp,0.01)/100);   // patch area = wp% of screen
    bw=Math.round(W*f); bh=Math.round(H*f);
    lx = s.wx<0 ? Math.round((W-bw)/2) : Math.min(Math.max(Math.round(W*s.wx/100),0),W-bw);
    ly = s.wy<0 ? Math.round((H-bh)/2) : Math.min(Math.max(Math.round(H*s.wy/100),0),H-bh);
  }
  for(const el of [patch, example]){
    el.style.left=lx+'px'; el.style.top=ly+'px';
    el.style.width=bw+'px'; el.style.height=bh+'px';
  }
  hint.textContent = 'patch '+(s.wp>=100?'fullscreen':s.wp+'% of screen, '+bw+'×'+bh+' px')+
    ' — screenshot it now (run --imgout) and set --calib-tv-roi inside the red rectangle.';
}
window.addEventListener('resize',()=>{ if(last) layout(last); });
document.addEventListener('keydown',e=>{
  const m={'1':'r-','2':'r+','3':'g-','4':'g+','5':'b-','6':'b+','0':'step'};
  if(e.key in m) k(m[e.key]);
  else if(e.key==='Enter') k('accept');
  else if(e.key==='q') k('quit');
});
const es=new EventSource('/events');
es.onopen=()=>{st.textContent='connected';};
es.onerror=()=>{st.textContent='disconnected — is the grabber running?';};
// The grabber ends this stream when the page is opened elsewhere (last
// connection wins). Close so EventSource doesn't auto-reconnect and steal
// the stream back from the new tab.
es.addEventListener('replaced',()=>{es.close();st.style.display='';
  st.textContent='replaced — the page was opened elsewhere';});
es.onmessage=ev=>{
  let s; try{s=JSON.parse(ev.data);}catch(_){return;}
  last=s;
  patch.style.background='rgb('+s.r+','+s.g+','+s.b+')';
  // Hide the status text during a clean full-frame measurement. The keep-awake
  // video is transparent (opacity:0) so it never corrupts a sample and keeps
  // playing through every phase.
  st.style.display = s.clean ? 'none' : '';
  layout(s);
  if(s.controls){ui.classList.add('on');
    document.getElementById('dr').textContent=s.lr;
    document.getElementById('dg').textContent=s.lg;
    document.getElementById('db').textContent=s.lb;}
  else ui.classList.remove('on');
  st.textContent=s.seq+': '+s.r+', '+s.g+', '+s.b;
  if(s.seq!==lastSeq){ lastSeq=s.seq;
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      fetch('/ack?n='+s.seq,{method:'POST'}).catch(()=>{});
    }));
  }
};
</script>
</body></html>
)HTML";

	char keyToken(const std::string &c)
	{
		// Map page tokens to the characters the stdin reader / whiteKey expect.
		if (c == "r-")
			return '1';
		if (c == "r+")
			return '2';
		if (c == "g-")
			return '3';
		if (c == "g+")
			return '4';
		if (c == "b-")
			return '5';
		if (c == "b+")
			return '6';
		if (c == "step")
			return '0';
		if (c == "accept")
			return '\n';
		if (c == "start")
			return kCalibBeginKey; // begin != accept: a reload + begin must not confirm a phase
		if (c == "quit")
			return 'q';
		return 0;
	}
}

struct LedCalibWeb::Impl
{
	int port;
	event_base *base = nullptr;
	evhttp *http = nullptr;
	std::thread th;
	std::atomic<bool> running{false};
	std::atomic<bool> startFailed{false};

	// state shown on the page (guarded by sm)
	std::mutex sm;
	uint8_t color[3] = {0, 0, 0};
	uint8_t led[3] = {0, 0, 0};
	bool controls = false;
	uint32_t seq = 0;
	bool dirty = true;
	double wp = 100.0, wx = -1.0, wy = -1.0; // window-patch geometry (% / %pos)
	bool clean = false;						 // hide keep-alive + status (shading)

	std::atomic<uint32_t> acked{0};
	std::atomic<bool> hasClient{false};

	// keys to the engine
	std::mutex km;
	std::deque<char> keys;

	// single active SSE stream (last connection wins); base-thread only.
	evhttp_request *sse = nullptr;
	event *tick = nullptr;
	int heartbeatCtr = 0;

	explicit Impl(int p) : port(p) {}

	void pushKey(char c)
	{
		if (!c)
			return;
		std::lock_guard<std::mutex> lk(km);
		keys.push_back(c);
	}

	std::string buildSseData()
	{
		std::lock_guard<std::mutex> lk(sm);
		char buf[256];
		std::snprintf(buf, sizeof(buf),
					  "data: {\"r\":%d,\"g\":%d,\"b\":%d,\"seq\":%u,\"controls\":%s,\"lr\":%d,\"lg\":%d,\"lb\":%d,"
					  "\"wp\":%.3f,\"wx\":%.3f,\"wy\":%.3f,\"clean\":%s}\n\n",
					  color[0], color[1], color[2], seq, controls ? "true" : "false",
					  led[0], led[1], led[2], wp, wx, wy, clean ? "true" : "false");
		return buf;
	}
};

// ---- evhttp callbacks (all on the base thread) ----

namespace
{
	void cbRoot(evhttp_request *req, void *arg)
	{
		(void)arg;
		evbuffer *out = evhttp_request_get_output_buffer(req);
		evhttp_add_header(evhttp_request_get_output_headers(req), "Content-Type", "text/html; charset=utf-8");
		evbuffer_add(out, kPageHtml, std::strlen(kPageHtml));
		evhttp_send_reply(req, 200, "OK", out);
	}

	void cbClose(evhttp_connection *conn, void *arg)
	{
		auto *d = static_cast<LedCalibWeb::Impl *>(arg);
		// "Last connection wins": a newer /events stream may have replaced this
		// one while this (older) connection's closecb is still registered. Only
		// clear the shared pointer when the closing connection still owns it, so
		// an old connection closing late can't kill the live client's stream.
		if (d->sse && evhttp_request_get_connection(d->sse) != conn)
			return;
		d->sse = nullptr;
		d->hasClient.store(false, std::memory_order_release);
	}

	void cbEvents(evhttp_request *req, void *arg)
	{
		auto *d = static_cast<LedCalibWeb::Impl *>(arg);
		evkeyvalq *oh = evhttp_request_get_output_headers(req);
		evhttp_add_header(oh, "Content-Type", "text/event-stream");
		evhttp_add_header(oh, "Cache-Control", "no-cache");
		evhttp_add_header(oh, "Connection", "keep-alive");

		// Replace any previous stream (last connection wins). Tell the old tab
		// it was replaced — the page closes its EventSource on this event (so
		// it won't auto-reconnect and steal the stream back) — then finish the
		// old reply so that tab isn't left a silent half-open stream forever
		// showing "connected".
		if (d->sse && d->sse != req)
		{
			evhttp_request *old = d->sse;
			d->sse = nullptr;
			evhttp_connection *oc = evhttp_request_get_connection(old);
			if (oc)
				evhttp_connection_set_closecb(oc, nullptr, nullptr);
			evbuffer *bye = evbuffer_new();
			const char *msg = "event: replaced\ndata: 1\n\n";
			evbuffer_add(bye, msg, std::strlen(msg));
			evhttp_send_reply_chunk(old, bye);
			evbuffer_free(bye);
			evhttp_send_reply_end(old);
		}
		d->sse = req;
		d->hasClient.store(true, std::memory_order_release);
		{
			std::lock_guard<std::mutex> lk(d->sm);
			d->dirty = true; // push current state on the next tick
		}
		evhttp_send_reply_start(req, 200, "OK");
		evhttp_connection *conn = evhttp_request_get_connection(req);
		if (conn)
			evhttp_connection_set_closecb(conn, cbClose, d);
	}

	void cbAck(evhttp_request *req, void *arg)
	{
		auto *d = static_cast<LedCalibWeb::Impl *>(arg);
		const char *uri = evhttp_request_get_uri(req);
		evkeyvalq q;
		uint32_t n = 0;
		if (uri && evhttp_parse_query(uri, &q) == 0)
		{
			const char *v = evhttp_find_header(&q, "n");
			if (v)
				n = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
			evhttp_clear_headers(&q);
		}
		uint32_t cur = d->acked.load(std::memory_order_relaxed);
		while (n > cur && !d->acked.compare_exchange_weak(cur, n, std::memory_order_acq_rel))
		{
		}
		evhttp_send_reply(req, 204, "No Content", nullptr);
	}

	void cbKey(evhttp_request *req, void *arg)
	{
		auto *d = static_cast<LedCalibWeb::Impl *>(arg);
		const char *uri = evhttp_request_get_uri(req);
		evkeyvalq q;
		if (uri && evhttp_parse_query(uri, &q) == 0)
		{
			const char *v = evhttp_find_header(&q, "c");
			if (v)
				d->pushKey(keyToken(v));
			evhttp_clear_headers(&q);
		}
		evhttp_send_reply(req, 204, "No Content", nullptr);
	}

	void cbTick(evutil_socket_t, short, void *arg)
	{
		auto *d = static_cast<LedCalibWeb::Impl *>(arg);
		if (!d->sse)
			return;
		bool dirty;
		{
			std::lock_guard<std::mutex> lk(d->sm);
			dirty = d->dirty;
			d->dirty = false;
		}
		if (dirty)
		{
			const std::string msg = d->buildSseData();
			evbuffer *buf = evbuffer_new();
			evbuffer_add(buf, msg.data(), msg.size());
			evhttp_send_reply_chunk(d->sse, buf);
			evbuffer_free(buf);
		}
		// SSE comment heartbeat (~every 5 s) so a dead peer is noticed via closecb.
		if (++d->heartbeatCtr >= 200)
		{
			d->heartbeatCtr = 0;
			evbuffer *buf = evbuffer_new();
			const char *hb = ": ping\n\n";
			evbuffer_add(buf, hb, std::strlen(hb));
			evhttp_send_reply_chunk(d->sse, buf);
			evbuffer_free(buf);
		}
	}
}

LedCalibWeb::LedCalibWeb(int port) : _d(std::make_unique<Impl>(port)) {}

LedCalibWeb::~LedCalibWeb()
{
	if (_d->running.load(std::memory_order_acquire) && _d->base)
		event_base_loopbreak(_d->base);
	if (_d->th.joinable())
		_d->th.join();
	if (_d->tick)
		event_free(_d->tick);
	if (_d->http)
		evhttp_free(_d->http);
	if (_d->base)
		event_base_free(_d->base);
}

bool LedCalibWeb::start()
{
	_d->base = event_base_new();
	if (!_d->base)
	{
		std::cout << "ERROR: ledcalib web: event_base_new failed" << std::endl;
		return false;
	}
	_d->http = evhttp_new(_d->base);
	if (!_d->http)
	{
		std::cout << "ERROR: ledcalib web: evhttp_new failed" << std::endl;
		return false;
	}
	evhttp_set_cb(_d->http, "/", cbRoot, _d.get());
	evhttp_set_cb(_d->http, "/events", cbEvents, _d.get());
	evhttp_set_cb(_d->http, "/ack", cbAck, _d.get());
	evhttp_set_cb(_d->http, "/key", cbKey, _d.get());

	if (evhttp_bind_socket(_d->http, "0.0.0.0", static_cast<uint16_t>(_d->port)) != 0)
	{
		std::cout << "ERROR: ledcalib web: cannot bind port " << _d->port
				  << " (in use? try --calib-tv-port)" << std::endl;
		return false;
	}

	timeval tv{0, 25000}; // 25 ms SSE push cadence
	_d->tick = event_new(_d->base, -1, EV_PERSIST, cbTick, _d.get());
	event_add(_d->tick, &tv);

	_d->running.store(true, std::memory_order_release);
	_d->th = std::thread([this]
						 { event_base_dispatch(_d->base); _d->running.store(false, std::memory_order_release); });
	return true;
}

int LedCalibWeb::port() const { return _d->port; }

void LedCalibWeb::setCleanField(bool clean)
{
	std::lock_guard<std::mutex> lk(_d->sm);
	_d->clean = clean;
	_d->dirty = true;
}

void LedCalibWeb::setPatchGeometry(double pct, double x, double y)
{
	std::lock_guard<std::mutex> lk(_d->sm);
	_d->wp = pct;
	_d->wx = x;
	_d->wy = y;
	_d->dirty = true;
}

void LedCalibWeb::setState(uint8_t r, uint8_t g, uint8_t b, bool showControls,
						   uint8_t ledR, uint8_t ledG, uint8_t ledB, uint32_t seq)
{
	std::lock_guard<std::mutex> lk(_d->sm);
	_d->color[0] = r;
	_d->color[1] = g;
	_d->color[2] = b;
	_d->led[0] = ledR;
	_d->led[1] = ledG;
	_d->led[2] = ledB;
	_d->controls = showControls;
	_d->seq = seq;
	_d->dirty = true;
}

uint32_t LedCalibWeb::ackedSeq() const { return _d->acked.load(std::memory_order_acquire); }

bool LedCalibWeb::clientConnected() const { return _d->hasClient.load(std::memory_order_acquire); }

bool LedCalibWeb::popKey(char &out)
{
	std::lock_guard<std::mutex> lk(_d->km);
	if (_d->keys.empty())
		return false;
	out = _d->keys.front();
	_d->keys.pop_front();
	return true;
}
