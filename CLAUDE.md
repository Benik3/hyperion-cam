# hyperion-cam

Low-latency camera grabber for HyperHDR ambilight. Captures the TV picture with a Raspberry Pi camera (libcamera), perspective-warps it to a rectangle, and streams it to HyperHDR over a flatbuffer socket. **Latency beats everything**: dropping frames and picture tearing are acceptable if they reduce delay. Targets RPi4 (primary) and RPi5.

Usage/settings docs live in the GitHub Wiki (not in this repo). `doc/` holds design docs only (currently `led-lut-calibration.md`).

## Build

```bash
cmake -B build -G Ninja && cmake --build build -j4     # dev build
cmake --build Release -j4                              # production build tree
```

- `-Ofast -fno-fast-math`, C++17, `-Werror`.
- GPU transform (EGL/GLES3) compiles in only if `egl` + `glesv2` pkg-config packages exist (`HAVE_GPU_TRANSFORM`).
- Deps: libcamera, libevent_pthreads, Boost.program_options, OpenCV, flatbuffers (vendored headers in `src/flatbuffers/`).
- 32-bit ARM needs explicit `-latomic` (handled in CMakeLists).

## Architecture / data path

```
sensor → ISP → libcamera Request complete
  → requestComplete() [libcamera CameraManager thread]
      direct dispatch (default): processRequest() runs right here
      (--no-direct-dispatch / imgout / autocalib: hop via _loop eventfd)
  → processRequest(): drop check (_pendingProcess >= max → drop frame),
      enqueue lambda to _loopTr (transform thread, core 2, SCHED_FIFO)
      [no-transform mode: enqueue to _loopFb send thread instead]
  → transform thread: build flatbuffer request with uninitialized vectors,
      GPU warp writes straight into it, then FUSED send on the same thread
  → FlatBufferConnection: writev(header+payload), drain replies non-blocking
```

Key files:
- `src/hyperion-cam.cpp` — everything: CLI, camera setup, dispatch, calibration modes
- `src/gpuNv12Transform.cpp` — EGL/GLES3 NV12 warp (production path)
- `src/gpuRgbTransform.cpp` — RGB warp (calib / non-NV12 GPU path)
- `src/opencvTransform.cpp` — CPU fallback (`--cpuonly`), precomputed cv::remap maps
- `src/FlatBufferConnection.cpp` — socket, registration, send, reply drain
- `src/event_loop.h/.cpp` — libevent loop with small-buffer-optimized `callLater`
- `src/ledCalib.cpp` — LED 3D-LUT calibration measurement engine (`--ledcalib`; design + procedure in `doc/led-lut-calibration.md`)

### Latency design decisions (deliberate — don't "fix")

- `max-pending-process/send = 1` + drop-at-dispatch: latest-frame-wins. A mailbox scheme was considered and rejected — waiting for the next fresh frame beats sending a stale one.
- Fused transform+send on one thread: the in-flight slot is released only after the send completes, so backpressure drops frames instead of queueing them.
- Camera buffers are requeued immediately while the GPU may still read them via dmabuf → tearing possible, accepted.
- Sync (blocking) GPU readback: freshest data. Pipelined readback adds a frame of staleness.
- Threads: capture/dispatch core 1, transform/send core 2, SCHED_FIFO prio 10 (needs root; degrades gracefully). Core 0 left for kernel/IRQs. `mlockall(MCL_CURRENT|MCL_FUTURE)` at startup.

## Measuring (always measure before/after)

```bash
sudo systemctl stop hypercam
sudo ./build/hyperion-cam <prod args> --latency-stats        # p50/p95/p99 sensor→sent, 1/s on stderr
sudo HYPERCAM_GL_PROFILE=1 ...                               # GL stage breakdown, 1/s
sudo HYPERCAM_GL_PROFILE=1 HYPERCAM_GL_PROFILE_SYNC=1 ...    # glFinish between stages → true per-stage cost
sudo systemctl start hypercam                                # restore!
```

- `--latency-stats` measures libcamera `SensorTimestamp` (CLOCK_BOOTTIME) → send-complete. First 1-2 windows are startup noise (EGL init, ~100 drops) — ignore them.
- Reference numbers (RPi4, prod config, 2026-06): p50 16.5 ms, p95 18.4 ms, GPU render 1.94 ms/frame. p50 floor is sensor readout (~13.9 ms at 72 fps) + ISP; the software-controlled slice is ~2.5 ms.
- `HYPERCAM_CPU_PROFILE=1` profiles the CPU transform paths.

### Measured dead ends — do not retry without new evidence

- **Sync-PBO readback**: slower than direct `glReadPixels` (render 2.4→3.3 ms). Mesa v3d de-tiles on the CPU either way; the PBO adds a copy. Left opt-in: `HYPERCAM_GL_SYNCPBO=1`.
- **Combined Y+UV single FBO + single readback**: no gain; the apparent per-call fixed cost was GPU-wait, already pipelined. Reverted.
- **`HYPERCAM_NV12_PBO` (double-buffered readback)**: hangs — process alive but zero frames. Pre-existing, env-gated, unfixed.
- v3d `glReadPixels` cost scales with **pixel count**, not bytes → that's why RGBA-packing works.
- **Tetrahedral 3D-LUT in the NV12 shader**: +8 ms p50 (render 1.95→9.5 ms) — 16 dependent `texelFetch`es + branches per packed fragment kill v3d. HW-**trilinear** LUT costs only +1 ms and is size-insensitive (17³≈64³); use a denser grid instead of tetra. Hooks: `HYPERCAM_LUT_TEST/_TRILINEAR/_SIZE`.

## GL transform details (gpuNv12Transform)

- Homography is evaluated **in the vertex shader** and interpolated as homogeneous coords across the fullscreen triangle (mathematically exact — each component is linear in screen position). Fragment shader: one divide + texture fetch. Moving math back per-fragment costs ~0.7 ms/frame.
- **RGBA-packed readback** (default): 4 Y pixels (or 2 UV pairs) per RGBA8 texel, ¼ the pixels to de-tile. Output bytes are identical NV12 (verified A/B: ~0.004% pixels differ from NEAREST rounding at texel boundaries). Requires `dstW % 4 == 0`; `HYPERCAM_GL_NO_PACK=1` falls back.
- All uniforms set once at init (constant for transform lifetime).
- Input upload: zero-copy dmabuf import (`EGL_EXT_image_dma_buf_import`, R8 + GR88 planes), ~0.04 ms. Falls back to glTexSubImage2D.
- EGL: headless pbuffer; falls back to Mesa surfaceless platform.

## Footguns (learned the hard way)

- **FlatBuffers `PreAlign(n, 1)` does NOT reserve capacity** — alignment-1 padding is 0 bytes. To pre-size a builder, create a throwaway `CreateUninitializedVector` then `Clear()` (Clear keeps the allocation). Getting this wrong dangles the `CreateUninitializedVector` out-pointers when the buffer grows mid-build → segfault in the GPU readback that writes through them.
- The builder in the fused path is `thread_local` and reused per frame; pointers into it are valid only until the next `Clear()`.
- HyperHDR replies to **every** image message. If replies are never read (skip-reply mode) the socket buffer fills over hours → server stalls. `drainReplies()` (non-blocking, after each send) handles this; it also detects peer-close instantly.
- stdout is block-buffered when piped to a file — "process printed nothing after startup" usually means buffering, not a hang. Diagnostics print to stderr.
- `Camera::queueRequest()` from the `requestCompleted` handler is safe (async internally); heavy work (imwrite, autocalib) must NOT run on that thread — that's why those modes force the event-loop hop.
- When testing on this Pi: `pgrep -x hyperion-cam` (plain `pgrep -f` catches the sudo/bash wrappers). Wait ~2 s after killing before re-acquiring the camera.

## Calibration modes (one-shot, exit when done)

`--autowb`, `--autogain`, `--autocalib` (WB then gain), `--autosat [R,G,B]`, `--autofocus-once`, `--hyperhdr-calib` (forces NV12 1920x1080 16:9 output, INTER_LINEAR), `--imgout` / `--imgout-crophelp` (still capture; crophelp draws the sensor-crop rectangle). These paths are not latency-critical; they keep the event-loop dispatch and use per-frame camera controls. Add `--calib-tv-web` to make autocalib **web-driven** (reuses `LedCalibWeb`): the page shows the target patch on the TV — gray 127 for WB/gain, the `--autosat` color for sat — and you tap *begin*; the convergence machine is unchanged. Default full-screen patch + edge-strip ROI; `--calib-tv-roi`/`--patch-size` to window it. Measurement is **not** shaded (camera-ISP stage, before shading). Run `--autocalib` then `--autosat` separately.

`--ledcalib <phase>` — LED 3D-LUT calibration measurement; **no `all` phase, run `tv` and `led` separately** (avoids unshaded-`all` misuse). Phases: `tv` (cropped+shaded TV sweep, `--calib-tv-roi`), `led` (full-frame LED sweep, `--calib-led-roi`; `--calib-tv-roi` ignored), `shading`, `gamma`, `white` (manual eye white-balance, no ROI/measurement — see below). Outputs CSV/JSON into `--calib-out` (use the SAME dir for tv+led so the fit reads both); never streams camera frames (drives LEDs via synthetic solid frames). The LED sweep is a `--ledcalib-led-grid`³ cube (for the mixing matrix) + gray ramp + a fine per-channel axis ramp (`--ledcalib-led-axis-steps`, default 24, edge-dense near 0/255) that resolves the dark-crush and bright-clip transitions cheaply (1D, not the cube). TV patches are **web-only** (`--calib-tv-web`, required for tv/gamma/shading): grabber runs an evhttp+SSE server (`--calib-tv-port`, default 8080); open `http://<pi>:8080/` fullscreen in a browser on a PC on the TV's HDMI. Grabber commands each patch, page acks after paint → settle starts on ack. Page also hosts the white nudge UI + a video-based anti-screensaver (a muted, transparent (`opacity:0`) `<video>` playing a tiny low-frame-rate canvas-capture MediaStream earns the browser/OS media power-save blocker over plain HTTP, where the Screen Wake Lock API is unavailable; Firefox grants the blocker to an invisible *MediaStream* video — unlike a file, which must be visible and which the fullscreen root stretches over the patch — and the stream is driven by a ~4fps `setInterval`, not a 60fps `requestAnimationFrame` loop, so it can't hang weak TV browsers) and holds the first patch `--calib-start-delay-ms` (default 5s) after *begin* so the browser's fullscreen prompt clears. `src/ledCalibWeb.cpp` (libevent_extra). Requires the calib PC on the SAME TV input/picture-mode as the real source.

`--ledcalib shading` (web-driven) — ALSC/shading probe: full-screen safe gray (`--shading-level`, default 180), samples a `--shading-grid` (UxV) across the TV quad (keeps `--crop` + `--transform`, samples the frame via the inverse homography in software), writes `shading.csv` (per-cell RGB + darken-to-dimmest gains) and prints the luminance falloff + center→edge color drift. Apply the map at runtime with `--shading-map shading.csv`: uploaded as a bilinear RGBA8 gain texture, multiplied per output position in the NV12 warp shader (YUV→RGB→×gain→YUV, limited-range BT.601). Cost ~+0.7 ms render / +0.5 ms p50; zero when not passed. NV12 GPU path only. **Texture orientation VERIFIED on a real TV (2026-06-21, 4-corner colour probe in HyperHDR live view — no row flip). Gain magnitude still needs a visual check on the real TV (uniform gray should get flatter, not worse).** The TV-side calib (`--ledcalib tv`/`gamma`) keeps `--crop` and, with `--shading-map`+`--transform`, scales the TV-ROI reading by the shading gain at that ROI (homography-mapped) so the LUT input domain matches shaded production (warp itself is skipped — geometric, doesn't change a uniform patch); `--transform`/`--crop` optional if the camera frame is only the clean TV. `led` is full-frame and ignores `--shading-map` (wall stays uncorrected — its constant shading cancels between the sweep whites).

**Deferred fix — shading min (flagged in the 2026-06 review).** `writeShadingMap` normalizes "darken-to-dimmest" using the strict per-channel **minimum** cell mean as the gain numerator, so a single near-black cell — e.g. the `--transform` quad overhanging the lit TV onto the bezel/letterbox, or a letterboxed source — collapses every `gain_c = min/mean` toward 0 and blacks that channel out across the whole frame at runtime. It is currently only **warned** (from `writeShadingMap`, when a channel's min is < 33% of its brightest cell), not corrected. The real fix, if needed, is a **robust target** — normalize to a low percentile / median-relative floor instead of the strict min, and/or inset the sampled cell ROIs so corners don't catch the border — but it changes the produced map and this map's magnitude is still unvalidated on a real TV, so A/B it there first. The failure is opt-in (only with `--shading-map`) and immediately visible in HyperHDR live view.

`--ledcalib gamma` (web-driven) — diagnostic R/G/B/gray ramp probe of the TV→camera transfer (crush/clip/crosstalk); writes `gamma.csv`. `--gamma-raw` reruns it on the sensor Bayer stream (`SBGGR10_CSI2P`, BGGR, MIPI RAW10 unpack + Bayer-plane ROI averaging) → `raw_gamma.csv` for an ISP-vs-RAW A/B at the ends. `--ledcalib-gamma-selftest` unit-tests the RAW10 unpack (no camera). `tools/plot-gamma.py` (stdlib→SVG) overlays both. As a **pre-flight**, the gamma run auto-analyzes each ramp and prints per-channel `WARNING` lines when the camera **clips** (reading pins near full scale — lower exposure/brightness) or **crushes** (dark end stuck in the floor) with the usable command range, so a bad exposure/TV-brightness is caught before the long `tv`/`led` sweeps rather than after.

`--ledcalib white` (web or tty) — purely **manual** white-balance aid: drives the LEDs full white (and sends a solid white frame to HyperHDR) so you can eye-match the LED glow to the TV white in HyperHDR's white-balance settings. No camera ROI, no measurement, no file output — Enter/Accept just exits. The metameric white correction is derived later from the sweeps (below), not from this phase.

`--ledcalib-fit` (offline, no camera) — fit `--calib-out` sweeps into a `.cube` LUT (`src/ledCalibFit.cpp`); `--ledcalib-fit-selftest` verifies the math on a simulated chain. White correction is always derived from the sweep white points (cam_tv_white from `tv_sweep.csv` / cam_led_white from `led_sweep.csv`) at the brightest neutral level where neither clips (camera-clip detection); set HyperHDR white balance before measuring (use `--ledcalib white` for that). There is no longer a `white_anchor.json`. The fit also detects **per-channel sensor clipping** in `led_sweep.csv` (a channel whose reading pins near the camera's full-scale ceiling across many samples — e.g. bright R/B, while a WB-attenuated G that maxes well below is kept) and **excludes those saturated samples** so they can't bias the drive curves or the `e(255)=1` normalization; the curve then holds flat above the last responsive drive (conservative — that channel's bright output reaches max a touch early rather than being mis-scaled). Detected ceilings + excluded counts go to `fit_report.json` (`clip_ceiling`/`clip_excluded`/`sweep_floor`) and a warning prints per clipped channel (lower exposure or that channel's brightness to use its full range). This changes the produced LUT on a clipping rig, so A/B on the real TV.

**Deferred fix — `corr` clamp (flagged in the 2026-06 review).** The von-Kries white correction `corr[c] = cam_tv_white[c] / modelled_led_white[c]` (in `fitModel`, `src/ledCalibFit.cpp`) blows up when a channel's modelled LED white sits near the camera floor — a *severely* WB-attenuated channel, well past the normal ~0.25 green attenuation. The huge `corr[c]` drives that channel's targets negative in `invertNode`, crushing it across the whole LUT (colour cast). It is currently only **warned** (from `fitModel`), not corrected. The real fix, if needed, is to **clamp `corr[c]` to a sane max**; it changes output only in that attenuated regime — which the fit self-test's synthetic chain does not exercise — so A/B it on the real TV+LED rig before landing. Don't try to fix it at anchor selection instead: when the white is that imbalanced, no neutral level is both unclipped *and* above the floor (the clip ceiling and the noise floor squeeze the feasible window empty), so it can only detect-and-bail and risks silently dropping the white correction entirely.

Apply at runtime: `--lut <file.cube>` (trilinear 3D LUT) and/or `--shading-map shading.csv` — one unified colour-correction shader in `gpuNv12Transform` composes them (YUV→RGB→×shade→LUT→YUV, limited-range BT.601, units Y/UV=0/1 LUT=2 shade=3). Cost: LUT +0.9 ms, LUT+shading +1.1 ms p50; zero when neither passed. NV12 GPU path only. Full procedure: `doc/calibration-guide.md` (how-to) + `doc/led-lut-calibration.md` (design).
