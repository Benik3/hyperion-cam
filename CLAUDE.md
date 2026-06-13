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

`--autowb`, `--autogain`, `--autocalib` (WB then gain), `--autosat [R,G,B]`, `--autofocus-once`, `--hyperhdr-calib` (forces NV12 1920x1080 16:9 output, INTER_LINEAR), `--imgout` / `--imgout-crophelp` (still capture; crophelp draws the sensor-crop rectangle). These paths are not latency-critical; they keep the event-loop dispatch and use per-frame camera controls.

`--ledcalib [all|white|tv|led]` — LED 3D-LUT calibration measurement. Requires `--calib-tv-roi` and `--calib-led-roi` (raw full-frame coords); ignores `--transform`/`--crop`, never streams camera frames (drives LEDs via synthetic solid frames instead). Outputs measurement CSV/JSON into `--calib-out`. Two ways to drive the TV patches:
  - `--calib-tv-web` (recommended): grabber runs an evhttp+SSE server (`--calib-tv-port`, default 8080); open `http://<pi>:8080/` fullscreen in a browser on a PC on the TV's HDMI. Grabber commands each patch, page acks after paint → settle starts on ack. Page also hosts Phase 0 nudge UI. No video, no decoder. `src/ledCalibWeb.cpp` (libevent_extra). Requires the calib PC on the SAME TV input/picture-mode as the real source.
  - calib video (fallback, needs a tty): TV plays `tools/gen-calib-video.py` whose index-code colors/timings must match `src/ledCalib.cpp`.

`--ledcalib-fit` (offline, no camera) — fit `--calib-out` sweeps into a `.cube` LUT (`src/ledCalibFit.cpp`); `--ledcalib-fit-selftest` verifies the math on a simulated chain. Full procedure + math: `doc/led-lut-calibration.md`.
