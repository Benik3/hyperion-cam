# LED 3D-LUT color calibration — design

Status: agreed design (2026-06-12). Build order: measurement first, runtime shader second,
fit/inversion third, verification last. This doc is the contract between those pieces.

## Goal

LED wall-glow color matches the TV picture **to the human eye**. Not absolute colorimetric
accuracy — relative match. The chain TV → camera → grabber → HyperHDR → LEDs → wall contains
three unknown nonlinear devices (TV, camera, LEDs); we never characterize any of them absolutely.

## Key insights the design rests on

1. **Camera as relative comparator.** The same camera + ISP (with locked exposure/WB/gain)
   measures both the TV picture and the LED wall-glow. The grabber's runtime output *is* the
   camera-space measurement of the TV. If we calibrate LED drive values so the camera sees the
   glow as a scaled version of the same camera-space value, the camera's own response cancels.
   The TV's miscalibration is irrelevant: the TV is the reference.
2. **Metamerism does not cancel.** LED spectra are narrow spikes; the camera's color filters are
   not human cones, so camera-match ≠ eye-match (observed in practice: LED white that looks
   correct to the eye reads strongly red to the camera). No global white balance can fix this —
   WB is one gain pair for the whole frame, but TV light and LED light need *different*
   corrections. Defense: a manual **eye-anchor** step; the user visually matches LED white to TV
   white once, and the resulting per-channel (von-Kries) correction is applied to all camera
   readings of LED light during calibration. Optional extra anchor colors can refine this to a
   3×3 matrix later.
3. **Runtime cost must be ~zero.** The correction is baked into a 3D LUT (default 17³) applied
   inside the existing GPU warp fragment shader — one tetrahedral interpolation per output pixel
   (trilinear available as env toggle for A/B). No RAW capture, no extra pass at runtime.

## Decisions (from design review)

| Decision | Choice | Why |
|---|---|---|
| Calibration target | LED glow matches TV, relative | absolute accuracy not the goal; cancels camera+TV unknowns |
| LUT location | grabber GPU shader, per-pixel | lowest latency; per-LED (post-averaging) would be mathematically cleaner but needs a HyperHDR fork |
| Interpolation | ~~tetrahedral~~ → **trilinear 33³** (measured 2026-06-12, see Runtime) | tetra costs +8 ms p50 on v3d; dense trilinear matches its accuracy at +1 ms |
| TV patch display | generated calib video + on-screen index coding | survives pause/seek/frame-rate conversion; no controllable HDMI source needed |
| Measurement loop | open-loop sweeps + closed-loop verification pass | sweeps are fast and clean; verification catches model errors |
| Eye anchor | white mandatory, extra colors optional | white is what users can match reliably |
| Luminance policy | proportional, anchored at matched white | eye adapts to dimmer glow; white anchor sets scale |
| Gamut overflow | clip hue-true (preserve hue, reduce sat/lum) | artifact-free |
| Measurement bit depth | 8-bit NV12-path RGB + frame/ROI averaging; **RAW only if measured variance proves insufficient** | LUT input domain must be runtime grabber space; averaging beats quantization; calib logs variance stats to settle this with evidence |
| Time budget | ~30–45 min full run, unattended except eye anchor | sparse sweeps + smooth model fit |

## Calibration procedure (`--ledcalib`)

One-shot mode like `--autocalib`: forces RGB capture, locks AE/AWB with the manual
shutter/gain/wbgain from the command line, disables ScalerCrop and perspective transform, never
streams camera frames to HyperHDR. The only thing it sends to HyperHDR is synthetic solid-color
frames (small RGB image over the existing flatbuffer socket) to drive all LEDs to one color.

The user provides two axis-aligned ROIs in raw full-frame coordinates (pick them from an
`--imgout` still taken without `--crop`):

- `--calib-tv-roi x,y,w,h` — entirely inside the TV picture area
- `--calib-led-roi x,y,w,h` — on the wall where LED glow is visible (repositioning the camera
  for the calibration run is allowed; reposition as little as possible — the TV shifts color
  off-axis, which skews the TV-side readings)

Requirements during the run: dark room, HyperHDR running with the grabber as the only active
source, HyperHDR color/smoothing settings in their final state (the LUT bakes them in — change
them later and the LUT is stale).

### Getting patches onto the TV: web page (default-recommended) vs calib video

Two interchangeable ways to drive the TV. Both feed the same measurement loop; pick at the CLI.

**Web path (`--calib-tv-web`, implemented 2026-06-12).** The grabber runs a tiny HTTP server
(evhttp, on `--calib-tv-port`, default 8080) and you open `http://<pi>:8080/` in a **fullscreen**
browser on a PC plugged into the TV. The grabber pushes the fullscreen color over SSE; the page
paints it, double-`requestAnimationFrame`s to guarantee the frame is up, then `POST /ack?n=seq`.
The grabber starts its settle timer **on the ack**, so timing covers only TV-internal latency, not
network/paint lag — and a closed/frozen page is caught by an 8 s ack timeout (clean abort). Because
the grabber *commands* each patch it never has to decode anything: the index-code state machine and
the calib video are bypassed entirely. The page also hosts the Phase 0 nudge buttons (below) so the
whole run is driven from the TV-side PC. `src/ledCalibWeb.cpp`.

**Hard requirement:** plug the calib PC into the **same HDMI input / picture mode** you normally
watch through. TVs apply different gamma/chroma/processing per input ("PC mode", "Game mode",
4:4:4 vs 4:2:0); the camera-as-comparator only holds if the calib display path gets the same TV
processing as runtime content. Browser color management is *not* the main risk (we measure the
camera's reading of whatever the TV actually shows, not the commanded value) — the per-input TV
processing difference is.

**Video path (fallback, `tools/gen-calib-video.py`).** For users with no spare PC for the TV: play
a generated video (USB stick / media app). Each patch is preceded by a base-6 index code the
grabber decodes from the camera, so it survives pause/seek/frame-rate conversion. Kept working but
secondary; see "Index coding" below.

### Phase 0 — eye anchor (interactive, ~2 min)

Web path: page shows white fullscreen with the nudge UI; you stand at the TV and press the page's
R/G/B ± buttons (or 1/2 3/4 5/6 keys) until the wall glow matches the TV white **by eye**, then
Accept. Video path: same, but white comes from the calib video and you nudge over SSH/Pi tty.
Output: anchor drive value, camera readings of both ROIs, von-Kries gains `g_c = tv_c / led_c`.
All subsequent camera readings of LED light are multiplied by `g`.

TV shows the white section of the calib video. LEDs driven white. The user nudges the LED drive
RGB from the keyboard until the wall glow looks like the same white as the TV picture, then
accepts. Output: anchor drive value, camera readings of both ROIs, von-Kries gains
`g_c = tv_c / led_c`. All subsequent camera readings of LED light are multiplied by `g`.

### Phase 1 — TV sweep (~15 min, unattended)

LEDs forced black. A grid of full-screen patches (default 6³ = 216) is shown on the TV.
Web path (`--calib-tv-web`, grid = `--calib-tv-grid`): the grabber commands each patch, waits for
the page ack, settles, measures — and records the exact commanded patch RGB alongside the camera
reading. Video path: the patches come from the calib video, each preceded by an index code so the
grabber knows which patch it is measuring — robust against pausing, seeking, TV frame-rate
conversion, and dropped camera frames.

Index coding (video path only; defined by `tools/gen-calib-video.py`, decoded by the grabber):

```
WHITE .4s | BLACK .3s | D1 .4s | BLACK .2s | D2 .4s | BLACK .2s | D3 .4s
| BLACK .3s | PATCH 1.5s | BLACK .3s | (next cycle)
```

Digits are full-screen colors, base 6: WHITE=0 RED=1 GREEN=2 BLUE=3 YELLOW=4 CYAN=5;
`index = D1·36 + D2·6 + D3`. Classification uses normalized channel thresholds, tolerant of an
uncalibrated camera. The patch itself is *not* classified (it can be any color including black):
it is measured in a fixed time window after D3, derived from the nominal timings. Per patch the
grabber records mean RGB, temporal std (across frames), spatial std (within ROI) — the variance
stats that decide whether 8-bit averaging suffices or RAW is needed.

### Phase 2 — LED sweep (~10–15 min, unattended)

TV shows black (web path blacks it out automatically; video path: pause on black). Baseline first:
LEDs black, measure both
ROIs (room ambient + TV black-level glow; subtracted later in the fit). Then for each drive
value in an 8³ grid + 17-step gray ramp: send solid frame, wait settle (HyperHDR smoothing + LED
response, default 700 ms), measure LED ROI (and TV ROI as a darkness sanity check), record same
stats as Phase 1.

### Phase 3 — fit + LUT build (offline; implemented 2026-06-12: `--ledcalib-fit`)

`hyperion-cam --ledcalib-fit --calib-out <dir> [--lut-out <file>] [--lut-size 33]` — no camera,
reads the sweep files, writes `lut.cube` + `fit_report.json`, exits. Math (`src/ledCalibFit.cpp`):

- **Model**: LED light adds linearly → `lin(reading) − lin(baseline) = A · [e_R(S_r), e_G(S_g),
  e_B(S_b)]` with a 3×3 mixing matrix `A` and monotone per-channel drive curves `e_c`.
- **Linearization**: camera response treated as a pure power law `x^p`; `p` is self-estimated
  from the sweep's own additivity (single-channel axis readings must sum to mixed readings in
  linear space) — no gamma assumption, no TV data needed.
- **Fit**: alternating least squares — `z = A⁻¹·Y` bucketed by drive level rebuilds the curves,
  then `A` refit by normal equations; 6 iterations, curves monotonized, `e(255)=1` normalized.
- **Anchor**: encoded von-Kries gains act as `g^p` in linear light. Gains were measured against
  *total* light (glow + ambient floor) and the viewer sees total light too, so the white pin and
  every inversion target include the ambient term: solve `corr·(A·e(S) + ambient) = lin(v)`.
  Targets below the ambient floor → LEDs off (the glow cannot be darker than the room).
- **Gamut (hue-true)**: negative `e*` components → blend toward the luminance-matched gray until
  feasible (desaturation, hue kept); overflow → scale the whole vector (chromaticity kept,
  luminance reduced); then invert the monotone curves into drive values.
- **Self-test**: `--ledcalib-fit-selftest` simulates a known chain (different per-channel LED
  exponents, crosstalk matrix, ambient, measurement noise, encode gamma 2.2), runs the whole
  pipeline, and closes the loop through the truth chain. Result 2026-06-12: exponent recovered
  exactly, closed-loop error median 0.10 / p95 0.45 / max 1.9 encoded units at 33³ (max 1.04 at
  65³), anchor round-trip error 0.05 drive units.

`.cube` contract: standard Resolve format, `LUT_3D_SIZE N`, domain [0,1], red index fastest;
values = LED drive / 255. Header comments carry fit stats. `fit_report.json` holds `p`, `A`,
curve tables, residuals, and gamut percentages — check it before trusting a real-data LUT
(`rel_resid_p95 > 0.10` prints a warning: noisy measurement or model mismatch).

**Open: the v1 fit does not yet consume `tv_sweep.csv`.** The LUT is built entirely from the LED
sweep + white anchor: the runtime LUT input is the camera's reading of the TV (`v`), and the fit
maps `v` → eye-corrected target glow → drive without needing the TV's own transfer. The TV sweep is
still recorded (now with exact commanded patch RGB in web mode) for two later uses: (a) gamut/
coverage — restrict or weight LUT accuracy to colors the TV actually produces; (b) a richer
metameric correction than the single white anchor (multiple TV/LED color correspondences → a 3×3
instead of a diagonal). Until then it is verification/diagnostic data, not a fit input.

## Gamma ramp probe (`--ledcalib gamma`, implemented 2026-06-15)

A standalone feasibility/diagnostic step, separate from the full calib: characterize the
**TV → camera transfer per channel** before trusting the pipeline. Shows R `(v,0,0)`, G, B and
gray `(v,v,v)` ramps (v = 0..255, `--gamma-step`, default every level → 4×256 patches), measures
the TV ROI, writes a CSV. Web-driven (requires `--calib-tv-web`); needs only `--calib-tv-roi`.
Locked exposure, LEDs off — this isolates what the camera sees from the TV. The curve is the
**combined TV∘camera** response (can't separate them from this test), which is exactly the LUT
input domain. Reading it: a flat low end = shadow crush, a flat high end = highlight clip,
per-channel divergence = the camera/TV channel imbalance the LUT must correct; recording all three
measured components also exposes cross-channel bleed.

**ISP vs RAW (two sequential runs, `--gamma-raw`).** Run 1 measures the ISP-RGB output (8-bit,
the LUT domain). Run 2 reconfigures the camera to the sensor's Bayer stream and measures the same
ramp by **Bayer-plane ROI averaging** (no demosaic): unpacks MIPI RAW10 (4 px / 5 bytes) and
averages the R / (Gr+Gb) / B sites. This sensor negotiates `SBGGR10_CSI2P` 1536×864 (BGGR);
`libcamera`'s `fourcc()` returns the base `'BG10'`-style code (packing is a modifier), so the
Bayer-phase table keys on those, defaulting to BGGR. The TV ROI (picked at the processed
resolution) is scaled into raw coords per-axis — fine because the patch is uniform; the run prints
the scaled raw ROI to sanity-check it lands on the TV. RAW is 10-bit (0..1023). Purpose: see
whether RAW preserves gradation at the ends where the ISP crushes/clips — the evidence for the
still-open "RAW for measurement precision" question. `--ledcalib-gamma-selftest` unit-tests the
RAW10 unpack + plane averaging on a synthetic frame (no camera).

Output: `gamma.csv` and/or `raw_gamma.csv` (`channel,level,meas RGB,temporal-std,spatial-std,
frames`). `tools/plot-gamma.py` (stdlib-only → SVG) overlays both runs, normalized, so crush/clip
and ISP-vs-RAW divergence are visible in a browser on the Pi.

### Phase 4 — verification (later milestone)

Closed-loop spot check on ~20 colors through the built LUT; report residuals in anchored camera
space.

## Outputs (`--calib-out` dir, default `ledcalib-out/`)

- `meta.json` — config echo, camera settings, timestamps
- `white_anchor.json` — drive RGB, both ROI stats, von-Kries gains
- `tv_sweep.csv` — `index, commanded patch RGB, TV-ROI stats, LED-ROI stats, frames`
- `led_sweep.csv` — `drive RGB, led-ROI stats, tv-ROI mean (sanity), frames`
- `baseline.json` — LEDs-off / TV-black readings
- `lut.cube`, `fit_report.json` — written by `--ledcalib-fit` (Phase 3)

## Runtime (later milestone)

The LUT is applied inside the packed NV12 warp fragment shaders: each pass samples both source
planes (same normalized coordinate addresses Y and UV textures), converts YUV→RGB, applies the
LUT, converts back. No LUT enabled → shaders identical to today. `--cpuonly`: no LUT in v1.

**Range contract (fixed 2026-06-12):** the ISP emits *limited-range* (SMPTE170M) NV12, while the
calibration measures in the ISP's full-range RGB-capture space — the LUT's input domain. The
shader conversion is therefore **limited-range BT.601**; a full-range matrix would compress the
runtime LUT inputs ~14% relative to the calibrated domain (systematic brightness/color error).
Residual mismatch between ISP RGB888 output and ISP NV12 + this matrix (rounding, 4:2:0
subsampling) is accepted.

**Interpolation decision (measured on RPi4, prod config 640×480, 2026-06-12 — overrides the
earlier tetrahedral preference):**

| Variant | GPU render | sensor→sent p50 |
|---|---|---|
| baseline (no LUT) | 1.95 ms | 16.5 ms |
| tetrahedral 17³ | ~9.5 ms | 24.5 ms |
| trilinear 17³ | 2.91 ms | 17.45 ms |
| trilinear 33³ | 3.00 ms | 17.55 ms |
| trilinear 64³ | ~3.0 ms | 17.5 ms |

Tetrahedral is **rejected** on v3d: 16 dependent `texelFetch`es + branching per packed Y fragment
→ +8 ms p50. Chosen: **HW trilinear with a 33³ (or up to 64³) LUT** — trilinear's interpolation
error shrinks ~quadratically with grid size, so a denser trilinear LUT replaces tetrahedral's
accuracy at ~1/7 the cost. Net feature cost ~+1 ms p50, accepted. Test hooks (identity LUT):
`HYPERCAM_LUT_TEST=1`, `HYPERCAM_LUT_TRILINEAR=1`, `HYPERCAM_LUT_SIZE=N`.

## Known caveats (documented, accepted)

- Per-pixel LUT sits before HyperHDR's region averaging; LUT(avg) ≠ avg(LUT) at hard edges —
  acceptable for glow.
- Saturated colors keep residual hue error after a white-only anchor (metamerism); optional
  extra anchors can reduce it.
- LED heat drift over long runs and OLED burn-in risk are why the run is 30–45 min, not hours.
- HyperHDR settings and camera color controls are baked into the LUT; changing either means
  recalibrating.
