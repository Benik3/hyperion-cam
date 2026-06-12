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
| Interpolation | tetrahedral (trilinear toggle) | user preference; standard for color LUTs |
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

### Phase 0 — eye anchor (interactive, ~2 min)

TV shows the white section of the calib video. LEDs driven white. The user nudges the LED drive
RGB from the keyboard until the wall glow looks like the same white as the TV picture, then
accepts. Output: anchor drive value, camera readings of both ROIs, von-Kries gains
`g_c = tv_c / led_c`. All subsequent camera readings of LED light are multiplied by `g`.

### Phase 1 — TV sweep (~15 min, unattended)

LEDs forced black. The calib video shows a sequence of full-screen patches (default 6³ grid =
216), each preceded by an index code so the grabber always knows which patch it is measuring —
robust against pausing, seeking, TV frame-rate conversion, and dropped camera frames.

Index coding (defined by `tools/gen-calib-video.py`, decoded by the grabber):

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

TV shows the black section (or is paused on black). Baseline first: LEDs black, measure both
ROIs (room ambient + TV black-level glow; subtracted later in the fit). Then for each drive
value in an 8³ grid + 17-step gray ramp: send solid frame, wait settle (HyperHDR smoothing + LED
response, default 700 ms), measure LED ROI (and TV ROI as a darkness sanity check), record same
stats as Phase 1.

### Phase 3 — fit + LUT build (offline, later milestone)

Fit smooth models to both sweeps; for each LUT node (input = runtime grabber value) compute the
eye-corrected target glow (luminance proportional, anchored at white; hue-true clip when out of
LED gamut) and invert the LED model → drive value. Output: LUT file for `--lut`.

### Phase 4 — verification (later milestone)

Closed-loop spot check on ~20 colors through the built LUT; report residuals in anchored camera
space.

## Outputs (`--calib-out` dir, default `ledcalib-out/`)

- `meta.json` — config echo, camera settings, timestamps
- `white_anchor.json` — drive RGB, both ROI stats, von-Kries gains
- `tv_sweep.csv` — `index, mean RGB, temporal-std RGB, spatial-std RGB, frames`
- `led_sweep.csv` — `drive RGB, led-ROI stats, tv-ROI mean (sanity), frames`
- `baseline.json` — LEDs-off / TV-black readings

## Runtime (later milestone)

`--lut <file>` loads a 17³ RGB→RGB LUT into a 3D texture; the NV12 warp fragment shader converts
YUV→RGB, applies tetrahedral interpolation, converts back. Watch the packed Y pass: 4 output
pixels per fragment → 4 LUT applications per fragment (~16–20 texture fetches with tetrahedral).
Measure with `HYPERCAM_GL_PROFILE=1` + `--latency-stats` before accepting; fallbacks if too slow:
trilinear (`HYPERCAM_LUT_TRILINEAR=1`), smaller LUT, or moving the LUT to a HyperHDR-side hook.
No `--lut` → behavior identical to today. `--cpuonly` path: no LUT support in v1.

## Known caveats (documented, accepted)

- Per-pixel LUT sits before HyperHDR's region averaging; LUT(avg) ≠ avg(LUT) at hard edges —
  acceptable for glow.
- Saturated colors keep residual hue error after a white-only anchor (metamerism); optional
  extra anchors can reduce it.
- LED heat drift over long runs and OLED burn-in risk are why the run is 30–45 min, not hours.
- HyperHDR settings and camera color controls are baked into the LUT; changing either means
  recalibrating.
