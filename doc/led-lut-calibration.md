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
| TV patch display | web page on a PC on the TV's HDMI (`--calib-tv-web`) | grabber commands each patch directly, page acks after paint; no decoder/video |
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

**Picking the TV ROI against the patch window:** the page's start screen (before you tap *begin*)
draws an **example patch** — a dim-gray rectangle with a red outline — at the exact configured
size/position. Run the **dry run** (`--ledcalib-web-dryrun`, camera-free) to show it on the TV,
then capture the camera's view with `hyperion-cam --imgout still.jpg` in another terminal (the dry
run doesn't hold the camera) and pick `--calib-tv-roi` inside the red rectangle.

The user provides two axis-aligned ROIs in raw full-frame coordinates (pick them from an
`--imgout` still taken without `--crop`):

- `--calib-tv-roi x,y,w,h` — entirely inside the TV picture area
- `--calib-led-roi x,y,w,h` — on the wall where LED glow is visible (repositioning the camera
  for the calibration run is allowed; reposition as little as possible — the TV shifts color
  off-axis, which skews the TV-side readings)

Requirements during the run: dark room, HyperHDR running with the grabber as the only active
source, HyperHDR color/WB settings in their final state (the LUT bakes them in — change them later
and the LUT is stale), and HyperHDR **LED smoothing disabled** so the LEDs jump to each commanded
colour instantly (the short settle would otherwise measure mid-ramp; smoothing doesn't change the
steady-state colour, so it's off only for the run and re-enabled for viewing).

### Getting patches onto the TV: the web page (`--calib-tv-web`)

The grabber runs a tiny HTTP server (evhttp, on `--calib-tv-port`, default 8080) and you open
`http://<pi>:8080/` in a **fullscreen** browser on a PC plugged into the TV. The grabber pushes the
fullscreen color over SSE; the page paints it, double-`requestAnimationFrame`s to guarantee the
frame is up, then `POST /ack?n=seq`. The grabber starts its settle timer **on the ack**, so timing
covers only TV-internal latency, not network/paint lag — and a closed/frozen page is caught by an
8 s ack timeout (clean abort). After the ack it waits both the settle time **and** a minimum of 3
fresh camera frames before measuring, so the stale in-flight frames are flushed even if the camera
fps is low. The grabber *commands* each patch directly (no decoding). The page
also hosts the white-phase nudge buttons so the whole run is driven from the TV-side PC.
`src/ledCalibWeb.cpp`. Required for `tv`/`gamma`/`shading`. (A calib-video + index-decoder fallback
existed early on and was removed 2026-06-16 — web only.)

**Window patches (`--patch-size`, `--patch-x`, `--patch-y`).** TVs (especially OLED) dim bright
full-screen fields via ABL/ASBL, distorting the measurement. `--patch-size` shrinks the patch to
a percentage of screen **area** (calibration convention; e.g. `10` or `3`) on a black surround;
**default `10`** (set `--patch-size 100` for a fullscreen field). `--patch-x`/`--patch-y` place the patch top-left
in screen-percent, default centered (`<0`). The box preserves the screen aspect ratio. When a
window is used you must pick `--calib-tv-roi` **inside** the patch — view it via the dry run +
`--imgout`. The page's status label (bottom-left) shows `seq: R, G, B` in dim gray to limit OLED
retention of the static text.

**Dry run:** `--ledcalib-web-dryrun --calib-tv-port 8080 [--patch-size 10 --patch-x .. --patch-y ..]`
starts only the page (no camera, no HyperHDR, no service stop needed) and cycles demo patches —
WHITE with the white-phase nudge UI, then R/G/B/gray/white/black. Open it fullscreen on the TV PC to
check the patch size/position, colors, the nudge buttons (they move the numbers), and the SSE/ack
connection before committing to a real run.

**Hard requirement:** plug the calib PC into the **same HDMI input / picture mode** you normally
watch through. TVs apply different gamma/chroma/processing per input ("PC mode", "Game mode",
4:4:4 vs 4:2:0); the camera-as-comparator only holds if the calib display path gets the same TV
processing as runtime content. Browser color management is *not* the main risk (we measure the
camera's reading of whatever the TV actually shows, not the commanded value) — the per-input TV
processing difference is.

### White — manual white balance (`--ledcalib white`, interactive, ~2 min)

A **purely manual** aid for setting HyperHDR's white balance; it makes no measurement and writes no
file. After you tap *begin* (web path, `--calib-tv-web`) — or press Enter on a tty — the page shows a
**full-screen white** field and the grabber sends a solid white frame to HyperHDR, so the LEDs go
white. Make the glow match the TV white **either** by tuning **HyperHDR's white balance**
(recommended if your LEDs are off-white — e.g. bluish) **or** by nudging the LED drive with the
page's R/G/B ± buttons (or 1/2 3/4 5/6 keys) to preview. Enter/**Accept** just exits. The actual
white correction is derived later, from the sweep white points ("White from the sweeps" below) — this
phase only helps you dial in the HyperHDR WB that those sweeps then freeze in. Takes no ROI.

### Phase 1 — TV sweep (~15 min, unattended)

LEDs forced black. A grid of patches (default 6³ = 216, grid = `--calib-tv-grid`, edge-dense
levels) is shown on the TV via the web page: the grabber commands each patch, waits for the page
ack, settles, measures — and records the exact commanded patch RGB alongside the camera reading
(mean, temporal std, spatial std — the variance stats that decide whether 8-bit averaging suffices
or RAW is needed).

### Phase 2 — LED sweep (~10–15 min, unattended)

TV shows black (the page blacks it out automatically). Baseline first:
LEDs black, measure both
ROIs (room ambient + TV black-level glow; subtracted later in the fit). Then for each drive
value in an 8³ grid + 17-step gray ramp: send solid frame, wait settle (LED + panel response;
default 100 ms, valid because smoothing is off), measure LED ROI (and TV ROI as a darkness sanity
check), record same stats as Phase 1.

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
- **Dark-end curve regularization** (`--calib-curve-floor`, default 0.05): the camera can't
  measure dim LED glow at the locked exposure — a heavily white-balance-attenuated channel (e.g.
  HyperHDR green at 0.25× floors out first), so the bottom of each measured curve is noise that
  `monotonize()` flattens into a plateau. Inverting a plateau makes that channel switch on abruptly
  at a drive level unrelated to its neighbours → dark-grey **crush** (everything maps to ~0 drive)
  + **colour cast** (one channel — typically green — slams on first). Fix: per channel, replace
  every curve point below `floor × full-output` with a smooth power law `e = e(s*)·(s/s*)^γ`
  anchored at the first reliable level `s*`, with `γ` fit log-log from the low reliable region
  (≥3 points, ≤½ output; falls back to γ=1, "LED ~ linear"). The bottom then rises monotonically
  from 0 with no plateau and all channels turn on consistently. `--calib-curve-floor 0` disables
  (raw measured bottom). This only reshapes the unreliable bottom — `e(255)` and the mid/top are
  untouched. It does **not** fix the underlying measurement limit (see "RAW for the LED dark end"
  in caveats); for that the LED sweep would have to be measured on the 10-bit RAW stream.
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

**The fit consumes `tv_sweep.csv` only for the white point (so far).** The LUT is built from the
LED sweep + the von-Kries white correction: the runtime LUT input is the camera's reading of the
TV (`v`), and the fit maps `v` → eye-corrected target glow → drive without needing the TV's full
transfer. The white correction is **derived from the sweep white points** — `cam_tv_white` (the
white TV patch) and `cam_led_white` (LED at full drive), see "White from the sweeps" below. The rest of the TV sweep is
still unused, kept for two later uses: (a) gamut/coverage weighting; (b) a richer metameric
correction than the diagonal white (multiple TV/LED correspondences → a 3×3).

### White from the sweeps (the standard workflow)

There is **no `all` phase**: run `--ledcalib tv` (cropped+shaded) and `--ledcalib led` (full frame)
separately into the same `--calib-out`. `--ledcalib-fit` **derives** the von-Kries white from the
sweep white points — `cam_tv_white / cam_led_white` at the
**brightest neutral level where neither side clips the camera** (`--calib-clip-level`, default 250;
the camera maxes below 255 and the TV or LEDs may be the brighter side, so true-255 white often
saturates → a dimmer clean neutral is used, with the LED side interpolated to that level). The eye-truth comes from
**HyperHDR's white balance** (set it to match the LED white to the TV white before measuring —
e.g. to pull down bluish LEDs). The TV and LED can't
share a frame anyway (cropped TV vs full-frame wall). **Timing rule:** HyperHDR WB (and all HyperHDR
colour settings) are frozen into the LED sweep, so set white *before* measuring; a big WB change
*after* invalidates the LUT (re-measure). `--ledcalib white` is a **purely manual** aid for setting
that HyperHDR white balance: it drives the LEDs full white and sends a white frame to HyperHDR so you
can eye-match in HyperHDR's settings. It takes no ROI, measures nothing, and writes no file — the
white correction always comes from the sweeps, never from this phase.

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

## Shading / ALSC correction (`--ledcalib shading` + `--shading-map`, 2026-06-16)

Position-dependent error (lens vignetting + TV-panel non-uniformity) that a single global LUT
**cannot** fix — it miscolors edge-mapped LEDs. This is the *TV-side* calibration: it runs through
the production `--crop` + `--transform`, unlike the LED sweep (which needs the full frame).

- **Measure** `--ledcalib shading`: full-screen safe gray (`--shading-level`, default 180 — never
  white, to dodge clipping/ABL on any TV), samples a `--shading-grid` (UxV, default 24×14) across
  the TV quad (each cell mapped through the inverse homography of `--transform`, averaged), writes
  `shading.csv` (per-cell RGB + gains) and prints the magnitude (luminance falloff + center→edge
  R/G/B-ratio drift — the latter decides if it's worth applying). Bakes in *both* lens and panel
  non-uniformity, which is wanted (correct THIS TV; especially LCDs).
- **Apply** `--shading-map shading.csv`: bilinear RGBA8 gain texture, multiplied per output
  position in the NV12 warp shader (YUV→RGB→×gain→YUV, limited-range BT.601). Gains are
  **darken-to-dimmest** (≤1), so the correction can never push a region into clipping. Cost ~+0.7
  ms render / +0.5 ms p50 (folds into the LUT's RGB round-trip if both are on); zero when not
  passed. NV12 GPU path only.
- The wall/LED-ROI is left uncorrected on purpose: its (constant) shading cancels between the white
  anchor and the LED sweep, so it doesn't affect the LED calibration.

**TV-sweep / gamma consistency with shaded production.** The TV-side measurements (`--ledcalib
tv`, `gamma`) keep the production `--crop`, and when `--shading-map` (+ `--transform`) is given
they scale the TV-ROI reading by the shading gain at that ROI's output position (mapped through the
homography, bilinear-sampled — the same gain the shader applies there). This makes the LUT input
domain match the shaded production output **without** running the GPU warp in the calib path:
crop+warp are geometric and don't change a uniform patch's colour (locked AE/WB), so only the
shading gain has to be reproduced. It also resolves the "wrong part of the TV" worry — after
shading every position reads the same flattened value, so the global LUT is valid everywhere and
the exact ROI choice stops mattering. The LED wall ROI is left unshaded (see above); `all`/`led`
ignore `--shading-map` (they need the full frame and can't keep the TV crop) — run `tv` and `led`
as separate phases and derive white from the sweeps (the standard workflow).
- **Verify on the real TV** (can't be checked off-Pi): show uniform gray with the map applied — it
  should get *flatter*, not worse. If a corner gets worse it's an axis flip in the gain texture
  upload, a one-line fix.

### Phase 4 — verification (later milestone)

Closed-loop spot check on ~20 colors through the built LUT; report residuals in anchored camera
space.

## Outputs (`--calib-out` dir, default `ledcalib-out/`)

- `meta.json` — config echo, camera settings, timestamps
- `tv_sweep.csv` — `index, commanded patch RGB, TV-ROI stats, LED-ROI stats, frames`
- `led_sweep.csv` — `drive RGB, led-ROI stats, tv-ROI mean (sanity), frames`
- `baseline.json` — LEDs-off / TV-black readings
- `lut.cube`, `fit_report.json` — written by `--ledcalib-fit` (Phase 3)

## Runtime (later milestone)

**Implemented 2026-06-16: `--lut <file.cube>`.** Loads the `.cube` from `--ledcalib-fit` as a
trilinear RGBA8 3D texture and applies it inside the packed NV12 warp shaders: each pass samples
both source planes, converts YUV→RGB (limited-range BT.601), **× shading gain (if `--shading-map`)
then the LUT**, converts back. One unified colour-correction shader composes shading + LUT (sharing
the single round trip); either may be absent; no flags → shaders identical to today. `--cpuonly`:
no LUT. Measured cost on the prod config: LUT 1.95→2.88 ms render (p50 16.5→17.3); LUT+shading
3.08 ms / 17.5 — composing adds only ~0.2 ms over LUT alone. Texture units: Y/UV=0/1, LUT=2,
shade=3. The env identity-LUT test (`HYPERCAM_LUT_TEST`) still works for profiling when no real
`--lut` is given.

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
- **RAW for the LED dark end (open).** The locked exposure that keeps the bright LED glow from
  clipping leaves the dim low-drive glow below the 8-bit ISP noise floor, worst on a
  white-balance-attenuated channel — so the dark end of the drive curves is unmeasurable and is
  reconstructed by `--calib-curve-floor` regularization, not measured. Measuring the LED sweep on
  the 10-bit RAW Bayer stream (the machinery exists for `--gamma-raw`) would recover real low-end
  data; the cost is mapping RAW back into the ISP-RGB LUT-input domain. Note the *gamma probe*
  (`--ledcalib gamma`) characterizes the TV→camera transfer (the LUT **input**), not these
  LED→camera curves (the **output** mapping), so it does not address this directly.
- LED heat drift over long runs and OLED burn-in risk are why the run is 30–45 min, not hours.
- HyperHDR settings and camera color controls are baked into the LUT; changing either means
  recalibrating.
