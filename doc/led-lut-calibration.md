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
| Luminance policy | proportional, anchored at matched white, **soft knee above 85% of full LED output** so the brightest camera input reaches full output (2026-11) | eye adapts to dimmer glow; white anchor sets scale; the TV gets ~2-3× brighter from mid-gray to white than the LEDs can, so a hard ceiling flattened every bright tone |
| Gamut overflow | clip hue-true (preserve hue, reduce sat/lum) | artifact-free |
| Measurement bit depth | TV side: ISP RGB (the LUT input domain). LED sweep, TV primaries and the response sweep: **RAW + ISP in the same request** (2026-11) | the LUT input must be runtime grabber space; the LED glow is read in the ISP's clip/shoulder (red at 246 of 255), where RAW (the clip is the ISP's WB gain, not the sensor) has ~3× headroom and is linear |
| Camera model | **measured tone curve** (`--ledcalib response`), legacy self-estimated power law kept as fallback (2026-11) | the power law was the gray-drift cause — see "Camera model fix" |
| ISP settings | **frozen tuning file** (`--make-tuning`): no adaptive contrast, no adaptive lens shading, power-2.2 tone curve, TV-fitted colour matrix (2026-11) | the stock ISP is scene-adaptive and clips saturated TV colours to 0 — see "Camera model fix" |
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
default 100 ms, valid because smoothing is off) **plus a minimum of 3 fresh camera frames since
the drive change** (same guard as the web ack path — at low camera fps, in-flight frames showing
the previous drive outlive the settle time), measure LED ROI (and TV ROI as a darkness sanity
check), record same stats as Phase 1.

### Phase 3 — fit + LUT build (offline; implemented 2026-06-12: `--ledcalib-fit`)

`hyperion-cam --ledcalib-fit --calib-out <dir> [--lut-out <file>] [--lut-size 33]` — no camera,
reads the sweep files, writes `lut.cube` + `fit_report.json`, exits. Math (`src/ledCalibFit.cpp`):

- **Model**: LED light adds linearly → `lin(reading) − lin(baseline) = A · [e_R(S_r), e_G(S_g),
  e_B(S_b)]` with a 3×3 mixing matrix `A` and monotone per-channel drive curves `e_c`.
- **Linearization** (`--calib-response auto|on|off`): the camera tone curve measured by the
  response phase (`camera_response.csv`, see "Camera model fix"), applied to every camera reading
  (LUT input, white anchor). Without it (old calibration dirs, or `off` for A/B) the legacy model:
  a pure power law `x^p` with `p` self-estimated from the sweep's additivity — which the
  near-diagonal mixing matrix barely constrains, so it is only right for a camera that really is a
  power law.
- **LEDs on RAW** (automatic with the measured curve + RAW columns in `led_sweep.csv`): LED light
  = RAW means at the LED ROI mapped to ISP-linear units by a 3×3 fitted on the sweep's own clean
  samples (`lin(isp_r) = Σ M[r][c]·raw_c`, ISP reading 12..245 and unclipped). The map absorbs the
  colour matrix, WB, lens shading at the LED ROI and the ISP's black/scale handling. Only a
  saturated *sensor* excludes a sample; `raw_map_resid` in the report is the RAW-vs-ISP agreement.
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
  luminance reduced) for saturated colours, while near-neutral colours (<10% saturation relative to
  the eye-matched white, blended up to 30%) keep their luminance and give up a little saturation
  toward the eye-matched white instead. With the plain scale-down, a slightly tinted node next to
  white was dimmed as a whole and trilinear interpolation dipped the gray axis' brightness ~4.5%
  near white; now ~1.6%. Then invert the monotone curves into drive values.
- **Soft knee** (`--calib-knee auto|off|<start %>`, default 85%): the required brightness,
  expressed as the max-drive fraction white would need at that luminance (`m*`), maps through a
  C1 roll-off (identity up to the start, `1-exp` above, slope 1 at the start) that reaches full
  output exactly at camera white (LUT input 255,255,255). All channels are scaled by one factor, so
  hue is kept; driving it by luminance rather than by the brightest channel keeps it smooth across
  hue (no interpolation dips). `fit_report.json` `knee.white_needs` = how many times the LEDs'
  full output camera white would need (the real rig: ~2-3×).
- **Dark fallback** (`--calib-dark-floor auto|off|<luma>`): below camera luma = TV black + 8 codes
  (one LUT grid step) a dark colour spans only a few camera codes and its chroma is unresolvable;
  the LUT fades (smoothstep from half the floor) to **equal R=G=B drive at the same luminance** —
  HyperHDR's eye-matched white by construction. (The ROI's spatial std is not usable as a noise
  measure: on the real rig it is ~20% of the mean at every level — panel structure, not noise.)
- **Self-test**: `--ledcalib-fit-selftest` simulates a known chain (different per-channel LED
  exponents, crosstalk matrix, ambient, RAW sensor matrix, measurement noise) whose camera encodes
  with the **stock imx708 tone curve** (not a power law), synthesizes all four input files incl. a
  response sweep, runs the whole pipeline and closes the loop through the truth chain. Result
  2026-11-20: response recovered within 1.6% (codes 16..250); measured model closed loop p95 0.96
  / max 1.6 encoded units, anchor round-trip 0.1 drive units; **legacy power-law model on the same
  data p95 30 / max 43** (the rig's drift, reproduced — the test checks the legacy model is worse,
  so it really exercises the failure); knee reaches full output at white with a monotone gray
  axis; dark nodes neutral.

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
sweep white points — `cam_tv_white / cam_led_white` at the **white-anchor neutral level**: by
default the `--ledcalib-white-level` the sweeps ran with (recorded in `meta.json`, default mid-gray
**127**), i.e. *the level where you eye-matched WB* — not the brightest clean level, because the
metameric gains are one per-channel scalar applied across the whole LUT and are most trustworthy where
the eye-match was done, and 127 also stays clear of camera clipping. `--calib-white-anchor-level N`
overrides at fit time. The sweeps inject that exact neutral into both tv and led plans so the TV side
is a real measured patch (the TV reading is not interpolated); the LED side is interpolated to the
level if needed. If the target level clips the camera (`--calib-clip-level`, default 250) the fit
falls back to the nearest clean neutral. The eye-truth comes from
**HyperHDR's white balance** (set it to match the LED white to the TV white before measuring —
e.g. to pull down bluish LEDs). The TV and LED can't
share a frame anyway (cropped TV vs full-frame wall). **Timing rule:** HyperHDR WB (and all HyperHDR
colour settings) are frozen into the LED sweep, so set white *before* measuring; a big WB change
*after* invalidates the LUT (re-measure). `--ledcalib white` is a **purely manual** aid for setting
that HyperHDR white balance: it drives the LEDs full white and sends a white frame to HyperHDR so you
can eye-match in HyperHDR's settings. It takes no ROI, measures nothing, and writes no file — the
white correction always comes from the sweeps, never from this phase.

## Camera model fix (2026-11): why the white drifted with level

**Symptom (real rig, with `--lut`):** grays matched the 127 eye-match but drifted toward the ends
— brights pink/magenta, mid-darks (64-114) greenish, ~42 yellow-green (predicted from the data,
confirmed on films).

**Diagnosis (from the July sweeps + the stock `imx708.json`):**
- The fit linearized every camera reading with one power law (`p=2.21`), but the ISP tone curve's
  local exponent runs **1.6 (shadows) → 4.2 (highlights)**. TV and LEDs are read at different points
  of it: TV gray 127 reads ~130, LED drive 127 already reads R 204 (the shoulder). A per-channel
  ratio that is right at the anchor is wrong everywhere else — exactly the symptom. Re-linearizing
  the sweep neutrals with the true curve flattens the TV/LED ratio above ~114.
- `p` was self-estimated from LED additivity, but the mixing matrix is nearly diagonal (each camera
  channel sees mostly one LED), so additivity hardly constrains the curve shape.
- Two ISP stages adapt to the scene: contrast enhancement (`ce_enable: 1`) and adaptive lens/colour
  shading (ALSC `n_iter: 100`) — calibration scenes (TV lit / wall dark, TV black / wall lit) and
  films get different camera responses.
- The stock colour matrix (+ `--saturation 0.73`) has negative coefficients: **867 of 4097** bright
  TV patches read 0 in some channel (TV blue reads R 0 / G 0.1) — those colours collapse and no LUT
  can separate them again.
- The shading map multiplies encoded values, which is exact at every level only for a pure power
  curve; with the stock curve a 0.95 gain is −7% light at level 40 but −15% at 220.
- Not the cause: HyperHDR is linear (gamma 1, temperature R 1 / G 0.38 / B 0.22) and the TV's gray
  is stable within ±5% as the camera sees it.

**Decisions** (so the camera becomes a known, scene-independent comparator):
1. **Frozen tuning file** (`--make-tuning OUT [--from-calib DIR]`, `--tuning FILE`; `src/tuning.cpp`,
   Boost.JSON): `ce_enable 0`, ALSC `n_iter 0`, tone curve = power 2.2 on the vc4 hardware knots
   (`--tuning-curve stock|<exponent>`). A power curve (not the stock one) because it gives the darks
   ~2× the codes (the hardware's first 1/64 segment is linear: slope 9.7 vs 4.9), makes the shading
   map exact at every level, and spreads the LUT's grid evenly. Generated once and kept with the
   calibration, so a libcamera update can't silently change it. The non-LUT look is not a goal.
2. **TV-fitted colour matrix** (`--from-calib`): from `--ledcalib tv-raw` (TV patches on RAW), one
   matrix with the TV's primaries on the axes and TV white neutral (WOLED white subpixel handled by
   the white normalisation), a small safety desaturation (≥5%, raised until every measured bright
   patch stays ≥0.5% of its brightest channel), the per-raw-channel ISP gain at the TV ROI (lens
   shading, black/scale) measured from the probe's own ISP/RAW pairs, and scaled so the **10%
   peak-white window lands at 95%** of the ISP range (OLED highlights outshine any full-field patch).
   `-b` stays as measured, `--saturation 1` (the matrix contains everything). Verified: libcamera
   applies the single-entry matrix as written (metadata); on a synthetic WOLED-like TV the fit
   recovers the shading gains exactly.
3. **Measured tone curve** (`--ledcalib response`): shutter sweep 1/64×..4× (quarter stops, a
   reference step every 6) over a fixed LED white (+ optional TV gray), RAW captured in the same
   requests. Light per sample = the RAW sum (linear, drift-proof); the curve is a Debevec-style
   least-squares fit of `ln f⁻¹(z)` with per-series scales and a smoothness prior relative to a
   2.2 power (λ=10: follows the hardware knots' corners; 40 rounds them off >4%, 4 fits noise). A
   frozen ISP gives one consistent curve; a residual >4% in any series is flagged right after the
   sweep (adaptive ISP / moving scene) — before the long sweeps.
4. **LEDs on RAW** (dual stream in the LED sweep), the RAW→ISP-linear map fitted by `--ledcalib-fit`.
5. **Dual-stream infrastructure**: the RAW size is chosen by probing which one reproduces the sensor
   mode an ISP-only (production) configuration selects (ScalerCropMaximum match — on this camera the
   1536×864 mode, sensor area (768,432)/3072×1728); ROIs map ISP→RAW via ScalerCrop metadata + mode
   crop + binning (verified: ISP vs mapped-RAW correlation peaks at zero offset / scale 1.00);
   black levels from `SensorBlackLevels`.
6. **Soft knee + dark fallback** in the fit (Phase 3).
7. **Eye check** (`--eyecheck-web`): production pipeline + a client-side gray-slider/colour-picker
   page — the acceptance test. Multi-point eye calibration of the white curve stays the fallback,
   only if the eye check still fails.

**A/B from one measurement session:** fit A `--calib-response off --calib-knee off
--calib-dark-floor off` (the frozen tuning alone, legacy model), B `--calib-knee off
--calib-dark-floor off` (+ the camera model fix), C defaults (+ knee + dark fallback).

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
resolution) is mapped into raw coords via the ScalerCrop metadata, the sensor mode's crop and the
binning (fixed 2026-11: it used to be scaled proportionally, which is wrong with `--crop` because
the RAW frame is the whole sensor mode); the run prints the raw ROI. RAW is 10-bit, black-subtracted. Purpose: see
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

- `meta.json` — LED-sweep config echo (white level etc.; read by the fit)
- `meta_<phase>.json` — every phase: ROIs, camera args, tuning file + hash, frame metadata
  (exposure, gains, CCM); the fit cross-checks them
- `tv_raw.csv` — TV RAW probe: patch, window %, RAW means/sat (black-subtracted), ISP means/sat
- `tuning.json` (suggested name) — the frozen tuning file written by `--make-tuning`
- `camera_response.csv` — response shutter sweep: per step exposure, ISP + RAW per ROI
- `tv_sweep.csv` — `index, commanded patch RGB, TV-ROI stats, frames`
- `led_sweep.csv` — `drive RGB, led-ROI ISP stats, frames` (+ ISP sat, RAW means/stds/sat,
  exposure when captured dual-stream)
- `baseline.json` — LEDs-off / TV-black readings (+ RAW)
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
- **RAW for the LED dark end — resolved 2026-11.** The LED sweep is measured on RAW (dual stream)
  and mapped into the ISP-linear domain by the fit (see Phase 3), which also removes the bright
  end's ISP clip/shoulder. `--calib-curve-floor` regularization still applies below 5% output.
- **Trilinear near white.** Interpolating between gray nodes and tinted neighbours that the gamut
  mapping had to adjust still dips the gray axis' brightness by up to ~1.6% between nodes near
  white (was ~4.5% before the near-neutral overflow change). Below visibility; a denser LUT would
  shrink it further.
- **One tuning file everywhere.** The frozen tuning file, `-b` and `--saturation 1` must be
  identical in every phase and in production; the fit compares the `meta_<phase>.json` records and
  warns on a mismatch.
- LED heat drift over long runs and OLED burn-in risk are why the run is 30–45 min, not hours.
- HyperHDR settings and camera color controls are baked into the LUT; changing either means
  recalibrating.
