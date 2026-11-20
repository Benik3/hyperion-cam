# LED color calibration — user guide

End-to-end: TV RAW probe → frozen tuning file → shading → camera response → TV sweep → LED sweep →
fit → eye check → apply. Design/maths live in `led-lut-calibration.md` (see "Camera model fix" for
why the camera is set up this way); this is the how-to with commands.

> Usage doc kept here for convenience; the canonical home is the GitHub Wiki.

## What you produce
- `tuning.json` — a **frozen libcamera tuning file**: no scene-adaptive processing, a known tone
  curve, a colour matrix fitted to your TV. Used by **every** later step and by production.
- `shading.csv` — TV/lens uniformity correction (optional; only if your TV drifts).
- `lut.cube` — 3D color LUT making the LED glow match the TV.

All are applied as **grabber flags** (edit the `hypercam` service `ExecStart`), not via a web UI.
The web page shows calibration patches on the TV (and, for the final check, a slider page).

## The web interface
A PC on the **same TV HDMI input/picture-mode you normally watch** runs a fullscreen browser at
`http://<pi-ip>:8080/`. The grabber drives the patches; the page acks each paint. Use a PC browser
(Firefox/Chrome) — a TV's built-in browser may not stay awake.

## Ground rules (read once)
- **Dark room.** The page keeps the display awake. If your TV still blanks on a static image,
  disable its screensaver. Also disable **eco/ambient auto-dimming** (corrupts sweeps).
- HyperHDR running, grabber the only source, **HyperHDR color + white-balance settings final** —
  they bake into the LUT; change later → recalibrate.
- **Disable HyperHDR LED smoothing for the calibration run** (the LEDs must jump instantly to each
  commanded colour). It doesn't affect the LUT — re-enable it for normal viewing afterwards.
- Stop the service while calibrating, restart after:
  ```bash
  sudo systemctl stop hypercam      # before each run
  sudo systemctl start hypercam     # when done
  ```
- **Use a NEW output dir** and keep the old one as the "before" for comparison. The examples use
  these shell variables — substitute your geometry / camera settings:
  ```bash
  CAL=~/calib3
  GEO="-w 768 -h 432 --crop 1686,1014,1173,751 -t 0,22:765,0:0,403:768,432"   # production geometry
  CAM="-b 2.632,1.626 -s 13600 --focus 0.93"                                   # production camera args
  T="--tuning $CAL/tuning.json --saturation 1"                                 # from Step 2 on
  ```
- **The same `$CAM` and (from Step 2) `$T` in every step and in production.** The tuning file's
  colour matrix already contains the saturation — `--saturation` must be **1** with it. The fit
  compares what each phase recorded (`meta_<phase>.json`) and warns on a mismatch.

---

## Step 0 — Prep

### 0a. White balance (first; don't change after)
Set white in **HyperHDR's white balance** so the LED glow matches the TV white by eye (this is
where you fix bluish LEDs). `--ledcalib white` is an optional aid — it drives the LEDs white and
shows a matching full-screen white reference on the TV; `--ledcalib-white-level N` (1-255,
**default 127** = mid-gray) sets the level for both. This same level is the fit's white anchor —
eye-match WB at it. It takes no ROI, measures nothing, and writes no file (Enter/Accept exits).
This eye-match does not involve the camera, so it stays valid across the tuning change.

### 0b. ROIs and aim
The page shows a 10%-of-screen **window patch** (dodges TV auto-dimming). Find where it lands and
pick the TV ROI inside it:

```bash
# Terminal A — show the example patch (camera-free); DON'T tap begin:
./Release/hyperion-cam --ledcalib-web-dryrun --calib-tv-port 8080 --patch-size 10
#   open http://<pi>:8080/ fullscreen on the TV PC -> red rectangle marks the patch

# Terminal B — capture the camera's cropped view:
sudo systemctl stop hypercam
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --imgout /tmp/aim.jpg $GEO
```
Open `/tmp/aim.jpg`: read `--calib-tv-roi x,y,w,h` **inside the red rectangle**, near the TV
centre (off the red border). For the **LED ROI**, capture a full-frame still (`-w 1024 -h 576`, no
`--crop`) with the LEDs lit and read `--calib-led-roi x,y,w,h` on a clearly-lit patch of wall.

---

## Step 1 — TV RAW probe (input of the TV colour matrix)
Measures the TV's own colours on the camera's **RAW** frame (before any colour processing).

```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain $GEO $CAM --calib-tv-web \
  --calib-tv-roi 278,165,214,104 --calib-out $CAL --ledcalib tv-raw
```
Open page → **tap begin**. ~230 patches (colour cube, gray ramp, and a final 10% peak-white
window), ~2 min → `$CAL/tv_raw.csv` + `meta_tv-raw.json`. No `--tuning` needed here (RAW doesn't
depend on it). If it warns that the **peak-white window clips the SENSOR**, lower `--gain` (add it
to `$CAM`) and rerun — the shutter stays fixed for latency.

## Step 2 — Frozen tuning file (offline)
```bash
./Release/hyperion-cam --make-tuning $CAL/tuning.json --from-calib $CAL
```
Starts from the system tuning file for your sensor and writes `$CAL/tuning.json`: adaptive contrast
and adaptive lens shading off, a power-2.2 tone curve, one colour matrix fitted to your TV (scaled so
the peak-white window lands at 95%). It reports how many TV colours the stock matrix pushed to zero
and prints the args to use — **from now on add `$T` to every command** (and keep `-b` as printed).

> Without `--from-calib` it freezes the stock tuning (keeps the stock colour matrix).
> `--tuning-curve stock` keeps the stock tone curve. `--tuning-base FILE` picks another base.

---

## Step 3 — Shading calibration (optional)
Maps lens + TV-panel non-uniformity so edge LEDs aren't miscolored. Uses production crop+transform;
a safe gray (never white).

```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --ledcalib shading --calib-tv-web \
  $GEO $CAM $T --calib-out $CAL --shading-grid 24x14
```
Open page → **tap begin**. Fills the TV with gray 180, samples a 24×14 grid, writes
`$CAL/shading.csv`, prints `max colour (R/G,B/G) drift center->edge: X%`.

> `--crop`/`--transform` are **optional**. They're only needed when the camera sees more than the
> screen. If your camera is aimed so the frame is **only the clean TV screen**, omit them.

**Decide:** `X < ~2%` → skip shading (don't pass `--shading-map` anywhere). Larger → keep it.
With the frozen tuning's power-law curve the map is exact at every brightness (with the stock curve
it was exact only near gray 180).

---

## Step 4 — Camera response (full frame, ~1 min)
Steps the shutter from 1/64× to 4× over a fixed LED white (and, optionally, a fixed TV gray),
capturing RAW alongside: this **measures the camera's tone curve** the fit uses.

```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain -w 1024 -h 576 $CAM $T --calib-tv-web \
  --calib-led-roi 705,265,20,106 --calib-out $CAL --ledcalib response
```
Open page → **tap begin**. Keep the room still. → `$CAL/camera_response.csv`. It prints the
recovered curve's local exponents and must say **`consistent single curve: the ISP behaves as
frozen`**. A `WARNING: ... NOT one consistent curve` means the tuning file isn't in effect (missing
`$T`?) or something moved — fix before continuing.

Optional second source: add `--calib-tv-roi x,y,w,h` **in full-frame (1024×576) coordinates** (this
phase has no crop) to also measure the TV gray (`--response-tv-level`, default 180).
`--response-led-level` sets the LED drive (default 255).

---

## Step 5 — TV calibration (cropped + shaded)
How the camera sees the TV's colors, through the same crop + shading as production.

```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --ledcalib tv --calib-tv-web \
  $GEO $CAM $T --calib-tv-roi 278,165,214,104 \
  --shading-map $CAL/shading.csv --calib-out $CAL --calib-tv-grid 6
```
Open page → **tap begin**. 6³ = 216 patches (~few min) → `$CAL/tv_sweep.csv`. (Drop
`--shading-map` if you skipped Step 3.)

## Step 6 — LED calibration (full frame)
Drives the LEDs through a sweep and measures the wall glow — on **RAW and ISP at once** (RAW
doesn't clip where the ISP does). The page blacks the TV automatically.

```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --ledcalib led --calib-tv-web \
  -w 1024 -h 576 $CAM $T --calib-led-roi 705,265,20,106 \
  --calib-out $CAL --ledcalib-led-grid 16
```
Open page → **tap begin**. No `--crop`/`--shading-map` (the wall is outside the TV). Dark room.
Writes `$CAL/led_sweep.csv`, `baseline.json`, `meta.json`.

---

## Step 7 — Build the LUTs (offline, no camera)
One measurement, three fits, so you can see what each change does:
```bash
# A: frozen tuning alone (legacy power-law camera model)
./Release/hyperion-cam --ledcalib-fit --calib-out $CAL --lut-out $CAL/lut_A.cube \
  --calib-response off --calib-knee off --calib-dark-floor off
# B: + measured camera response, LEDs on RAW
./Release/hyperion-cam --ledcalib-fit --calib-out $CAL --lut-out $CAL/lut_B.cube \
  --calib-knee off --calib-dark-floor off
# C: + soft knee + dark fallback (the defaults)
./Release/hyperion-cam --ledcalib-fit --calib-out $CAL --lut-out $CAL/lut_C.cube
```
Each run rewrites `$CAL/fit_report.json` (so it describes the last fit, C). Check its output:
`Camera response: measured curve`, `LED sweep on RAW ... relative RMS` (under ~5%), the knee line
(`camera white needs N x full LED output`) and the dark floor. `gray_axis` in the report lists
the drive the LUT sends for each measured TV gray.

> **White anchor level / clipping.** The fit anchors the von-Kries white at the **white-anchor
> neutral level** — the `--ledcalib-white-level` the sweeps ran with (default mid-gray **127**),
> i.e. where you eye-matched WB. Override with `--calib-white-anchor-level N`. If that level clips,
> the fit falls back to the nearest clean neutral and says so.

> Tunables: `--calib-knee <start %>` (default 85; `off`), `--calib-dark-floor <camera luma>`
> (default auto = TV black + 8 codes; `off`).

## Step 8 — Eye check (the acceptance test)
Run the normal production pipeline with a LUT plus the slider page:
```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --nv12 $GEO $CAM $T \
  --shading-map $CAL/shading.csv --lut $CAL/lut_C.cube --eyecheck-web
```
Open `http://<pi>:8080/` fullscreen on the TV PC. **gray** mode: slide the brightness 0..255
(←/→ ±1, ↓/↑ ±16, presets) and compare the LED glow with the TV at every level — darks, mid and
brights. **colour** mode: hue + saturation at that brightness. The panel hides itself after 4 s
(mouse/keys bring it back). Repeat with `lut_A.cube` / `lut_B.cube` to compare (Ctrl-C, change
`--lut`, rerun). If C still drifts somewhere, note where and how (level, direction) — the fallback
is a multi-point eye calibration of the white curve.

---

## Step 9 — Apply
Edit the `hypercam` service `ExecStart` (add the flags), then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart hypercam
```
Production line:
```bash
... --nv12 -w … -h … --crop … -t … -b … -s … --focus … \
    --tuning $CAL/tuning.json --saturation 1 \
    --shading-map $CAL/shading.csv \
    --lut $CAL/lut_C.cube
```
- `--tuning` — the frozen tuning file. A LUT is only valid with the tuning file it was measured
  with: never combine an old LUT with the new tuning (or the reverse).
- `--shading-map` — TV/lens uniformity correction.
- `--lut` — the color LUT. Trilinear, applied in the warp shader, composes with shading.
Re-enable HyperHDR LED smoothing for normal viewing.

---

## Quick reference

| Phase | Command core | Tuning | Crop | Shading | ROIs |
|---|---|---|---|---|---|
| TV RAW probe | `--ledcalib tv-raw` | any | yes | no | `--calib-tv-roi` |
| Frozen tuning | `--make-tuning F --from-calib D` | writes | — | — | offline |
| Shading | `--ledcalib shading` | `$T` | yes | produces | none (uses quad) |
| Response | `--ledcalib response` | `$T` | no | no | led (+ tv, full-frame coords) |
| TV | `--ledcalib tv` | `$T` | yes | applied | `--calib-tv-roi` |
| LED | `--ledcalib led` | `$T` | no | no | `--calib-led-roi` |
| Fit | `--ledcalib-fit` | — | — | — | offline |
| Eye check | production + `--eyecheck-web` | `$T` | yes | applied | — |

**Grids are edge-dense:** the TV/LED cube levels are clustered near 0 and 255 (cosine/Chebyshev
nodes — e.g. 6 → `0,24,88,167,231,255`), since crush/clip live at the ends. The runtime LUT stays
a uniform 33³ and interpolates the fitted model.

**Tunables:** `--calib-tv-grid` (216=6³), `--ledcalib-led-grid` (cube + gray ramp + axis ramps),
`--lut-size` (33), `--patch-size`/`--patch-x`/`--patch-y`, `--shading-grid`, `--ledcalib-frames`,
`--ledcalib-settle-ms`, `--response-led-level`, `--response-tv-level`.

**Diagnostics:** `--ledcalib gamma [--gamma-raw]` (TV→camera transfer) + `tools/plot-gamma.py`;
`--ledcalib gamma_leds` (the same ramp driven into the LEDs); `--ledcalib-web-dryrun` (test the
page); `HYPERCAM_CALIB_RAW_DUMP=<dir>` during any RAW phase dumps one ISP + RAW frame pair and the
ISP→RAW mapping (`isp.ppm`, `raw.ppm`, `map.txt`) to check the geometry.

**Recalibrate when:** HyperHDR color/WB changes, camera moves, TV picture-mode changes, or you
regenerate the tuning file. (A libcamera update no longer matters: the frozen file is yours.)
