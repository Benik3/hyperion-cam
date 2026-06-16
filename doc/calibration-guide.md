# LED color calibration — user guide

End-to-end: shading → TV sweep → LED sweep → fit → apply. Design/maths live in
`led-lut-calibration.md`; this is the how-to with commands.

> Usage doc kept here for convenience; the canonical home is the GitHub Wiki.

## What you produce
- `shading.csv` — TV/lens uniformity correction (optional; only if your TV drifts).
- `lut.cube` — 3D color LUT making the LED glow match the TV.

Both are applied as **grabber flags** (edit the `hypercam` service `ExecStart`), not via a
web UI. The web page only shows calibration patches on the TV.

## The web interface
A PC on the **same TV HDMI input/picture-mode you normally watch** runs a fullscreen browser at
`http://<pi-ip>:8080/`. The grabber drives the patches; the page acks each paint. That's its
only role.

## Ground rules (read once)
- **Dark room.** The page keeps the PC awake (Screen Wake Lock) and shows a tiny moving corner
  marker so most TVs won't screensaver. If your TV still blanks on a static image, disable its
  screensaver. Also disable **eco/ambient auto-dimming** (corrupts sweeps).
- HyperHDR running, grabber the only source, **HyperHDR color + smoothing settings final** — they
  bake into the LUT; change later → recalibrate.
- Stop the service while calibrating, restart after:
  ```bash
  sudo systemctl stop hypercam      # before each run
  sudo systemctl start hypercam     # when done
  ```
- Examples use a sample production geometry — substitute yours:
  `--crop 1690,1034,1160,742 -t 0,26:640,0:0,450:640,480 -b 2.632,1.626 -s 13600 --saturation 0.7383`
- Reuse one output dir for every phase: `--calib-out cal`.

---

## Step 0 — Prep

### 0a. White balance (first; don't change after)
Set white in **HyperHDR's white balance** so the LED glow matches the TV white by eye (this is
where you fix bluish LEDs). It freezes into the LED sweep, so set it **before** measuring. This
replaces the Phase-0 eye anchor — the fit derives the white correction from the sweeps.

### 0b. ROIs and aim
The page shows a 10%-of-screen **window patch** (dodges TV auto-dimming). Find where it lands and
pick the TV ROI inside it:

```bash
# Terminal A — show the example patch (camera-free); DON'T tap begin:
./Release/hyperion-cam --ledcalib-web-dryrun --calib-tv-port 8080 --patch-size 10
#   open http://<pi>:8080/ fullscreen on the TV PC -> red rectangle marks the patch

# Terminal B — capture the camera's cropped view:
sudo systemctl stop hypercam
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --imgout /tmp/aim.jpg --crop 1690,1034,1160,742
```
Open `/tmp/aim.jpg`: read `--calib-tv-roi x,y,w,h` **inside the red rectangle** (off the red
border). For the **LED ROI**, capture a full-frame still (no `--crop`) with the LEDs lit and read
`--calib-led-roi x,y,w,h` on a clearly-lit patch of wall.

---

## Step 1 — Shading calibration (optional; do first)
Maps lens + TV-panel non-uniformity so edge LEDs aren't miscolored. Uses production crop+transform;
a safe gray (never white).

```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --ledcalib shading --calib-tv-web \
  --crop 1690,1034,1160,742 -t 0,26:640,0:0,450:640,480 \
  --calib-out cal --shading-grid 24x14
```
Open page → **tap begin**. Fills the TV with gray 180, samples a 24×14 grid, writes
`cal/shading.csv`, prints `max colour (R/G,B/G) drift center->edge: X%`.

**Decide:** `X < ~2%` → skip shading (don't pass `--shading-map` anywhere). Larger (common on LCD)
→ keep `cal/shading.csv`.

**Orientation check (once):** apply it, show flat gray, confirm corners match center — flatter,
not worse:
```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --nv12 \
  --crop 1690,1034,1160,742 -t 0,26:640,0:0,450:640,480 -b 2.632,1.626 -s 13600 \
  --saturation 0.7383 --shading-map cal/shading.csv
```
Corner worse → axis flip, one-line fix.

---

## Step 2 — TV calibration (cropped + shaded)
How the camera sees the TV's colors, through the same crop + shading as production.

```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --ledcalib tv --calib-tv-web \
  --crop 1690,1034,1160,742 -t 0,26:640,0:0,450:640,480 \
  --calib-tv-roi 320,360,80,60 \
  --shading-map cal/shading.csv \
  --calib-out cal --calib-tv-grid 6
```
Open page → **tap begin**. 6³ = 216 patches (~few min) → `cal/tv_sweep.csv`. `INFO: TV shading
gain at ROI …` confirms shading applied. (Drop `--shading-map` if you skipped Step 1.)

---

## Step 3 — LED calibration (full frame)
Drives the LEDs through a sweep, measures wall glow. The page blacks the TV automatically.

```bash
sudo ./Release/hyperion-cam -c /tmp/hyperhdr-domain --ledcalib led --calib-tv-web \
  --calib-tv-roi 320,360,80,60 --calib-led-roi 900,400,120,80 \
  --calib-out cal --ledcalib-led-grid 8
```
Open page → **tap begin**. No `--crop`/`--shading-map` (wall is outside the TV; full frame). ~10–15
min, dark room. Writes `cal/led_sweep.csv`, `cal/baseline.json`, `cal/meta.json`.

---

## Step 4 — Build the LUT (offline, no camera)
```bash
./Release/hyperion-cam --ledcalib-fit --calib-out cal --lut-out cal/lut.cube --lut-size 33
sudo systemctl start hypercam        # camera free again
cat cal/fit_report.json              # rel_resid_p95 small (<0.10); note gamut %
```
Reads `led_sweep.csv` + `baseline.json` + the **white point from `tv_sweep.csv`** (no
`white_anchor.json` → derives von-Kries white from the sweeps = your HyperHDR WB). Writes
`cal/lut.cube` + `cal/fit_report.json`.

---

## Step 5 — Apply
Edit the `hypercam` service `ExecStart` (add the flags), then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart hypercam
```
Production line:
```bash
... --nv12 --crop … -t … -b … -s … --saturation … \
    --shading-map cal/shading.csv \
    --lut cal/lut.cube
```
- `--shading-map` — TV/lens uniformity correction.
- `--lut` — the color LUT. Trilinear, applied in the warp shader, composes with shading.

---

## Quick reference

| Phase | Command core | Crop | Shading | ROIs |
|---|---|---|---|---|
| Shading | `--ledcalib shading` | yes | produces | none (uses quad) |
| TV | `--ledcalib tv` | yes | applied | `--calib-tv-roi` |
| LED | `--ledcalib led` | no | no | tv + led |
| Fit | `--ledcalib-fit` | — | — | offline |

**Tunables:** `--calib-tv-grid` (216=6³), `--ledcalib-led-grid` (8³ + gray ramp), `--lut-size`
(33), `--patch-size`/`--patch-x`/`--patch-y`, `--shading-grid`, `--ledcalib-frames`,
`--ledcalib-settle-ms`.

**Diagnostics:** `--ledcalib gamma [--gamma-raw]` (TV→camera transfer) + `tools/plot-gamma.py`;
`--ledcalib-web-dryrun` (test the page).

**Recalibrate when:** HyperHDR color/WB/smoothing changes, camera moves, TV picture-mode changes.
