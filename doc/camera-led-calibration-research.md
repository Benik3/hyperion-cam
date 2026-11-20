# Camera-based LED calibration: a primary-source research review

Status: research review, 2026-11-20. It covers the code at `884e900` on branch `LED-calibration`.
The new camera-model code (812e1ab..884e900) **has not yet run on the real TV**. Everything below
was checked against primary sources: upstream code at the exact installed versions, kernel docs,
the Raspberry Pi tuning guide, the papers and the installed files and configuration on this Pi.
Where a source could not be reached, the claim is marked **UNVERIFIED**.

**Versions examined (all exact matches unless noted):**

| Component | Version | Source used |
|---|---|---|
| libcamera (RPi fork) | `0.7.1+rpt20260609-1` (dpkg) | tag `v0.7.1+rpt20260609` = commit `06c385619acb10bbfb33f52f3abeb8f8c095f42b` |
| HyperHDR | `22.0.0~trixie~beta1` (dpkg); binary string `HEAD detached at v22.0.0.0beta1 … e3dfe92` | tag `v22.0.0.0beta1` = commit `e3dfe922d55358406075de9496f7159854f69b21` |
| imx708 kernel driver | running kernel `6.12.47+rpt-rpi-v8` | `raspberrypi/linux` branch `rpi-6.12.y` head `dcc3a57b…` (2026-11-20). The exact match to 6.12.47 is **UNVERIFIED**. |
| V4L2 pixfmt docs | — | `torvalds/linux` master `65e4fd94…` |
| RPi tuning guide | "Camera Algorithm and Tuning Guide", RP-009070-TC, Release 2, build 20/10/2026 | `datasheets.raspberrypi.com/camera/raspberry-pi-camera-guide.pdf` (redirects to `pip-assets…/RP-009070-TC-3-…pdf`) |
| Tuning file | `/usr/share/libcamera/ipa/rpi/vc4/imx708.json` | read locally |
| HyperHDR config | `~/.hyperhdr/db/hyperhdr.db` (opened read-only, `mode=ro`), `~/.hyperhdr/lut_lin_tables.3d` | read locally |

**Citation conventions.**

- `LC:path:lines` = `https://github.com/raspberrypi/libcamera/blob/06c385619acb10bbfb33f52f3abeb8f8c095f42b/path#Lx-Ly`
- `HH:path:lines` = `https://github.com/awawa-dev/HyperHDR/blob/e3dfe922d55358406075de9496f7159854f69b21/path#Lx-Ly`
- `KD:lines` = `https://github.com/raspberrypi/linux/blob/dcc3a57b713fe33c33a9eba7e0704bc87aed6698/drivers/media/i2c/imx708.c#Lx-Ly`
- Guide sections are cited as `Guide §x (PDF p.N)`. N is the PDF page index; the printed page number is N−1.
- Our own code is cited as repo-relative `path:line`. §9 lists every source with its full URL.

---

## 1. Executive summary: top findings ranked by impact on the LUT result

1. **BUG (new code): the dark-floor fallback assumes equal R=G=B drive is the eye-matched white at every level. This is false with HyperHDR v22.**
   - **Mechanism.** HyperHDR v22 linearises each LED colour with the sRGB EOTF, multiplies by the "temperature" gains, and re-encodes with the sRGB OETF (`HH:sources/infinite-color-engine/InfiniteProcessing.cpp:174-191, 540-547`).
   - **Effect on equal-drive white.** With our settings (R 1 / G 0.38 / B 0.22), an equal R=G=B input gives these output ratios:

     | Input level | G/R | B/R |
     |---|---|---|
     | ≤ ≈10 | 0.38 | 0.22 |
     | 32 | 0.52 | 0.33 |
     | 127 (eye-match) | 0.632 | 0.480 |
     | 255 | 0.650 | 0.506 |

   - **Consequence.** Equal drive is the eye-matched white only at the 127 anchor. The dark-floor nodes (`src/ledCalibFit.cpp:1227-1246, 1263-1274`) therefore come out red/orange: G is about −40% and B about −54% against the eye-matched mix.
   - **Second problem.** The floor criterion is BT.601 *luma* of the encoded camera RGB (`src/ledCalibFit.cpp:1266`). It therefore also fades saturated dark colours to "neutral": blue up to B≈64, fully neutral up to ≈32, and red up to ≈24.
   - **Fix.** Use the anchor's *light* direction, `e = k·e(anchorDrive)`, inverted per channel, and gate on the max channel rather than luma. (§2.2, §5.5)

2. **WRONG ASSUMPTION (new code): ISP clipping at the white-balance gain stage is invisible to every clip check.**
   - **Mechanism.** The ISP applies the colour gains (R 2.632, B 1.626) *before* the CCM. The vc4 IPA forces all colour gains ≥ 1 to avoid "saturation artifacts" (`LC:src/ipa/rpi/vc4/vc4.cpp:371-376`), which implies a full-scale clip after the gain stage. The comment itself is about gains below 1, so the clip position is an inference. Our own LED data shows the effect directly: LED red pins at 246 while RAW has about 3× headroom (`doc/led-lut-calibration.md:43`).
   - **Thresholds.** For any stimulus redder or bluer than the WB white, ISP R saturates at 1/2.632 = **38%** of the sensor range and B at **61.5%**.
   - **What the checks miss.** `fitTvMatrix` only checks *sensor* saturation (`src/tuning.cpp:327-334, 459-461`). `fitRawToLin` filters on *post*-CCM ISP values and RAW saturation (`src/ledCalibFit.cpp:794-797`). A pre-CCM-clipped LED-red sample can therefore pass as "clean" and bias the RAW→ISP map. A TV-fitted CCM cannot undo a clip that happens upstream of it.
   - **Fix.** Run every phase and production with `-b 1,1`. The TV-fitted CCM already folds in `1/wb` (`src/tuning.cpp:463-467`), so the ISP clip then coincides with the sensor clip, which *is* detected. Also add an explicit pre-CCM check. (§3.3)

3. **HyperHDR's NV12 decode matches our encoder exactly. No range or matrix mismatch.**
   - **Which table.** With `flatbufServer.hdrToneMapping=false`, flatbuffer NV12 is decoded through table 2 of `~/.hyperhdr/lut_lin_tables.3d` (`HH:sources/utils/LutLoader.cpp:82-95`).
   - **Measured.** I fitted that table on the installed file. It is **limited-range BT.601** to rounding: coefficients 1.1640 / 1.5958 / −0.3918 / −0.8130 / 2.0172, max residual 0.50 code. Its source is the default LUT, `BT601` + `LIMITED` (`HH:sources/lut-calibrator/LutCalibrator.cpp:1802-1838, 1538-1541`).
   - **Round trip.** RGB → our NV12 → HyperHDR differs from the input by at most 2 codes (mean 0.40); 84% of colours are off by 1 in some channel. (§5.2)

4. **HyperHDR averages each LED area in linear light but sends sRGB-encoded drive to linear-PWM LEDs.** This is a production-only error that uniform-patch calibration cannot see.
   - **Mechanism.** `HH:sources/base/ImageColorAveraging.cpp:216-235` takes the mean of sRGB-EOTF-linearised pixels. With `gamma 1`, the drive equals `OETF(mean)`.
   - **Size.** An LED area that is 10% white and 90% black gets drive 89 (35% PWM) instead of 10%. A 50/50 area gets 188 instead of 128.
   - **Fix (needs re-calibration).** Set HyperHDR gamma to about 2.2–2.4. (§5.4)

5. **The frozen tuning really freezes the ISP, with two residual issues.** Each mechanism below was verified in source.
   - **Contrast enhancement.** `ce_enable 0` disables it completely (`LC:src/ipa/rpi/controller/rpi/contrast.cpp:170-177`; Guide §5.11.5, PDF p.44 and p.64).
   - **ALSC.** `n_iter 0` keeps the lambdas at 1, so only the calibrated tables apply (`LC:…/alsc.cpp:758-785`; Guide p.64: "turning off the adaptive algorithm completely").
   - **ALSC colour temperature.** With manual `ColourGains`, the colour temperature is derived from the gains (`LC:…/awb.cpp:218-228`), so it is fixed.
   - **Saturation** is applied on top of the CCM (`LC:…/ccm.cpp:130-145, 173-179`).
   - **Tuning file.** `LIBCAMERA_RPI_TUNING_FILE` is read when the IPA is created in `CameraManager::start()` (`LC:src/libcamera/ipa_proxy.cpp:39-48`, `LC:src/libcamera/pipeline/rpi/common/pipeline_base.cpp:1177`).
   - **Gamma knots** match `src/tuning.cpp:149-156` (`LC:src/ipa/rpi/vc4/vc4.cpp:423-433`).
   - **Residual issue 1 (BUG, small): digital gain in the response sweep.** The IPA adds digital gain to compensate for exposure quantisation (`LC:src/ipa/rpi/common/ipa_base.cpp:1846-1855`). RAW does not see this gain; the ISP does. The response fit ignores it. The error is estimated at up to about 2–4% at the shortest (212 µs) sweep steps. (§4.3)
   - **Residual issue 2: tv-raw runs under the stock tuning.** That tuning has contrast enhancement on and adaptive ALSC (±5% per cell). The CCM fit decodes those ISP readings with the static curve (`src/tuning.cpp:271-287, 402-451`). The TV-fitted CCM's "white neutral / primaries on axes / ≥0.5% margin" guarantee can therefore be off by up to about 5% in production. (§4.2a)

6. **The frozen "power-2.2" curve is linear below code 38.5.** The first vc4 knot is 1024/65535, and the segment from 0 to that knot is a straight line. Any pure power-law model is wrong there by up to 99%. This affects:
   - legacy fit "A";
   - the response-fit prior `kGamma0=2.2` (`src/cameraResponse.cpp:94`);
   - the doc claim that the shading map is "exact at every level" (`doc/led-lut-calibration.md:272`). At code 20 a 0.9 encoded gain is 0.90 in linear light; at codes ≥60 it is 0.79.

   **Fix.** Linearise with the analytic inverse of the PWL in `tuning.json` (exact), and keep `camera_response.csv` as a validation. (§2.3)

7. **Trilinear interpolation leaks off-axis nodes into grays; 65³ costs nothing extra.**
   - **Leak.** At the midpoint of a cell's main diagonal, 75% of the trilinear weight comes from the six off-diagonal (non-gray) nodes, `3t(1−t)`. Diagonal tetrahedral interpolation uses 0%.
   - **Sources.** Kasson et al. 1995 recommend tetrahedral. The Cube spec §8 says readers "should use … tetrahedral interpolation for three-dimensional tables".
   - **Cost on v3d.** Our own measurements show 64³ costs the same as 33³ (`doc/led-lut-calibration.md:410-416`).
   - **Fix.** Use `--lut-size 65`: about 4× lower error where the LUT is smooth and about 2× at gamut kinks. (§2.4)

8. **The diagonal (von-Kries) white correction is sound across levels only under stated conditions.** It is exact for every gray level only if:
   - the camera linearisation is exact; and
   - the TV gray and the LED white keep their *spectral composition* as the level changes.

   §2.2 gives the derivation. A WOLED gray mix, saturated colours and per-LED die variation remain approximations. A full 3×3 metameric correction needs eye anchors at R, G, B as well as W (the four-colour-matrix idea of Ohno et al.).

Smaller confirmed items:

- **`imx708.json` was not locally edited.** Its md5 `79f40549…` equals the dpkg record and upstream from `v0.7.0+rpt20260205` on. `imx708.json_bak` (md5 `91fd7a88…`) is byte-identical to upstream `v0.6.0+rpt20251202`. The only differences are the new, disabled `rpi.nn.awb` block and `"enabled": true` on `rpi.awb`. Neither has any effect with manual white balance (§4.1).
- **`--gain` cannot go lower.** The imx708's minimum analogue gain is **1.123** (driver min gain code 112, `KD:68-71`; `LC:src/ipa/rpi/cam_helper/cam_helper_imx708.cpp:101-104`). The guide's "lower `--gain`" advice (`doc/calibration-guide.md:87`) cannot be followed at the default 1.0; only the shutter can reduce exposure.

---

## 2. Q1: The calibration method

### 2.1 Camera as comparator, the Luther condition and metamerism

- **The Luther condition.** A camera is colorimetric only if its spectral sensitivities are a linear combination of the CIE colour-matching functions. Physical cameras only approximate this (Finlayson & Zhu, IEEE TIP 2020, arXiv:2003.12645, abstract). The original Luther (1927) and Wyszecki & Stiles texts were not accessed (**UNVERIFIED** here).
- **The imx708 is not colorimetric.** Our data shows it. In the July sweeps an LED white that the eye matched to the TV read R 203.8 / G 144.5 / B 191.6 at drive 127, against a TV gray 127 reading of 128.4 / 133.6 / 132.4. This is the camera-versus-eye metamerism the design anticipates (`doc/led-lut-calibration.md:19-26`).
- **What is exact.** For a display whose light is a *linear combination of a fixed set of three spectra*, any 3-channel camera with independent channels is related to the eye by an exact 3×3 matrix. Write the LED light as `a` (die intensities); then camera = C·a and eye = E·a with constant 3×3 C and E, so eye = E·C⁻¹·camera.
  - The **WS2812 dies** satisfy this as long as PWM does not change the die spectra.
  - An **RGB TV** satisfies it too.
  - A **WOLED** panel has four subpixels (W+RGB) driven by internal processing, so its light is *not* a 3-D linear space. There the camera→eye relation is only approximately a 3×3.
  - This is the principle behind the "four-colour matrix" correction of tristimulus colorimeters for one display type (Ohno & Hardis, CIC 5 1997, pp. 301-305, **not accessed**; the NIST page for Part 2, Ohno & Brown 1998, reports residuals within 0.002 in x,y when calibration and use are limited to one display type).
- **What the system actually needs.** It needs the *camera→eye relation for TV light* and the *camera→eye relation for LED light* to differ only by something the eye anchor removes. One white anchor fixes one direction of that difference (§2.2). Saturated colours therefore keep a residual metameric hue error. This is already acknowledged (`doc/led-lut-calibration.md:428-429`).

### 2.2 A single von-Kries white correction at one level: when it holds across levels

Notation:

- C_T, C_L: camera responses to the TV subpixel spectra and the LED die spectra (3×3).
- E_T, E_L: eye (cone) responses to the same spectra.
- b, a: subpixel and die intensities.

Our LUT drives the LEDs so that `C_L·a = D·C_T·b`, where D is diagonal (`corr`, `src/ledCalibFit.cpp:1053-1067, 1098-1104`). The eye match requires `E_L·a ∝ E_T·b`, which gives:

- **Exact for all b** only if `D = k·C_L·E_L⁻¹·E_T·C_T⁻¹`. That is a full 3×3 in general.
- **At the anchor gray b_w** a diagonal D solves the 3 equations. By linearity it then also holds for **every b = s·b_w**, that is every gray level with the same spectral composition.

So one von-Kries anchor is valid at all gray levels **only if** all of these hold:

1. **Exact camera linearisation.** This was the old drift (§2.3).
2. **The TV's gray composition is level-invariant.** On a WOLED panel the W/RGB mix can vary with level. Our doc claims the TV gray is stable within ±5% as the camera sees it (`doc/led-lut-calibration.md:264-265`): **UNVERIFIED** here. After the frozen tuning, check it from `tv_sweep.csv` neutrals in linear light.
3. **The LED white at other levels is the same mix of dies, scaled.** The LUT achieves this through its model, *except* in the dark floor. There `neutralDrive` (`src/ledCalibFit.cpp:1227-1246`) uses equal *drive*. HyperHDR v22 does not map equal drive to an equal die mix at every level:
   - It computes `out_c = OETF(t_c · EOTF(v))` per channel, with t = (1, 0.38, 0.22) (`HH:…/InfiniteProcessing.cpp:174-191, 540-547`; LUT tables `HH:include/infinite-color-engine/InfiniteProcessing.h:202-240`).
   - The resulting G/R and B/R ratios are tabulated in §1, finding 1.
   - The July LED sweep agrees with this toe. G-only drive 11 (HyperHDR output 4) barely registers in the camera (0.12), while R-only drive 5 (output 5) reads 1.17. The old July LUT's dark gray drives (5.5, 10.7, 7.9) at input 8 also over-drive G, which is the shape a compensation of this toe would take. That second observation is supportive only: the old camera model was itself wrong.

   **This is a wrong assumption in the new code** (comments at `src/ledCalibFit.cpp:141-142, 1227`, design text `doc/led-lut-calibration.md:192-196`).

   **Fix.** Inside the dark floor, target `k·eW` with `eW = e(anchorDrive)` (`src/ledCalibFit.cpp:1061-1063`) and invert each channel: `drive_c = e_c⁻¹(k·eW_c)`. Solve for k by luminance as `neutralDrive` does now.

Separately, the floor gate `luma = 0.299R+0.587G+0.114B < black+8` on *encoded camera* values (`src/ledCalibFit.cpp:1266`, `1348-1352`) has a side effect. With the automatic floor of about 8.1 (OLED black ≈0.1), 28 nodes of the 33³ LUT are in the fade and 7 are fully neutral. Pure blue up to **63.75** and pure red up to **23.9** lie inside the fade. Trilinear interpolation spreads this toward B≈96. A deep-blue night scene is clearly resolvable (B≈32–64 codes), yet it would get a neutral glow. Gate on `max(R,G,B)` (or a per-channel SNR) instead.

### 2.3 Debevec–Malik response recovery: assumptions versus our implementation

What the paper says (Debevec & Malik, SIGGRAPH 97):

- **Reciprocity** (§2.1): "only the product EΔt is important … halving E and doubling Δt will not change the resulting" value.
- **f is monotonic**, so g = ln f⁻¹ exists (§2.1).
- **Objective** (Eq. 3/5): Σ{w(Z)[g(Z) − ln E − ln Δt]}² + λΣ[w(z)·g″(z)]².
- **Weights** (Eq. 4): hat function w(z) = z−Zmin for z ≤ ½(Zmin+Zmax), else Zmax−z.
- **Gauge**: g(Z_mid) = 0.
- **Colour** (§2.6): "reconstructing the imaging system response curve for each channel independently".

What `src/cameraResponse.cpp` does differently:

- **Exposure reference.** It uses the black-subtracted **RAW sum** of the same ROI as the known exposure (`src/cameraResponse.cpp:110-146`), not Δt. This removes the reciprocity and shutter-accuracy assumptions. It replaces them with *"RAW is linear in light"* and *"the ISP maps RAW to code through one fixed function"*.
- **Shared curve.** It fits one curve shared by all channels and ROIs, with a free log-scale per series. This is valid because vc4 applies a single gamma table to R, G and B (`LC:src/ipa/rpi/vc4/vc4.cpp:418-438`, one `bcm2835_isp_gamma`) and each ISP-linear channel is proportional to the RAW sum for a fixed spectrum.
- **Hat weights.** Floored at 0.05 (`:97`).
- **Gauge.** g(128) = 0 (`:190`).
- **Smoothness prior.** On g − 2.2·ln z, not on g (`:94-95, 181-189`).

Where these assumptions break:

- **The curve does *not* stay the same across exposure steps, because of digital gain (BUG, small).**
  - **What happens.** After quantising the requested exposure to sensor lines, the IPA multiplies the ISP digital gain by `requested/actual` (`LC:src/ipa/rpi/common/ipa_base.cpp:1846-1855`, "make up for any quantisation"). The ISP output therefore corresponds to the *requested* exposure while RAW reflects the *actual* one.
  - **Size.** The imx708 1536×864 mode has a line time of 5216 px / 566.4 MHz = **9.21 µs** and an exposure step of 2 lines (`KD:722-742`). At the shortest sweep step (13600/64 = 212.5 µs, `src/ledCalib.cpp:1921-1936`), one or two lines is about 2–4%. The error shrinks as 1/exposure. This size is an estimate: whether the driver's 2-line step is reflected in the IPA's "actual" value was not verified.
  - **Fix.** `camera_response.csv` does not record `DigitalGain` (`src/ledCalib.cpp:2039`). Record it per row from metadata (reported post-quantisation, `LC:…/ipa_base.cpp:1609-1613`) and use `light·dg`. Alternatively, snap the plan to whole even lines.
- **The prior is wrong in the frozen curve's linear toe.** The vc4 gamma is a 33-point PWL whose first segment runs from 0 to x = 1024/65535 = 1.56% linear (`LC:src/ipa/rpi/controller/controller.cpp:42`, `numGammaPoints = 33`). With the frozen power-2.2 curve that segment maps codes 0..**38.5** *linearly* (local exponent 1, not 2.2). Comparing a pure power-law inverse with the actual PWL inverse:

  | Code range | Max relative error of a pure power-law inverse |
  |---|---|
  | 1–38 | 99% |
  | 38–53 | 3.2% |
  | 53–100 | 1.1% |
  | above 100 | <0.1% |

  The fit can follow the toe only as far as its data and λ allow. The self-test uses the stock curve, whose linear toe ends at code 19.6, and reports accuracy only for codes 16..250 (`doc/led-lut-calibration.md:201-205`). The frozen curve is *known exactly* (we wrote it), so use its analytic inverse and keep the measured curve as a residual check.
- **RAW linearity up to saturation and the true black pedestal** are assumed (§3.2). The response's dimmest samples (RAW sum just above 3 counts, `src/cameraResponse.cpp:134`) are sensitive to a pedestal error δ: the sum shifts by 3δ.

### 2.4 Trilinear versus tetrahedral interpolation near the gray axis

- **Kasson, Nin, Plouffe & Hafner** (J. Electronic Imaging 4(3):226-250, 1995, doi:10.1117/12.208656; abstract via Crossref): "tetrahedral interpolation, with close to half the computational cost of trilinear interpolation, is capable of providing better accuracy. Of the tetrahedral interpolation techniques, one diagonal extraction from cubic packing is useful as a general-purpose color space interpolator." The full text was not accessed; its quantitative tables are **UNVERIFIED** here.
- **Adobe Cube LUT Spec 1.0:**
  - §7.1: "The values in the table shall be set so that tetrahedral interpolation will generate correct output values".
  - §8: readers "should use … tetrahedral interpolation for three-dimensional tables".
- **Gray-axis derivation (exact, no citation needed).** For the point (t,t,t) inside a cell, trilinear interpolation gives weight (1−t)³ to node 000 and t³ to node 111. The six off-diagonal nodes share a total weight of **3t(1−t)**, which is 0.75 at t = ½. Diagonal tetrahedral interpolation (tetrahedra sharing the 000–111 edge) gives exactly `(1−t)f000 + t·f111` on the axis. So a trilinear gray is only as good as its tinted neighbours. The gamut and knee kinks near white produce the "≤1.6% dip" our doc reports (`doc/led-lut-calibration.md:433-436`).
- **Our choice.** Trilinear was chosen because tetrahedral costs +8 ms on v3d (measured, `doc/led-lut-calibration.md:407-422`). 64³ trilinear costs the same as 33³ (3.0 ms render). The loader and fit accept up to 65 (`src/hyperion-cam.cpp:2123`, `src/ledCalibFit.cpp:1487`). Interpolation error falls as h² where the LUT is smooth and roughly as h at kinks, so **65³ cuts the gray-axis dip by about 2–4×** at no runtime cost.
- **Optional.** Because the reader is trilinear, the spec's own logic ("values … set so that [the reader's] interpolation will generate correct output") suggests optimising the node values for trilinear. Do a least-squares fit at a dense set of gray and near-gray inputs, instead of sampling the model only at the nodes.

### 2.5 Additivity of the LED mixing matrix

- **What holds.** Light from separate dies superposes exactly, and wall reflection is linear. The model `lin − baseline = A·[e_R, e_G, e_B]` (`src/ledCalibFit.cpp:833-1092`) also needs the HyperHDR→LED path to be **channel-separable**. In our configuration every active stage is per-channel: temperature, the sRGB OETF and rounding. The HSV gains, colour-calibration LUT and power limit are all disabled (§5.5). A non-identity colour calibration, HSV gain or `powerLimit<1` would break separability.
- **What can break it** (physics; not measured here, **UNVERIFIED**):
  - **Thermal droop.** The red die's efficiency falls with junction temperature, so all-channels-on reads less red than the sum of single channels.
  - **Supply sag.** Voltage drop along the strip at high total current affects blue and green first, since they have the higher forward voltage.
  - **Neither is in the model.** Both show up only as fit residual (`rel_resid_p95`).
- **Test.** Interleave reference whites through the sweep. Also compare the gray ramp with A·e built from the axis ramps at high drive.

### 2.6 Wall reflectance and LED-ROI geometry

- **Wall tint.** The wall's spectral reflectance multiplies the LED spectra for both the camera and the viewer. The eye match is done on the same glow, so a tinted wall is consistent; it only adds its own camera/eye metamerism.
- **Shading cancels for the white.** Lens shading at the LED ROI is a constant per-channel multiplier in linear light (ALSC is a multiplicative gain grid, Guide §5.10.2, PDF p.39; vc4 programs 4 gain planes R, G, G, B per cell, i.e. Bayer-domain before demosaic/CCM, `LC:src/ipa/rpi/vc4/vc4.cpp:521-573`). It cancels in `corr` because `corr` compares the same two ROIs (`src/ledCalibFit.cpp:1064-1067`). This is consistent with the doc (`doc/led-lut-calibration.md:351-352`).
- **Not modelled, all UNVERIFIED:**
  - (a) Individual WS2812s vary in die wavelength and brightness, so one ROI calibrates the LEDs that light it.
  - (b) The R, G and B dies sit at slightly different positions with different emission patterns, so the R/G/B ratio of the glow depends on where on the wall you look.
  - (c) The viewer's angle differs from the camera's; a glossy wall breaks the Lambertian assumption.
- **Test.** Measure two or three LED ROIs (left, top, right) once and report the spread of `corr`.

---

## 3. Q2: RAW handling

### 3.1 Verification against primary sources

| Item | Primary source | Our code | Verdict |
|---|---|---|---|
| SBGGR10 CSI-2 packing: 4 px in 5 bytes, 5th byte holds the LSBs, pixel 0 in bits 1:0 | V4L2 `pixfmt-srggb10p.rst` lines 23-27, 44-51 | `src/ledCalib.cpp:1264-1273` | CONFIRMED |
| BGGR phase (row 0 B G, row 1 G R) | same doc, example table | `src/ledCalib.cpp:1246-1262` (`'BG'` → row 0: B,G; row 1: G,R) | CONFIRMED |
| `SensorBlackLevels`: order R, Gr, Gb, B, units out of 16-bit | `LC:src/libcamera/control_ids_core.yaml:627-636` | `src/hyperion-cam.cpp:2541-2549` (divides by 2^(16−10) = 64, averages Gr/Gb) | CONFIRMED. **But** the RPi IPA reports the *tuning-file constant* (`black_level` 4096 = 64 @10-bit), not a measurement (`LC:src/ipa/rpi/controller/rpi/black_level.cpp:33-45, 56-67`; `LC:src/ipa/rpi/common/ipa_base.cpp:1625-1631`). The ISP subtracts the same constant (`LC:src/ipa/rpi/vc4/vc4.cpp:403-416`), so RAW and ISP are consistent with each other. Whether 64 is the true pedestal is **UNVERIFIED** (§3.2). |
| `ScalerCrop` is in native active-area pixels, even when binning | `LC:src/libcamera/control_ids_core.yaml:685-699` | `src/hyperion-cam.cpp:2550-2565` | CONFIRMED |
| `ScalerCropMaximum` = the mode's mandatory crop | `LC:src/libcamera/property_ids_core.yaml:667-680` | same | CONFIRMED |
| 1536×864 mode = 2×2 binned crop at (768,432)/3072×1728 of the active area | `KD:722-742` (`left = PIXEL_ARRAY_LEFT+768`, `top = …+432`, 3072×1728) | `src/ledCalib.cpp:1399-1408` (self-test), doc | CONFIRMED |
| A RAW stream sets the sensor mode by its size | `LC:src/libcamera/pipeline/rpi/common/pipeline_base.cpp:211-225` | `src/hyperion-cam.cpp:4014-4058` (probe by `ScalerCropMaximum`) | CONFIRMED; the probe is a correct guard |
| The Raw-role buffer is the exact frame the ISP processed in the same request | vc4 pipeline: the Unicam buffer is queued as ISP input (`LC:src/libcamera/pipeline/rpi/vc4/vc4.cpp:911-925`), then returned to the stream/request after the ISP consumes it (`:843-855`); request ↔ Bayer pairing in `tryRunPipeline` (`:965-1007`) | dual stream in the same request | CONFIRMED: same exposure, same frame |

### 3.2 Linearity, black stability and sensor-side processing (imx708)

- **Linear up to saturation?** There is no public imx708 datasheet: **UNVERIFIED**. It can be measured from data we already collect. In `camera_response.csv`, RAW-sum ÷ `exposure_us` should be constant for every unsaturated step. Today the code checks only the reference steps for drift (`src/cameraResponse.cpp:245-280`). Recommendation 10.
- **Black level stable across gain?** The value is not measured at all (see above). With fixed gain it is constant per run. For the LED sweep the baseline subtraction (`src/ledCalibFit.cpp:851-861`) removes any pedestal error. It is *not* removed in `tv-raw` (black = 0 assumed, `src/tuning.cpp:338`) or in the dim end of the response. One lens-cap RAW frame at the production gain settles it.
- **PDAF pixels.** The driver programs on-sensor PDAF-pixel ("SPC") correction gains (`KD:193-197, 1500-1517`), so PDAF sites in RAW are already gain-corrected by the sensor. Their density and residual are **UNVERIFIED**. Their effect on ROI means of thousands of pixels is expected to be negligible. ISP DPC strength is a constant 1 (`LC:…/dpc.cpp:45-50`).
- **Binning and HDR.** The mode is 2×2 binned (`KD:722-742`); binning is linear. The on-sensor HDR (`V4L2_CID_WIDE_DYNAMIC_RANGE`) defaults to 0 (`KD:1886-1887`); only the separate HDR mode list uses it (`KD:746-768`). The ISP-side `HdrMode` control is not set (default Off). Quad-Bayer re-mosaic and QBC line correction apply only to the full-resolution remosaic mode (`KD:27-29, 115-120`; `remosaic=false` for the binned modes).

### 3.3 The ISP clips before the CCM (white-balance gain stage)

The vc4 IPA forces extra digital gain whenever a colour gain is below 1 "which would cause saturation artifacts" (`LC:src/ipa/rpi/vc4/vc4.cpp:371-380`). In other words, the channel gains saturate a channel at the ISP's full scale. Applying R×2.632 and B×1.626 (`LC:…/vc4.cpp:356-365`) therefore pins ISP R at **38%** and B at **61.5%** of the sensor range, *before* the CCM. For a stimulus balanced to the WB white this matches the G sensor clip. For anything redder or bluer than that white it comes first: LED white "reads strongly red", and red and blue primaries are affected too. This matches our own measurement (`doc/led-lut-calibration.md:43`: "the clip is the ISP's WB gain, not the sensor"). The hardware pipeline itself is not publicly documented, so the exact clip placement is inferred from these two sources.

Consequences for the new code:

- **`fitRawToLin` keeps "clean" samples by post-CCM value only.** It uses 12..245 and RAW saturation (`src/ledCalibFit.cpp:794-797`). With the TV-fitted CCM scaled down so the TV peak lands at 95%, a pre-CCM-clipped LED-red sample can read below 245 after the CCM and still be used. That biases `rawToLin`. The only symptom is a higher `raw_map_resid`.
- **`fitTvMatrix` checks sensor clipping only** (`src/tuning.cpp:327-334, 459-461`). The 10% peak window is the brightest stimulus. If it clips at the WB stage, the "peak → 95%" scaling is computed from an unclipped prediction (`src/tuning.cpp:453-458, 468`) that the ISP cannot produce.
- **Fix.** Use `-b 1,1` everywhere and re-run `tv-raw` with it. `fitTvMatrix` already divides by the colour gains in effect (`S = diag(1/(wb·dg·a))`, `src/tuning.cpp:463-467`), so any WB works. The CCM coefficients grow by up to 2.6× on the R column; the warning at >7.9 still guards the ±8 range (`src/tuning.cpp:607-609`; clamp `LC:…/ccm.cpp:180-183`).
- **Side effect of `-b 1,1`.** The ALSC colour temperature changes, because it is derived from the gains (`LC:…/awb.cpp:222-228`). With gains 1,1 it clamps to the curve end (≈2500 K), so the 3000 K tables are used instead of about 6400 K → 5000 K. They are still static. Noise is unchanged, because the gain moves from WB into the CCM.

### 3.4 What else RAW could do

1. **TV side on RAW as well.** Add RAW columns to the `tv` sweep, as already done for `led`, and fit `M_TV` the way `fitRawToLin` fits the LED map. The white anchor `corr = M_TV·raw_TV / M_L·raw_LED` then no longer depends on the tone curve. The LUT input still needs ISP→linear, but the anchor, the most sensitive single number, would be curve-free.
2. **Analytic ISP model instead of a measured response.** With the frozen tuning, a uniform patch's ISP code is fully determined: `PWL(clip(CCM·clip(dg·wb·LS·(raw−black))))`. The LS gain can be computed from the tuning tables, the colour temperature and the mode crop (`LC:…/alsc.cpp:489-540` resample, `:787-814`). Predict it from RAW and compare with the dual-stream ISP values. If the residual is ≤1%, drop `camera_response.csv`, or keep it as a regression test.
3. **Exposure bracketing for the dark end.** RAW is linear, so `light = raw / exposure` merges brackets trivially. Re-measure the LED sweep's low drives and the TV black at 4× exposure. The dark end of the curves would then be measured instead of regularised (`--calib-curve-floor`, `src/ledCalibFit.cpp:681-726`).
4. **Pre-CCM clip detection** from RAW × gains (§3.3).

---

## 4. Q3: The ISP and the tuning JSON

### 4.1 Every stage in `imx708.json`

Present in the file (`/usr/share/libcamera/ipa/rpi/vc4/imx708.json`, version 2.0, target bcm2835): `rpi.black_level`, `rpi.dpc`, `rpi.lux`, `rpi.noise`, `rpi.geq`, `rpi.sdn`, `rpi.awb`, `rpi.nn.awb`, `rpi.agc`, `rpi.alsc`, `rpi.contrast`, `rpi.ccm`, `rpi.sharpen`, `rpi.af`, `rpi.hdr`, `rpi.sync`. The `_bak` file lacks `rpi.nn.awb` and `"enabled"` and is otherwise identical (diff of the normalised JSON).

| Stage | What it does to pixels | Adaptive per frame? | Frozen in our setup? |
|---|---|---|---|
| `rpi.black_level` (4096) | Subtracts 64 @10-bit per channel; also reported as `SensorBlackLevels` | No, constant (`LC:…/black_level.cpp:56-67`) | Static. True pedestal UNVERIFIED |
| `rpi.dpc` (default strength 1) | Adaptive defective-pixel filter | No, constant strength (`LC:…/dpc.cpp:45-50`) | Static; spatial only |
| `rpi.lux` | None (estimate only) | Yes, from stats every frame (`LC:…/lux.cpp:84-120`) | Not needed: its consumers are inactive. GEQ `strength` and CCM `saturation` PWLs are absent; AWB and AGC are manual |
| `rpi.noise` | None; noise profile ∝ √gain / mode factor | From device gain (`LC:…/noise.cpp:59-80`) | Fixed by fixed gain |
| `rpi.geq` (offset 215, slope 0.00287) | Green equalisation | Scaled by gain, and by lux only if `strength` is present (`LC:…/geq.cpp:55-80`; Guide §5.5, PDF p.23-24) | Fixed: no `strength` |
| `rpi.sdn` (defaults: deviation 3.2, strength 0.75) | Spatial denoise, plus colour denoise by mode | From noise profile (`LC:…/sdn.cpp:49-68`; Guide §5.6, PDF p.24-25) | Fixed. Approximately mean-preserving on uniform ROIs |
| `rpi.awb` | R/B gains before the CCM; colour-temperature estimate | Adaptive in auto | `AwbEnable=false` + `ColourGains` (`src/hyperion-cam.cpp:4374-4375`). Colour temperature comes from the gains (`LC:…/awb.cpp:209-230`) |
| `rpi.nn.awb` | — | Disabled (`"enabled": false`; skipped before creation, `LC:src/ipa/rpi/controller/controller.cpp:146-155`) | n/a |
| `rpi.agc` | Exposure, analogue gain, ISP digital gain | Manual: fixed (`LC:…/agc_channel.cpp:765-787, 880-950`). **Digital gain = requested/actual quantisation ratio** (`LC:…/ipa_base.cpp:1846-1855`) | Fixed in steady phases; **varies in the response sweep** (§2.3) |
| `rpi.alsc` (n_iter 100, luminance_strength 0.5, 2 colour temperatures) | 16×12 bilinear per-channel gains = colour tables(colour temperature) × λ × luminance^strength (Guide §5.10, PDF p.38-42) | Stock: λ adapts, clamped to ±5% (`lambda_bound` default 0.05, `LC:…/alsc.cpp:187, 689-740`), with an IIR filter | `n_iter 0`: λ ≡ 1 (`:303-304, 758-785`); tables fixed by the colour temperature from the gains. Start-up transient of 10 frames (`startup_frames`, `:151, 384-418`) |
| `rpi.contrast` (ce_enable 1, 33-knot curve) | Gamma PWL, histogram stretch (CE), then brightness/contrast | Stock: CE from the Y histogram (`LC:…/contrast.cpp:99-141, 160-191`) | `ce_enable 0`, Brightness 0, Contrast 1 (`src/hyperion-cam.cpp:4372-4373`) |
| `rpi.ccm` (5 stock matrices by colour temperature) | 3×3, interpolated by colour temperature, × saturation (× lux PWL if present), clamped to [−8, 8), sent to the hardware at 1/1000 precision (`LC:…/ccm.cpp:114-194`; `LC:…/vc4.cpp:387-401`) | Only through the colour temperature (several matrices) or a lux PWL | One TV-fitted matrix, saturation 1 |
| `rpi.sharpen` | Sharpening | Mode factor and user strength (`LC:…/sharpen.cpp:52-85`) | `Sharpness 0` → strength 0 (`src/hyperion-cam.cpp:4379`; Guide §5.13.3, PDF p.46) |
| `rpi.af` | Lens position | In the auto modes only | Manual lens position (`src/hyperion-cam.cpp:357-376`) |
| `rpi.hdr` | Multi-exposure, only when `HdrMode` ≠ Off; it then also disables CE (`LC:…/ipa_base.cpp:1440-1470`) | n/a | Off (never set) |
| `rpi.sync` | Multi-camera sync | n/a | Inactive |

### 4.2 Specific verifications (a)–(g)

**(a) ALSC with `n_iter 0`. CONFIRMED.**
- **Adaptation stops.** With zero iterations the λ arrays keep their reset value of 1.0 (`LC:…/alsc.cpp:303-304, 758-785`). The Guide says so explicitly: "The value zero can be useful as a way of turning off the adaptive algorithm completely (only the pre-measured calibrated tables will be applied)" (Guide §6 ALSC tuning notes, PDF p.64).
- **Tables are still interpolated by colour temperature** (`getCalTable`/`resampleCalTable`, `:459-540`; colour temperature via `getCt`, `:328-339`). The colour temperature comes from `awb.status`. With manual gains, AWB sets it from the inverse colour-temperature curves (`LC:…/awb.cpp:218-228`). For `-b 2.632,1.626` that is about 6400 K, so the 5000 K tables are used.
- **`luminance_strength` still applies, statically** (`:787-814`; Guide p.64: "colour shading correction is always applied fully").
- **Two caveats.**
  - (i) **The tv-raw calibration ran with *adaptive* ALSC.** fitTvMatrix's per-channel ISP gain `a` (`src/tuning.cpp:402-451`) includes λ ∈ [0.95, 1.05] that will be 1.0 in production.
  - (ii) **`normalise()` and `reaverage()` in `alsc.cpp:742-756` are no-ops.** The `for_each` lambdas take each value by copy. This is an upstream libcamera bug. It is harmless with `n_iter 0` (λ ≡ 1) but explains why stock λ can wander within the ±5% bound.

**(b) `ce_enable 0`. CONFIRMED.**
- **CE is fully disabled.** With `ceEnable_` false the stretch is skipped (`LC:…/contrast.cpp:170-177`). The Guide: "If set to zero, the gamma curve is used exactly as specified (subject to user brightness/contrast settings)" (§5.11.5, PDF p.44), and "Setting ce_enable to zero completely disables any runtime adaptation of the gamma curve" (PDF p.64).
- **Brightness and Contrast still modify the curve:** `(y−32768)·contrast + 32768 + brightness·65536` (`:143-156`; `LC:…/ipa_base.cpp:1258, 1273`; Guide §5.11.3, PDF p.43). We set Brightness 0 and Contrast = `--contrast` (default 1). `--contrast` must stay 1 in every phase. `checkPhaseMetas` compares it (`src/ledCalibFit.cpp:446`).

**(c) Saturation is applied on top of the CCM. CONFIRMED.**
- **Formula.** `CCM_out = Y2RGB·diag(1,s,s)·RGB2Y·CCM` (`LC:…/ccm.cpp:130-145, 173-179`; Guide §5.12.2, PDF p.45).
- **At s = 1** the rounded matrices give a product within **8·10⁻⁴** of identity (computed), below the 1/1000 hardware precision. So `--saturation 1` applies the matrix as written.
- **What "required" really means.** Saturation must be *identical* everywhere, and 1 is what the safety-desaturation margin was computed for. The libcamera default is also 1.0 (`LC:…/ccm.cpp:33`). Production currently runs 0.7383 with the *old* LUT, which is consistent with how that LUT was made.
- **tv-raw itself may run at any saturation.** It reads the applied CCM from metadata, which already includes saturation.

**(d) Anything still adaptive with manual exposure, gain and colour gains?**
- **Lux** is computed every frame but unused.
- **Digital gain** carries the exposure-quantisation ratio (`LC:…/ipa_base.cpp:1846-1855`). It is constant at a fixed shutter and varies in the response sweep (§2.3). It is ≥1 in AGC (`LC:…/agc_channel.cpp:943-946`), but the quantisation ratio is applied *after* that clamp, so it can fall slightly below 1.
- **Denoise and GEQ** depend on the gain only, so they are fixed.
- **AWB IIR** is bypassed by manual gains (`LC:…/awb.cpp:209-221`).
- **ALSC IIR** converges within the start-up frames.
- **Minimum gain.** A requested gain below the sensor minimum is clamped: `limitGain`, `LC:…/agc_channel.cpp:568-577, 987-1003`; min gain = 1024/(1024−112) = **1.1228** (`KD:68-71`, `LC:src/ipa/rpi/cam_helper/cam_helper_imx708.cpp:96-104`). So `-g 1.0` actually runs at 1.123.

**(e) `LIBCAMERA_RPI_TUNING_FILE`. CONFIRMED.**
- **When it is read.** The variable is built from the IPA name and read in `ipaConfigurationFile()` (`LC:src/libcamera/ipa_proxy.cpp:39-48`). That runs from `CameraData::loadIPA()` (`LC:…/pipeline_base.cpp:1164-1195`) during camera registration in `CameraManager::start()`.
- **Our order is correct.** We call `setenv` at `src/hyperion-cam.cpp:3402`, before `cm->start()` at `:3761`. The variable is documented in `LC:Documentation/runtime_configuration.rst:141-144`.

**(f) Gamma knots. CONFIRMED.**
- **Positions.** vc4 writes 32 points at x = i·1024 (i<16), (i−16)·2048+16384 (i<24) and (i−24)·4096+32768, plus a forced (65535, 65535) (`LC:src/ipa/rpi/vc4/vc4.cpp:418-438`; `numGammaPoints=33`, `LC:…/controller.cpp:42`). These are exactly `src/tuning.cpp:149-156`.
- **Values on the knots.** vc4 *evaluates* the curve at the knots, and the Guide says any PWL is allowed (PDF p.64). Our curve has its breakpoints on the knots, so it is reproduced exactly. Pi5/PiSP uses 64 points (`controller.cpp:58`) and would resample.
- **Linear toe.** The realised curve is linear for codes 0–38.5 (§2.3).

**(g) Clipping points along the path:**
1. **Sensor ADC** (1023).
2. **Pre-CCM gain stage** (WB × DG × LS gain): strongly indicated (§3.3).
3. **Negative CCM outputs → 0.** Empirically confirmed: 867 of 3972 bright (max ≥64) TV patches read ≤0.6 in some channel under the stock CCM (my count from `~/calib/tv_sweep.csv`; the doc's denominator 4097 is the whole sweep, `doc/led-lut-calibration.md:259-261`). The hardware documentation is not public: **mechanism UNVERIFIED**.
4. **Gamma input/output** clamped to 16 bits (`LC:…/vc4.cpp:428`), and the brightness/contrast clamp (`contrast.cpp:151-153`).
5. **8-bit output.**
6. **Our shader** clamps to [0,1] before the LUT (`src/gpuNv12Transform.cpp:726`).
7. **HyperHDR** `round_to_0_255` (`HH:sources/led-drivers/LedDevice.cpp:590-594`).

### 4.3 Recommended tuning and capture changes for a better instrument

1. **`ColourGains = 1,1`** (§3.3). This gives the largest gain in headroom and makes "ISP clip = sensor clip" true.
2. **Analytic linearisation** from `tuning.json` (§2.3). Keep the measured curve as a check.
3. **Record `DigitalGain`** in every CSV row, and snap response exposures to even lines (§2.3).
4. **Run `tv-raw` under an interim frozen tuning** (`--make-tuning interim.json` without `--from-calib`): CE off, ALSC frozen, stock CCM, `--saturation 1`. The `a` gains and the curve decode are then exactly the production ISP state (§4.2a).
5. **Optional: `luminance_strength 0`.** Luminance-shading gains above 1 at the frame edges raise the pre-CCM clip there; the corner luminance table goes up to 2.644 at full strength. Our `shading.csv` already corrects the TV side. Only do this if the LED ROI is near the frame edge; it costs SNR at the edges.
6. **Measure the true black pedestal** (lens cap, RAW, production gain). If it is not 64.0, set `black_level` in the frozen file. ISP and `SensorBlackLevels` both follow it.
7. **Pin the sensor mode** with `CameraConfiguration::sensorConfig` instead of probing RAW sizes (`LC:…/pipeline_base.cpp:211-214`). This is optional; the probe is already correct.
8. **Keep** `Sharpness 0`, Brightness 0, Contrast 1, manual focus and `HdrMode` unset, as now.

---

## 5. Q4: How the calibration is applied end to end

### 5.1 Our shader (`src/gpuNv12Transform.cpp`)

**Colour conversion: limited-range BT.601 both ways. CONFIRMED.**
- **Decode.** `yuv2rgb`: `yl=(y−16/255)·255/219`, `u,v=(c−128/255)·255/224`, R = yl+1.402v, G = yl−0.344136u−0.714136v, B = yl+1.772u (`:704-709`).
- **Encode.** Y = dot(rgb, .299/.587/.114)·219/255 + 16/255; U and V use the matching BT.601 rows ·224/255 + 128/255 (`:729-733`).
- **Matches the camera.** We request `ColorSpace::Smpte170m` for NV12 (`src/hyperion-cam.cpp:3993-3996`). The RPi pipeline maps it to `V4L2_COLORSPACE_SMPTE170M`; only the colourspace ID reaches the driver (`LC:…/pipeline_base.cpp:97-108`).
- **Calibration stream differs.** The RGB calibration stream asks for Rec709 (`:3999`), which becomes an RGB/full variant (`:128-131`). Whether the vc4 RGB888 output equals decoded NV12 for the same scene is **UNVERIFIED**; the ISP firmware is closed. Recommendation 7 removes the question.

**LUT sampling. CONFIRMED.**
- **Coordinate.** `coord = rgb·(N−1)/N + 0.5/N` (`:183`) maps 0 and 1 to the first and last texel centres, as Cube §7.1 requires ("lower and upper bound … shall map to the first and last sample point").
- **Filtering.** `GL_LINEAR`, `CLAMP_TO_EDGE` (`:836-842`).
- **Orientation.** The writer loops b → g → r with red innermost (`src/ledCalibFit.cpp:1255-1261`), matching Cube §7.2 ("Red changing most rapidly … C index r + N·g + N·N·b"). The upload uses x = r (`:401-420`). The loader rejects non-[0,1] domains (`src/hyperion-cam.cpp:2128-2136`). Comment lines (`#`) are allowed by Cube §5.8.

**Precision chain.**
- LUT nodes are stored as **RGBA8** (`:416, 865`): ±0.5 code at each node. The precision of v3d's filter weights is UNVERIFIED.
- The output is 8-bit NV12 limited range: 220 Y levels, 4:2:0 chroma point-sampled per 2×2 block.
- The NV12 round trip itself is measured in §5.2.
- A 16-bit-float LUT texture would remove the node quantisation; its cost on v3d is unmeasured.

**Shading.**
- `rgb *= g` is applied to the *encoded* RGB before the clamp and the LUT (`:725-728`). The tv/gamma phases scale the TV-ROI reading by the same gain, so calibration and runtime agree.
- It is exact in linear light only where the curve is a pure power law, so not below code about 38 (§2.3).

### 5.2 HyperHDR's flatbuffer NV12 decode: matrix and range

- **Path.**
  - The NV12 branch loads the LUT with pixel format YUYV (`HH:sources/flatbuffers/server/FlatBuffersServer.cpp:390-427`).
  - `LutLoader` picks "Index 2 for YUV" when tone mapping is off (`HH:sources/utils/LutLoader.cpp:82-95`).
  - Files are searched in this order: `~/.hyperhdr/flat_lut_lin_tables.3d` (absent here), then `~/.hyperhdr/lut_lin_tables.3d` (present, 150,994,944 B = 3 tables) (`FlatBuffersServer.cpp:282-309`).
  - Each pixel is looked up at index `Y | U<<8 | V<<16`, three bytes, with nearest-neighbour chroma (`HH:include/utils/VectorizedDecoders.h:183-211`; `HH:sources/utils/FrameDecoder.cpp:150-178`).
- **What table 2 contains.**
  - **Source.** The default LUT uses `coef = BT601`, `range = LIMITED` (`HH:sources/lut-calibrator/LutCalibrator.cpp:1802-1835`; enums in `HH:include/infinite-color-engine/YuvConverter.h:44-52`). Phase 2 is `toRgb(range, coef)` (`LutCalibrator.cpp:1538-1541`).
  - **Measured.** I least-squares fitted the installed file over 4440 unclipped (Y,U,V) samples, with max |residual| 0.50:

    | Output | Y term | U term | V term |
    |---|---|---|---|
    | R | 1.16398·(Y−16) | — | +1.59575·(V−128) |
    | G | 1.16442·(Y−16) | −0.39175·(U−128) | −0.81300·(V−128) |
    | B | 1.16437·(Y−16) | +2.01718·(U−128) | — |

    The textbook BT.601 limited values are 1.16438 / 1.59603 / −0.39176 / −0.81297 / 2.01723. BT.709 limited would be 1.7927 / −0.2133 / −0.5329 / 2.1124. **It matches BT.601 limited.**
  - **Grays.** Y 16 → 0, Y 125 (our encoding of 127) → 127, Y 235 → 255.
- **Round trip through our encoder** (whole 256³ RGB cube, `rint` quantisation, the installed table):

  | Error | Share of colours |
  |---|---|
  | exact | 15.9% |
  | max channel error 1 | 83.8% |
  | max channel error 2 | 0.35% |
  | mean absolute error | 0.40 code |

  Grays keep 220 of 256 levels; for example inputs 3 and 4 both come out as 3. Full red (255,0,0) returns as (254,0,0).
- **What a mismatch would have looked like** (hypothetical; it does not happen here). If HyperHDR decoded with *full-range BT.709*:

  | Input | Would come out as |
  |---|---|
  | black | 16 (LEDs never off) |
  | gray 32 | 43 |
  | white | 235 |
  | red | (255, 36, 10) |
  | green | (0, 203, 8) |
  | blue | (13, 28, 249) |

  Keep this table as the reference if the LUT file ever changes.
- **Guards.** The decode is fixed by a file, not by code. Any of the following would silently change it:
  - creating `flat_lut_lin_tables.3d`;
  - a user LUT file name in the flatbuffer settings (`FlatBuffersServer.cpp:117-123, 295-307`);
  - re-running HyperHDR's LUT calibrator;
  - a re-download of the default LUT.

  Re-verify with the probe in Appendix A.

### 5.3 Does HyperHDR tone-map or HDR-process flatbuffer input?

- **Only if `flatbufServer.hdrToneMapping` is true.** It is false in our DB.
- **Code path.**
  - `_flatbufferToneMappingMode` is read from that setting (`FlatBuffersServer.cpp:137`).
  - Toggling the global HDR component can enable tone mapping only when the mode is on (`:84-104`, `_hdrToneMappingEnabled = listen ? _flatbufferToneMappingMode : 0`).
  - With it on, NV12 would use table 1 ("HDR YUV") and RGB frames would get table 0 (`LutLoader.cpp:82-95`; `FlatBuffersServer.cpp:380-385`).
- **Other settings.** `automaticToneMapping.enable` is false. The flatbuffer path passes `nullptr` for automatic tone mapping (`:421-423`). `quarterOfFrameMode` is false (`:154`).

### 5.4 How an image area becomes one LED colour

**Averaging rule** (`imageToLedMappingType "advanced"` → multicolour):
- **Mean in linear light.** For each LED area, `mean(srgbNonlinearToLinear(pixel))` (`HH:sources/base/ImageColorAveraging.cpp:216-235`). The conversion uses an exact 8-bit → 16-bit sRGB EOTF table (`HH:include/infinite-color-engine/InfiniteProcessing.h:228-240`).
- **Large areas would be subsampled.** Areas over 1600 px are sparsely sampled, every second pixel and row (`:107-116`). At 768×432 our 223 LED areas have 385–418 px each (computed from the DB `leds` geometry), so every pixel is used.
- **The result is linear float.** It is passed on as such (`HH:sources/base/HyperHdrInstance.cpp:485-520`).

**Consequence.**
- **The output is sRGB-*encoded*.** The pipeline re-encodes with the sRGB OETF before the user gamma (`InfiniteProcessing.cpp:185-186`).
- **With `gamma 1` and linear PWM, LED light ∝ OETF(mean linear).** That is a 2.4-power mean of the pixel *drives*, which is always ≥ their arithmetic mean.
- **Examples.** An area 10% white / 90% black gets drive 89 (35% PWM) where the light average is 10%. 50/50 gets 188 instead of 128.
- **Why calibration cannot see it.** Uniform-patch calibration and the full-screen eye check only exercise uniform areas, so this mixed-area error never shows up in them.
- **Option (needs re-calibration).** Set HyperHDR gamma to about 2.2–2.4. Then `OETF(x)^γ ≈ x`, so HyperHDR's linear averaging and its linear temperature become physically consistent with linear-PWM LEDs; the residual is the toe below linear ≈0.3%. The WS2812 PWM linearity is an assumption (datasheet not accessed, **UNVERIFIED**); the new RAW sweep can check it: `curve_r` should be close to linear in drive when R temperature = 1 and gamma = 1.

### 5.5 Colour processing after averaging, in order, with our settings

Source: `HH:sources/infinite-color-engine/InfiniteProcessing.cpp:174-231`. It runs per LED unless processing is disabled. **Processing is disabled when the visible component is `COMP_COLOR`** (`HH:sources/base/HyperHdrInstance.cpp:651-655`), so never eye-match with HyperHDR's colour picker. Our solid frames are *images* (§5.7), so they get full processing.

| # | Stage (v22) | Domain | Our value (DB) | Effect | Interacts with our LUT? |
|---|---|---|---|---|---|
| 1 | `applyTemperature` | linear | custom R 1 / G 0.38 / B 0.22 | out = OETF(t·EOTF(v)); level-dependent in drive space (§2.2) | Absorbed by the LED sweep. **Must not change.** Breaks the "equal drive = white" assumption |
| 2 | `calibrateColorInColorspace` (primaries / 8-colour LUT) | linear | identity → `CalibrationMode::None` (`:320-331, 355-369`) | none | Keep identity, or re-measure (it is not separable) |
| 3 | `applyScaleOutput` | linear | 1 → off (`:591-614`) | — | Keep 1 |
| 4 | sRGB OETF | → encoded | always | — | Absorbed |
| 5 | `applyUserGamma` | encoded | 1 → off (default is 1.5!) (`:464-505`) | — | Absorbed; must not change. See §5.4 for gamma ≈2.4 |
| 6 | HSV brightness/saturation (`luminanceGain`, `saturationGain`) | encoded | 1/1 → off (`:549-588`) | — | Not separable. **Must stay 1** |
| 7 | `applyMinimalBacklight` (`backlightThreshold`) | encoded | 0 → off (default 0.0039) (`:233-283`) | — | Would put a floor under the darks. Keep 0 |
| 8 | `applyPowerLimit` | all LEDs | 1 → off (`:617-651`) | — | Content-dependent global scale. Keep 1 |
| 9 | colour order | — | grb (ws2812spi) | channel swap | none |
| 10 | Smoothing (after processing, encoded domain; `HH:…/CoreInfiniteEngine.cpp:65-69`) | encoded | `enable: false` (current DB) | none when off; direct output (`HH:…/InfiniteSmoothing.cpp:207-223`) | Temporal only. Off for the sweeps. That every interpolator converges exactly to target is **UNVERIFIED** |
| 11 | Quantise to 8 bits | — | `round_to_0_255(v·255)` (`HH:…/LedDevice.cpp:581-599`) | ±0.5 code | — |

- **Interpolation error.** HyperHDR's 512-entry interpolated sRGB tables (`InfiniteProcessing.h:167, 202-226`) add at most 1 code against an exact computation for t = 0.38 and 0.22, and none for t = 1 (computed).
- **No "brightness compensation" stage exists in v22.** A source search found nothing; this replaces the v20-era option.

### 5.6 Settings that must stay identical between calibration and use

| Category | Settings |
|---|---|
| HyperHDR | temperature (custom 1/0.38/0.22), gamma 1, luminance/saturation gain 1, backlightThreshold 0, powerLimit 1, scaleOutput 1, primaries identity, `flatbufServer.hdrToneMapping=false`, the decode LUT file (table 2 BT.601 limited, no `flat_lut_lin_tables.3d`), image-to-LED mapping "advanced", LED device/order |
| Camera | tuning file, `-b`, `-s`, `-g`, `--saturation`, `--contrast`, focus, sensor mode |
| Not colour-critical | smoothing (temporal only), black-border detection (geometry) |

### 5.7 Do the synthetic solid frames take the production path?

- **Calibration path.** `ledCalibSendSolid` sends a 64×36 **RGB RawImage** (`src/hyperion-cam.cpp:830-853`; `src/FlatBufferConnection.cpp:456-472`). HyperHDR takes the RGB branch without any LUT (`FlatBuffersServer.cpp:357-387`).
- **Production path.** Production sends NV12 (`src/FlatBufferConnection.cpp:490-506`), which goes through table 2 (`:390-427`).
- **Shared steps.** From `SignalSetGlobalImage` onward both paths are identical: same averaging, same processing, same component (flatbuffer, not `COMP_COLOR`).
- **Differences.**
  1. **The NV12 round trip:** ±1 code typically, 2 at most (§5.2). Near black that is 20–100% of a drive value, although HyperHDR's 0.22/0.38 toe compresses G/B output anyway.
  2. **Frame size and pixel sets:** irrelevant for uniform frames.
- **Cheap fix.** Send the calibration solid frames as NV12 encoded with the shader's exact formula, so the LED sweep measures the production quantisation as well.

---

## 6. Claims in our code and docs: CONFIRMED / REFUTED / UNVERIFIED

| # | Claim (where) | Verdict | Evidence |
|---|---|---|---|
| 1 | Stock ISP is scene-adaptive (CE, ALSC iterations) (`doc/led-lut-calibration.md:256-258`) | CONFIRMED | `imx708.json`: `ce_enable 1`, `n_iter 100`; `LC:…/contrast.cpp:170-177`; `LC:…/alsc.cpp:758-785`, λ bound ±5% |
| 2 | `ce_enable 0` + ALSC `n_iter 0` freeze the ISP (`:268-273`) | CONFIRMED, with caveats | §4.2(a),(b),(d): digital gain varies with exposure; start-up transient |
| 3 | Stock CCM clips saturated TV colours to 0: 867 of 4097 (`:259-261`) | CONFIRMED (count) | 867 of 3972 bright patches (4097 = whole sweep). Mechanism (negative → 0 in hardware) UNVERIFIED |
| 4 | Tone curve "power 2.2 on the vc4 knots" (`src/tuning.cpp:147-156`) | CONFIRMED | `LC:src/ipa/rpi/vc4/vc4.cpp:423-433` |
| 5 | Power curve "makes the shading map exact at every level" (`doc/led-lut-calibration.md:271-272`; `doc/calibration-guide.md` Step 3) | REFUTED (below code ≈38) | PWL first segment linear: encoded gain g is g in linear at code 20 vs ≈g^2.2 at codes ≥60 (§2.3) |
| 6 | `LIBCAMERA_RPI_TUNING_FILE` must be set before CameraManager starts (CLAUDE.md) | CONFIRMED | `LC:src/libcamera/ipa_proxy.cpp:39-48`; `LC:…/pipeline_base.cpp:1164-1195`; `src/hyperion-cam.cpp:3402, 3761` |
| 7 | Saturation is applied on top of the CCM, so use `--saturation 1` | CONFIRMED | `LC:…/ccm.cpp:130-145, 173-179`; Guide §5.12.2. At s = 1 the deviation is <8·10⁻⁴ |
| 8 | "libcamera applies the single-entry matrix as written (metadata)" (`doc/led-lut-calibration.md:280-281`) | CONFIRMED (≈) | Clamp [−8,8), 1/1000 precision (`LC:…/vc4.cpp:387-401`) |
| 9 | ISP emits limited-range SMPTE170M NV12; shader decode limited BT.601 (`doc/led-lut-calibration.md:400-405`) | CONFIRMED (request path); driver behaviour UNVERIFIED | `src/hyperion-cam.cpp:3993-3996`; `LC:…/pipeline_base.cpp:97-108` |
| 10 | HyperHDR decodes our NV12 compatibly | CONFIRMED | Installed table 2 = BT.601 limited (fit); `HH:…/LutCalibrator.cpp:1802-1835` |
| 11 | "HyperHDR is linear (gamma 1, temperature …)" (`doc/led-lut-calibration.md:264`) | **REFUTED** for v22 | `HH:…/InfiniteProcessing.cpp:174-191`: temperature in linear light, then sRGB OETF. Equal-input G/R ratio 0.38 → 0.65 with level |
| 12 | Dark fallback "equal R=G=B drive = HyperHDR's eye-matched white by construction" (`src/ledCalibFit.cpp:141-142, 1227`; `doc/led-lut-calibration.md:194-195`; CLAUDE.md) | **REFUTED** | Same as row 11. True only at the 127 anchor |
| 13 | RAW LED clip "is the ISP's WB gain, not the sensor" (`doc/led-lut-calibration.md:43`) | CONFIRMED (strongly indicated) | `LC:…/vc4.cpp:371-380`; arithmetic 1/2.632 = 38% |
| 14 | The new clip checks catch ISP clipping in tv-raw and the LED fit | **REFUTED** | `src/tuning.cpp:327-334, 459-461`; `src/ledCalibFit.cpp:794-797` miss pre-CCM clipping (§3.3) |
| 15 | RAW stream is the same exposure as the ISP output | CONFIRMED | `LC:src/libcamera/pipeline/rpi/vc4/vc4.cpp:843-855, 911-925, 965-1007` |
| 16 | SBGGR10_CSI2P unpack (`src/ledCalib.cpp:1264-1273`) | CONFIRMED | V4L2 `pixfmt-srggb10p.rst:23-27, 44-51` |
| 17 | `SensorBlackLevels` order and scale (`src/hyperion-cam.cpp:2541-2549`) | CONFIRMED | `LC:src/libcamera/control_ids_core.yaml:627-636` |
| 18 | Black levels "from SensorBlackLevels" means a measured black | REFUTED (it is the tuning constant) | `LC:…/black_level.cpp:56-67`; `LC:…/ipa_base.cpp:1625-1631` |
| 19 | ISP→RAW ROI mapping via ScalerCrop + mode crop + binning | CONFIRMED | `LC:…/control_ids_core.yaml:685-699`; `property_ids_core.yaml:667-680`; `KD:722-742` |
| 20 | RAW size probe reproduces the production sensor mode | CONFIRMED | `LC:…/pipeline_base.cpp:211-225` |
| 21 | Response curve is "drift-proof because referenced to RAW" (`src/cameraResponse.cpp:242-244`) | PARTLY REFUTED | Drift-proof for source drift, but not for ISP digital-gain quantisation (`LC:…/ipa_base.cpp:1846-1855`) |
| 22 | Self-test: response within 1.6% (codes 16..250); legacy p95 30 vs 0.96 (`doc/led-lut-calibration.md:197-205`) | UNVERIFIED (not re-run) | Also: the self-test uses the stock curve, so the frozen curve's linear toe up to code 38 is not exercised |
| 23 | "No `--tuning` needed for tv-raw (RAW doesn't depend on it)" (`doc/calibration-guide.md:86-87`) | **REFUTED in effect** | The CCM fit uses tv-raw *ISP* readings for per-channel gains (`src/tuning.cpp:402-451`), affected by CE and adaptive ALSC |
| 24 | "lower `--gain`" to fix sensor clip (`doc/calibration-guide.md:87`) | REFUTED at default gain | Minimum gain 1.123; `-g 1` is already clamped (`KD:68-71`; `LC:…/agc_channel.cpp:568-577`) |
| 25 | Trilinear error shrinks about quadratically with grid size (`doc/led-lut-calibration.md:420`) | CONFIRMED for smooth regions | Standard interpolation theory; about linear at kinks. Kasson tables UNVERIFIED |
| 26 | `.cube`: red fastest, domain [0,1] (`doc/led-lut-calibration.md:207-208`) | CONFIRMED | Cube spec §7.1, §7.2 |
| 27 | "LUT(avg) ≠ avg(LUT) at hard edges — acceptable" (`doc/led-lut-calibration.md:426-427`) | CONFIRMED, but larger than implied | HyperHDR averages in linear light and outputs encoded drive: 10% white area → 35% PWM (§5.4) |
| 28 | Smoothing doesn't change the steady-state colour (`doc/led-lut-calibration.md:72-73`) | PARTLY CONFIRMED | Smoothing runs after processing (`CoreInfiniteEngine.cpp:65-69`). Exact convergence per interpolator UNVERIFIED |
| 29 | Wall/LED-ROI shading cancels in the white anchor (`doc/led-lut-calibration.md:351-352`) | CONFIRMED | ALSC is a fixed per-channel linear gain per position once frozen; `corr` uses the same ROIs |
| 30 | `imx708.json` stock, not locally edited | CONFIRMED | md5 = dpkg record = upstream ≥ `v0.7.0+rpt20260205`; `_bak` = upstream `v0.6.0+rpt20251202` |
| 31 | TV gray stable within ±5% as the camera sees it (`doc/led-lut-calibration.md:264-265`) | UNVERIFIED | Re-check in linear light with the frozen tuning |
| 32 | TV gray 127 reads about 130, LED drive 127 reads R 204 (`doc/led-lut-calibration.md:250-252`) | CONFIRMED | `~/calib`: 128.4/133.6/132.4 and 203.8/144.5/191.6 |

---

## 7. Recommendations, ranked by expected effect on the LUT result

| # | Change | Why | Expected effect | A/B on the real TV |
|---|---|---|---|---|
| 1 | **Fix the dark floor** (`src/ledCalibFit.cpp:1227-1274`): (a) use the anchor's light direction `e = k·e(anchorDrive)` with per-channel inversion instead of equal drive; (b) gate on `max(R,G,B)` instead of luma | §2.2: equal drive is red/orange in the toe; the luma gate neutralises dark blues and reds | Neutral near-black glow; dark blues stay blue | Eye check: gray slider 0–30 and colour mode with B 20–80 at low brightness. Fits C vs C-with-fix vs `--calib-dark-floor off` |
| 2 | **`-b 1,1` everywhere**, re-run `tv-raw`, regenerate the tuning; add a pre-CCM clip check (`wb_c·dg·raw_c/FS > 0.95` → exclude or warn) in `fitTvMatrix` and `fitRawToLin` | §3.3: ISP R clips at 38% of the sensor range before the CCM | Clean RAW→ISP map (lower `raw_map_resid`); TV highlights and saturated reds not clipped; makes "peak → 95%" achievable | `--ledcalib gamma --gamma-raw` R ramp with `-b 2.632,1.626` vs `-b 1,1`: with 1,1, ISP R should track RAW up to the sensor clip. Compare `raw_map_resid` |
| 3 | **Run `tv-raw` under an interim frozen tuning** (`--make-tuning interim.json`, no `--from-calib`, `--saturation 1`) | §4.2(a): CE and adaptive ALSC (±5%) bias the `a` gains | The TV-fitted CCM really maps TV white → neutral and primaries → axes in production | In the `tv` sweep, TV white channel ratio in linear light ≤1%; no channel ≤0.5% of max on bright patches |
| 4 | **Exact linearisation**: analytic inverse of `tuning.json`'s PWL (the toe up to code 38 is linear); keep the measured response as a check (report max deviation in `fit_report.json`); record `DigitalGain` per row and snap response exposures to even lines | §2.3 | Removes prior bias and dg bias at the dark end | Fit B with measured vs analytic curve; compare `gray_axis` drives at TV levels 3–64 and eye-check darks |
| 5 | **`--lut-size 65`** (optionally trilinear-aware node optimisation near gray) | §2.4: 75% off-axis weight on the gray axis; free on v3d | Gray-axis dip about 1.6% → about 0.4–0.8% | `--latency-stats` (cost should be ≈0); eye check of the near-white gray slider 200–255 |
| 6 | **Send calibration solid frames as NV12** (same encoder as the shader) | §5.7: the sweep should include the production round trip (±1–2 codes) | The LED model matches production quantisation exactly | Re-run `gamma_leds` with RGB vs NV12 frames; compare the low end |
| 7 | **TV-side measurement in NV12** (decode as the shader does) instead of RGB888 | §5.1: the RGB888/NV12 equivalence of the vc4 ISP is unverified | The LUT input domain is identical by construction | One TV sweep both ways; per-patch difference histogram |
| 8 | **HyperHDR gamma ≈2.2–2.4** (then redo the white eye-match and all LED phases) | §5.4: HyperHDR averages in linear light and outputs encoded drive | Correct LED brightness for mixed-content areas. The temperature becomes nearly level-independent *above* the sRGB toe: with γ 2.2, equal-input G/R is 0.365 at 127 and 0.388 at 255. Inside the toe it is still not: G/R is 0.119 at codes ≤ 22 (B ≤ 29). So item 1's fix is still needed | Patterns where each LED area is half black / half white vs a uniform gray of equal *linear* light; LEDs should match. Normal films for general look |
| 9 | **RAW-based white anchor** (RAW columns in `tv` + fitted `M_TV`) | §3.4: removes tone-curve dependence of `corr` | More robust anchor | Compare `corr` from the ISP path vs the RAW path; the eye check at 127 should be unchanged |
| 10 | **Instrument checks**: lens-cap RAW pedestal at production gain; RAW/exposure linearity over all response steps; interleaved reference whites in the LED sweep (thermal/supply drift); 2–3 LED ROIs once | §2.5, §2.6, §3.2 | Quantifies residual error sources before the long sweeps | Report pedestal−64, linearity slope deviation, drift %, `corr` spread across ROIs |
| 11 | **Exposure bracketing** (RAW, 4×) for the low LED drives and TV black | §3.4: the dark-end curve is regularised, not measured | Measured dark end; `--calib-curve-floor` becomes unnecessary | Fit with vs without the bracketed low end; eye check 0–30 |
| 12 | **HyperHDR guards**: a script (Appendix A) that verifies table 2 is BT.601 limited and that `flat_lut_lin_tables.3d` / a user LUT file are absent; `hdrToneMapping` false; never eye-match with the HyperHDR colour picker (`COMP_COLOR` disables processing) | §5.2, §5.3, §5.5 | Prevents silent breakage after HyperHDR updates | Run the probe after every HyperHDR update |
| 13 | **Optional**: extra eye anchors (R, G, B) to fit a 3×3 metameric correction instead of the diagonal | §2.1–2.2 | Lower hue error for saturated colours | Colour mode of the eye check at full saturation, 6 hues |

**Other notes:**
- Shortening the shutter, rather than gain (which is already at its 1.123 minimum), is the remaining lever if the TV peak window clips the sensor. Measure its latency effect with `--latency-stats`.
- `--calib-response off` (fit "A") is not a fair "frozen tuning alone" baseline. The legacy pure power law is badly wrong below code 53 on the frozen PWL curve (§2.3), so expect fit A to look worst in the darks by construction.

---

## 8. Findings that mean the new, untested code has a bug or a wrong assumption

1. **Dark floor uses equal drive as eye white**: `src/ledCalibFit.cpp:1227-1246, 1263-1274` (§2.2). Wrong with HyperHDR v22.
2. **Dark-floor luma gate catches saturated dark blues and reds**: `src/ledCalibFit.cpp:1266`.
3. **Pre-CCM (WB-stage) clipping not detected**: `src/tuning.cpp:327-334, 459-461`; `src/ledCalibFit.cpp:794-797` (§3.3).
4. **Response fit ignores ISP digital-gain quantisation compensation**, and `DigitalGain` is not recorded: `src/cameraResponse.cpp:110-146`; `src/ledCalib.cpp:2039` (§2.3). Small.
5. **tv-raw under stock tuning feeds CE- and adaptive-ALSC-affected ISP readings** into the CCM's per-channel gain fit, decoded with the static curve: `src/tuning.cpp:271-287, 402-451`; `doc/calibration-guide.md:86-87`.
6. **Pure-power assumptions on the frozen PWL**: the response prior `src/cameraResponse.cpp:94`; the legacy linearizer `src/ledCalibFit.cpp:56-63` when used with frozen data; the shading exactness claim in the docs. The PWL is linear below code 38.5.
7. **Guide advice "lower `--gain`"** cannot work at the default gain (minimum 1.123).

---

## 9. Sources

**Upstream code**

- **libcamera (RPi fork)**, tag `v0.7.1+rpt20260609`, commit `06c385619acb10bbfb33f52f3abeb8f8c095f42b`:
  - `src/libcamera/ipa_proxy.cpp`: https://github.com/raspberrypi/libcamera/blob/06c385619acb10bbfb33f52f3abeb8f8c095f42b/src/libcamera/ipa_proxy.cpp#L33-L48
  - `src/libcamera/pipeline/rpi/common/pipeline_base.cpp`: …/pipeline_base.cpp#L97-L158 (colour spaces), #L205-L225 (sensor size), #L1164-L1195 (loadIPA)
  - `src/libcamera/pipeline/rpi/vc4/vc4.cpp`: #L843-L855, #L905-L925, #L965-L1007
  - `src/ipa/rpi/vc4/vc4.cpp`: #L356-L438 (AWB, DG, CCM, black level, gamma), #L454-L466 (denoise), #L521-L555 (LS)
  - `src/ipa/rpi/common/ipa_base.cpp`: #L75-L102, #L1250-L1311, #L1440-L1470, #L1595-L1631, #L1835-L1855
  - `src/ipa/rpi/controller/controller.cpp`: #L36-L60, #L146-L170
  - `src/ipa/rpi/controller/rpi/contrast.cpp`: #L41-L191
  - `src/ipa/rpi/controller/rpi/ccm.cpp`: #L30-L194
  - `src/ipa/rpi/controller/rpi/alsc.cpp`: #L146-L190, #L206-L230, #L272-L418, #L466-L545, #L689-L862
  - `src/ipa/rpi/controller/rpi/awb.cpp`: #L205-L330
  - `src/ipa/rpi/controller/rpi/agc_channel.cpp`: #L568-L577, #L765-L787, #L880-L1003
  - `src/ipa/rpi/controller/rpi/{black_level,dpc,lux,noise,geq,sdn,sharpen}.cpp`: prepare/process functions as cited
  - `src/ipa/rpi/cam_helper/cam_helper_imx708.cpp`: #L96-L104
  - `src/libcamera/control_ids_core.yaml`: #L627-L636 (SensorBlackLevels), #L685-L699 (ScalerCrop), #L701-L715 (DigitalGain)
  - `src/libcamera/property_ids_core.yaml`: #L667-L692
  - `Documentation/runtime_configuration.rst`: #L141-L144
  - `src/ipa/rpi/vc4/data/imx708.json` at tags `v0.6.0+rpt20251202` (= our `_bak`), `v0.7.0+rpt20260205`, `v0.7.1+rpt20260609` (= installed): https://raw.githubusercontent.com/raspberrypi/libcamera/v0.7.1%2Brpt20260609/src/ipa/rpi/vc4/data/imx708.json
- **HyperHDR**, tag `v22.0.0.0beta1`, commit `e3dfe922d55358406075de9496f7159854f69b21`:
  - `sources/flatbuffers/server/FlatBuffersServer.cpp`: https://github.com/awawa-dev/HyperHDR/blob/e3dfe922d55358406075de9496f7159854f69b21/sources/flatbuffers/server/FlatBuffersServer.cpp#L84-L158 , #L282-L309 , #L349-L428
  - `sources/utils/LutLoader.cpp` #L44-L142; `sources/utils/FrameDecoder.cpp` #L40-L178; `include/utils/VectorizedDecoders.h` #L183-L211
  - `sources/lut-calibrator/LutCalibrator.cpp` #L1510-L1549, #L1802-L1858; `include/infinite-color-engine/YuvConverter.h` #L44-L57; `sources/infinite-color-engine/YuvConverter.cpp` #L34-L46
  - `sources/base/ImageColorAveraging.cpp` #L41-L259; `sources/base/ImageToLedManager.cpp` #L58-L106; `sources/base/HyperHdrInstance.cpp` #L485-L520, #L651-L655, #L691-L761
  - `sources/infinite-color-engine/InfiniteProcessing.cpp` #L103-L651; `include/infinite-color-engine/InfiniteProcessing.h` #L160-L240
  - `sources/infinite-color-engine/CoreInfiniteEngine.cpp` #L65-L69; `sources/infinite-color-engine/InfiniteSmoothing.cpp` #L207-L232; `sources/led-drivers/LedDevice.cpp` #L581-L599
  - `sources/base/schema/schema-color.json` (defaults)
- **Linux kernel**:
  - imx708 driver, `raspberrypi/linux` `rpi-6.12.y` @ `dcc3a57b713fe33c33a9eba7e0704bc87aed6698`: https://github.com/raspberrypi/linux/blob/dcc3a57b713fe33c33a9eba7e0704bc87aed6698/drivers/media/i2c/imx708.c#L68-L71 , #L108-L120 , #L193-L197 , #L673-L768 , #L1500-L1517 , #L1886-L1887 (the running kernel 6.12.47 may differ slightly: UNVERIFIED)
  - V4L2 packed Bayer doc, `torvalds/linux` @ `65e4fd942f72a9dba0e4e257290c4fa1aa756854`: https://github.com/torvalds/linux/blob/65e4fd942f72a9dba0e4e257290c4fa1aa756854/Documentation/userspace-api/media/v4l/pixfmt-srggb10p.rst#L23-L51

**Documents and papers**

- **Raspberry Pi Ltd, *Camera Algorithm and Tuning Guide*** (RP-009070-TC, Release 2, build 20/10/2026): https://datasheets.raspberrypi.com/camera/raspberry-pi-camera-guide.pdf. Sections cited: §5.1–5.6 (PDF p.20-25), §5.10 (p.38-42), §5.11 (p.42-44), §5.12 (p.44-45), §5.13 (p.45-46), and the ALSC/contrast tuning notes in chapter 6 (p.63-64).
- **P. E. Debevec, J. Malik, "Recovering High Dynamic Range Radiance Maps from Photographs"**, SIGGRAPH 1997: https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf. §2.1 (reciprocity, monotonic f), Eq. 3–5, Eq. 4 (hat), the g(Z_mid)=0 gauge, §2.6 (per-channel colour).
- **J. M. Kasson, S. I. Nin, W. Plouffe, J. L. Hafner, "Performing color space conversions with three-dimensional linear interpolation"**, J. Electronic Imaging 4(3):226-250, 1995, doi:10.1117/12.208656. Abstract via https://api.crossref.org/works/10.1117/12.208656. Full text not accessed: UNVERIFIED beyond the abstract.
- **Adobe Systems, *Cube LUT Specification* Version 1.0** (2013): https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf (now 404). Read from the Internet Archive copy https://web.archive.org/web/2020id_/https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf. §5.6–5.8, §7.1, §7.2, §8.
- **G. D. Finlayson, Y. Zhu, "Designing Color Filters that Make Cameras More Colorimetric"**, IEEE TIP 2020, doi:10.1109/TIP.2020.3038523; arXiv:2003.12645 (abstract; Luther condition).
- **Y. Ohno, S. Brown, "Four-Color Matrix Method for Correction of Tristimulus Colorimeters – Part 2"**, Proc. IS&T/SID 6th CIC, 1998: https://www.nist.gov/publications/four-color-matrix-method-correction-tristimulus-colorimeters-part-2-0 (abstract).
- **Not accessed, UNVERIFIED:** Y. Ohno, J. Hardis, Part 1, Proc. IS&T 5th CIC 1997 pp. 301-305, doi:10.2352/CIC.1997.5.1.art00059 (HTTP 403); R. Luther, Z. Tech. Phys. 8 (1927); G. Wyszecki & W. S. Stiles, *Color Science* (1982); WS2812B datasheet.

**Local primary data (this Pi, read-only)**

- `/usr/share/libcamera/ipa/rpi/vc4/imx708.json` (md5 `79f40549866b670fff8cdc37868af537`, matches `/var/lib/dpkg/info/libcamera-ipa*.md5sums`) and `imx708.json_bak` (md5 `91fd7a88e65e57f3269f6e53168e7eb`).
- `~/.hyperhdr/db/hyperhdr.db` settings:
  - `color` channelAdjustment: gamma 1, temperature custom 1/0.38/0.22, luminanceGain 1, saturationGain 1, backlightThreshold 0, powerLimit 1, scaleOutput 1, primaries identity, classic_config false, imageToLedMappingType advanced.
  - `flatbufServer`: hdrToneMapping false, quarterOfFrameMode false.
  - `smoothing`: enable false.
  - `automaticToneMapping`: enable false.
  - `device`: ws2812spi grb.
- `~/.hyperhdr/lut_lin_tables.3d` (150,994,944 B; table 2 probed).
- `~/calib/{tv_sweep,led_sweep}.csv`, `fit_report.json`, `lut.cube` (July 2026, stock tuning).

---

## Appendix A: re-verifying HyperHDR's NV12 decode (read-only)

```python
import numpy as np
N = 256**3 * 3
m = np.memmap('/home/rpi/.hyperhdr/lut_lin_tables.3d', dtype=np.uint8, mode='r')
lut = lambda t, y, u, v: m[t*N + (y | u << 8 | v << 16)*3 : t*N + (y | u << 8 | v << 16)*3 + 3]
X, Y = [], []
for y in range(60, 200, 7):
    for u in range(100, 157, 4):
        for v in range(100, 157, 4):
            o = lut(2, y, u, v).astype(float)
            if (o > 2).all() and (o < 253).all():
                X.append([y-16, u-128, v-128, 1]); Y.append(o)
coef = np.linalg.lstsq(np.array(X, float), np.array(Y), rcond=None)[0].T
print(coef)   # expect ≈ [[1.164,0,1.596],[1.164,-0.392,-0.813],[1.164,2.017,0]] (BT.601 limited)
# also check: ls ~/.hyperhdr/flat_lut_lin_tables.3d must NOT exist
```

## Appendix B: HyperHDR v22 uniform transfer with our settings (computed from the source formulas)

`out_c = round(255 · OETF_sRGB(t_c · EOTF_sRGB(v/255)))`, t = (1, 0.38, 0.22):

| input v | out R | out G | out B | G/R | B/R |
|---|---|---|---|---|---|
| 2 | 2 | 1 | 0 | 0.380 | 0.220 |
| 5 | 5 | 2 | 1 | 0.380 | 0.220 |
| 10 | 10 | 4 | 2 | 0.380 | 0.220 |
| 20 | 20 | 9 | 5 | 0.438 | 0.254 |
| 32 | 32 | 17 | 10 | 0.523 | 0.327 |
| 64 | 64 | 38 | 27 | 0.595 | 0.430 |
| 127 | 127 | 80 | 61 | 0.632 | 0.480 |
| 200 | 200 | 129 | 100 | 0.645 | 0.499 |
| 255 | 255 | 166 | 129 | 0.650 | 0.506 |

Ratios are computed before rounding. HyperHDR's 512-entry interpolated tables differ from these exact values by at most 1 code.
