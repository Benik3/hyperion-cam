#ifndef __LED_CALIB_FIT_H__
#define __LED_CALIB_FIT_H__

#include <string>

// Phase 3 of the LED 3D-LUT calibration: fit a model to the --ledcalib sweep
// measurements and invert it into a .cube LUT for the runtime shader.
// Design + math: doc/led-lut-calibration.md ("Phase 3").
// No camera involved; runs offline on the measurement files and exits.

struct LedCalibFitConfig
{
	std::string calibDir = "ledcalib-out"; // input: led_sweep.csv, tv_sweep.csv, baseline.json
	std::string lutOut;					   // output .cube path (default <calibDir>/lut.cube)
	int lutSize = 65;					   // LUT grid per axis (trilinear runtime -> use dense grid)
	// Camera-clipping threshold: a reading >= this (per channel) is treated as
	// clipped/saturated. The white anchor uses the brightest neutral level where
	// neither the TV nor the LED reading clips (the camera maxes below 255, and
	// LEDs/TV may be brighter than each other). 0..255.
	double clipLevel = 250.0;
	// Dark-end curve regularization: per channel, drive-curve points below this
	// fraction of the channel's full output are unreliable (the camera floors out
	// on dim LED glow, worst on a heavily white-balance-attenuated channel) and
	// get replaced by a smooth power law extrapolated from the reliable region
	// above. Removes the monotonize() plateaus that cause dark-grey crush + colour
	// cast. 0 disables (use the raw measured bottom). doc/led-lut-calibration.md.
	double curveFloor = 0.05;
	// Neutral level at which the von-Kries metameric white is anchored (the fit
	// computes gains cam_tv_white/cam_led_white here). 0 = auto: use the sweep's
	// meta.json "white_level" (the --ledcalib-white-level the sweeps ran with),
	// falling back to 127. Set >0 to override at fit time. doc/led-lut-calibration.md.
	int whiteAnchorLevel = 0;
	// Camera model: "auto" = the measured tone curve when camera_response.csv
	// exists (and the LEDs on RAW when led_sweep.csv has RAW columns), else the
	// legacy power law; "on" = require it; "off" = force the legacy model.
	std::string response = "auto";
	// Soft knee: "auto" (start at 85% of full LED output), "off", or the start in %.
	std::string knee = "auto";
	// Dark fallback floor (encoded camera units, brightest channel): "auto" (TV
	// black + 8 codes), "off", or a level.
	std::string darkFloor = "auto";
	// Round-off-aware dark end (snap dark nodes to whole LED drive steps measured
	// from the sweep's low ramp): "auto" (when the staircase is resolved), "on", "off".
	std::string darkQuant = "auto";
	bool verbose = false;
};

/// Fit + invert + write LUT. Returns process exit code.
int runLedCalibFit(const LedCalibFitConfig &cfg);

/// Numeric self-test: simulates a known LED/camera chain, generates synthetic
/// measurement files, runs the fit, and verifies the closed-loop error of the
/// produced LUT. Returns EXIT_SUCCESS on PASS.
int runLedCalibFitSelftest(const LedCalibFitConfig &cfg);

#endif /* __LED_CALIB_FIT_H__ */
