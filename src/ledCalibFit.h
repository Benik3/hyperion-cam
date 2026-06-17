#ifndef __LED_CALIB_FIT_H__
#define __LED_CALIB_FIT_H__

#include <string>

// Phase 3 of the LED 3D-LUT calibration: fit a model to the --ledcalib sweep
// measurements and invert it into a .cube LUT for the runtime shader.
// Design + math: doc/led-lut-calibration.md ("Phase 3").
// No camera involved; runs offline on the measurement files and exits.

struct LedCalibFitConfig
{
	std::string calibDir = "ledcalib-out"; // input: led_sweep.csv, baseline.json, white_anchor.json
	std::string lutOut;					   // output .cube path (default <calibDir>/lut.cube)
	int lutSize = 33;					   // LUT grid per axis (trilinear runtime -> use dense grid)
	// Camera-clipping threshold: a reading >= this (per channel) is treated as
	// clipped/saturated. The white anchor uses the brightest neutral level where
	// neither the TV nor the LED reading clips (the camera maxes below 255, and
	// LEDs/TV may be brighter than each other). 0..255.
	double clipLevel = 250.0;
	bool verbose = false;
};

/// Fit + invert + write LUT. Returns process exit code.
int runLedCalibFit(const LedCalibFitConfig &cfg);

/// Numeric self-test: simulates a known LED/camera chain, generates synthetic
/// measurement files, runs the fit, and verifies the closed-loop error of the
/// produced LUT. Returns EXIT_SUCCESS on PASS.
int runLedCalibFitSelftest(const LedCalibFitConfig &cfg);

#endif /* __LED_CALIB_FIT_H__ */
