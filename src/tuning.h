#ifndef __TUNING_H__
#define __TUNING_H__

#include <string>
#include <vector>

// Frozen libcamera (Raspberry Pi IPA) tuning file for the calibrated LUT path
// (--make-tuning / --tuning). doc/led-lut-calibration.md, "Frozen tuning file".
//
// The stock tuning makes the camera a poor comparator for the LUT: adaptive
// contrast enhancement and adaptive lens shading change the camera's response
// with the scene, the stock tone curve compresses highlights ~4x (local
// exponent 1.6..4.2), and the generic colour matrix pushes saturated TV
// colours below zero where the ISP clips them. The frozen copy:
//   - rpi.contrast: ce_enable 0, gamma_curve = pure power (default 2.2) on the
//     hardware's knots (or the stock curve with --tuning-curve stock),
//   - rpi.alsc: n_iter 0 (calibrated tables only, no per-frame adaptation),
//   - rpi.ccm (with --from-calib): ONE matrix fitted to the TV's own
//     primaries from tv_raw.csv, so the camera reads ~TV RGB: no colour of the
//     TV goes below zero, TV grays land on the LUT diagonal, and the 10%
//     peak-white window lands at ~95% of the ISP range (headroom for OLED
//     highlights brighter than any full-field patch).

struct MakeTuningConfig
{
	std::string outPath;	// frozen tuning file to write
	std::string basePath;	// stock tuning to start from ("" = system file for the sensor)
	std::string sensorModel; // e.g. "imx708" (camera Model property)
	std::string curve = "2.2"; // "stock" or a power-law exponent
	std::string fromCalib;	// calib dir with tv_raw.csv (+ meta_tv-raw.json) -> TV-fitted CCM
	double desat = 0.05;	// safety desaturation of the TV matrix (floor above 0 for pure primaries)
	double peakTarget = 0.95; // ISP-linear level the peak-white window is scaled to
	bool verbose = false;
};

/// The stock tuning file libcamera loads for this sensor on this platform
/// (vc4 = RPi4, pisp = RPi5), or "" if not found.
std::string findSystemTuningFile(const std::string &sensorModel);

/// Write the frozen tuning file. Returns a process exit code.
int runMakeTuning(const MakeTuningConfig &cfg);

/// Linear light (0..1, pre-gamma ISP units) for ISP output codes 0..255 under
/// the tone curve of a tuning file, as the hardware realises it (vc4: the
/// curve sampled on its 33 knots, linear between). False if unreadable.
bool loadTuningToneCurve(const std::string &path, std::vector<double> &linOfCode);

/// FNV-1a 64 hex digest of a file ("" if unreadable) - provenance in meta_*.json.
std::string fileHash(const std::string &path);

#endif /* __TUNING_H__ */
