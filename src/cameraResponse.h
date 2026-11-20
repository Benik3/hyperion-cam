#ifndef __CAMERA_RESPONSE_H__
#define __CAMERA_RESPONSE_H__

#include <map>
#include <string>
#include <vector>

// Camera tone-curve recovery from the --ledcalib response shutter sweep
// (camera_response.csv). The ISP output code z of a fixed source is a single
// monotone function of the light reaching the sensor; the RAW frame of the
// same request measures that light directly (linear, black-subtracted, immune
// to source drift), so each (ROI, ISP channel) series gives z vs light up to an
// unknown per-series scale. A Debevec-Malik style least-squares fit recovers
// g(z) = ln f^-1(z) jointly with the scales. A frozen ISP gives one consistent
// curve; adaptive contrast/shading makes the curve scene-dependent and shows
// up as a large residual. doc/led-lut-calibration.md ("Camera response").

// Header-indexed numeric CSV (lines starting with '#' are skipped).
struct CsvTable
{
	std::map<std::string, int> col;
	std::vector<std::vector<double>> rows;
	std::vector<std::string> comments;

	bool has(const std::string &name) const { return col.count(name) > 0; }
	double get(std::size_t row, const std::string &name, double def = 0.0) const;
};
bool loadCsvTable(const std::string &path, CsvTable &out);

struct CameraResponse
{
	bool ok = false;
	// lin[z] = relative linear light for ISP code z = 0..255 (monotone,
	// lin[0] = 0, normalised so lin[255] = 1). Interpolated by eval().
	std::vector<double> lin;
	double rmsResid = 0.0;		 // RMS of the joint fit, ln units (~ relative error)
	double worstSeriesResid = 0.0; // worst per-series RMS, ln units
	double sourceDrift = 0.0;	 // max relative spread of RAW light / exposure over the reference steps
	int series = 0;
	int samples = 0;
	int zMin = 0, zMax = 255; // code range actually covered by data
	int spliceBelow = 0;	  // > 0: below this code the tuning curve's shape (no data there)

	double eval(double z) const;	 // encoded 0..255 -> relative linear
	double invert(double l) const; // relative linear -> encoded 0..255
};

/// What is known about the ISP state the sweep ran under (meta_response.json).
struct ResponseHints
{
	// Tone curve of the tuning file in effect (linear light per ISP code
	// 0..255): the smoothness prior of the fit (else a pure 2.2 power). The
	// frozen "power" curve is LINEAR below code ~38 (vc4's first knot segment),
	// so a power prior is wrong exactly where the dark end has the least data.
	std::vector<double> priorLin;
	// Inverse colour matrix in effect: with priorLin it reconstructs the
	// pre-matrix value, so samples clipped at the ISP's colour-gain stage (which
	// the RAW and the post-matrix readings can't show) are dropped.
	bool haveCcm = false;
	double ccmInv[9] = {};
	std::string tuningFile;
};

/// Hints from a phase meta file (tuning_file + frame.ccm). Missing parts stay empty.
bool loadResponseHints(const std::string &metaPath, ResponseHints &out);

/// Recover the tone curve from camera_response.csv. Prints details if verbose.
bool fitCameraResponse(const std::string &csvPath, CameraResponse &out, bool verbose,
					   const ResponseHints *hints = nullptr);

/// Fit + print the verdict the response phase reports (curve shape, fit
/// residual = frozen-ISP check, source drift). False only if no curve could be
/// recovered; a suspicious curve is a loud WARNING, not a failure.
bool ledCalibCheckResponse(const std::string &csvPath, bool verbose, const ResponseHints *hints = nullptr);

#endif /* __CAMERA_RESPONSE_H__ */
