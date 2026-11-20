#include "ledCalibFit.h"
#include "cameraResponse.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Math (doc/led-lut-calibration.md, Phase 3):
//
// LED light adds linearly, so after linearizing the camera's encoded readings
// with a power law (exponent p, self-estimated from the sweep's own
// additivity), the glow obeys
//     lin(reading) - lin(baseline) = A * [eR(Sr), eG(Sg), eB(Sb)]^T
// with a 3x3 mixing matrix A (LED spectra x camera CFA x wall) and three
// monotone drive curves e_c. The white-anchor gains g act on encoded values,
// i.e. g^p in linear space; for a pure power law the "glow looks like a
// proportionally dimmed patch" target then reduces to: corrected linear glow
// == lin(v) for LUT input v. Inversion per LUT node: e* = A^-1 * t, hue-true
// gamut mapping (desaturate towards the luminance-matched gray for negative
// components, scale down for overflow), then invert the monotone curves.

namespace
{
	struct SweepSample
	{
		double drive[3];
		cv::Vec3d led;			  // encoded LED-ROI mean (ISP)
		cv::Vec3d ledSat{0, 0, 0}; // ISP ceiling fraction (0 if not recorded)
		bool hasRaw = false;
		cv::Vec3d raw{0, 0, 0};	  // black-subtracted RAW means at the LED ROI
		cv::Vec3d rawSat{0, 0, 0}; // sensor ceiling fraction
	};

	// Camera encoded reading (0..255) <-> linear light. Either the tone curve
	// measured by --ledcalib response (camera_response.csv), or the legacy single
	// power law self-estimated from LED additivity. doc/led-lut-calibration.md.
	struct Linearizer
	{
		bool measured = false;
		double p = 2.2;
		CameraResponse resp;

		double lin(double enc) const
		{
			return measured ? resp.eval(enc) : std::pow(std::max(enc, 0.0) / 255.0, p);
		}
		double enc(double l) const
		{
			return measured ? resp.invert(l) : 255.0 * std::pow(std::clamp(l, 0.0, 1.0), 1.0 / p);
		}
		cv::Vec3d lin3(const cv::Vec3d &e) const { return {lin(e[0]), lin(e[1]), lin(e[2])}; }
	};

	struct CalibData
	{
		std::vector<SweepSample> samples;
		cv::Vec3d baselineLed{0, 0, 0};
		cv::Vec3d baselineRaw{0, 0, 0};
		bool baselineHasRaw = false;
		// Camera model chosen at load time: measured response (+ RAW LED path when
		// the sweep has RAW columns) or the legacy power law.
		Linearizer cam;
		bool useRaw = false;
		// Camera luma of the TV's black patch (tv_sweep 0,0,0), for the automatic
		// dark floor; < 0 = unknown.
		double tvBlackLuma = -1.0;
		double anchorDrive[3] = {255, 255, 255};
		cv::Vec3d anchorTv{255, 255, 255};
		cv::Vec3d anchorLed{255, 255, 255};
		cv::Vec3d gains{1, 1, 1};
		double anchorLevelTarget = 127; // neutral level to anchor the white at
	};

	// Monotone curve as a (level -> value) table with linear interpolation.
	struct Curve
	{
		std::vector<double> level; // ascending drive levels 0..255
		std::vector<double> value; // non-decreasing e values

		double eval(double s) const
		{
			if (level.empty())
				return 0.0;
			if (s <= level.front())
				return value.front();
			if (s >= level.back())
				return value.back();
			std::size_t hi = 1;
			while (hi < level.size() && level[hi] < s)
				hi++;
			const double l0 = level[hi - 1], l1 = level[hi];
			const double t = (l1 > l0) ? (s - l0) / (l1 - l0) : 0.0;
			return value[hi - 1] + t * (value[hi] - value[hi - 1]);
		}

		double invert(double e) const
		{
			if (value.empty())
				return 0.0;
			if (e <= value.front())
				return level.front();
			if (e >= value.back())
				return level.back();
			std::size_t hi = 1;
			while (hi < value.size() && value[hi] < e)
				hi++;
			const double v0 = value[hi - 1], v1 = value[hi];
			const double t = (v1 > v0) ? (e - v0) / (v1 - v0) : 0.0;
			return level[hi - 1] + t * (level[hi] - level[hi - 1]);
		}
	};

	struct FitModel
	{
		double p = 2.2;		  // power-law exponent (legacy model; reported only when measured)
		Linearizer cam;		  // the camera linearization actually used
		bool rawLed = false;  // LED light measured on RAW (mapped by rawToLin)
		cv::Matx33d rawToLin = cv::Matx33d::eye(); // RAW LED means -> ISP-linear
		double rawMapResid = 0.0;
		// Soft knee: max-drive fraction m -> knee(m) above kneeStart, so the
		// brightest LUT input (camera white) maps to full LED output.
		bool kneeOn = false;
		double kneeStart = 0.85, kneeMax = 1.0, kneeA = 0.0;
		double kneeLumWhite = 0.0; // modelled luminance camera white needs (pre-knee)
		// Eye-matched white in drive-fraction space (camera white's solution,
		// normalised to max 1): the gray that near-neutral overflow desaturates to.
		cv::Vec3d whiteDir{0, 0, 0};
		// Dark fallback: below darkFloor (encoded camera luma) blend to equal
		// R=G=B drive (HyperHDR's eye-matched white) at matching luminance.
		double darkFloor = 0.0;
		cv::Matx33d A;
		cv::Matx33d Ainv;
		Curve e[3];
		cv::Vec3d corr{1, 1, 1}; // g^p * anchor renorm, applied to A*e
		cv::Vec3d baselineLin{0, 0, 0};
		// diagnostics
		double additivityResid = 0.0;
		double residMedian = 0.0;
		double residP95 = 0.0;
		cv::Vec3d clipCeil{0, 0, 0};      // per-channel sensor-clip ceiling (0 = none)
		int clipExcluded[3] = {0, 0, 0};  // samples dropped as sensor-clipped, per channel
		cv::Vec3d sweepFloor{0, 0, 0};    // per-channel dark-floor reading (drive 0)
	};

	enum NodeFlag : uint8_t
	{
		NodeInGamut = 0,
		NodeDesaturated = 1,
		NodeLumClipped = 2,
		NodeBlack = 4,
		NodeKnee = 8, // brightness compressed by the soft knee (policy, not error)
		NodeDark = 16 // blended toward neutral equal drive below the dark floor
	};

	struct FitResult
	{
		FitModel model;
		int n = 33;
		std::vector<cv::Vec3d> lut; // drive 0..255, red index fastest
		std::vector<uint8_t> flags;
		double pctDesat = 0.0, pctClip = 0.0, pctKnee = 0.0, pctDark = 0.0;
		// Diagnostics: per TV gray patch {level, camera reading (G), LUT drive R,G,B}.
		std::vector<std::array<double, 5>> grayAxis;
	};

	double linv(double enc, double p)
	{
		return std::pow(std::max(enc, 0.0) / 255.0, p);
	}

	cv::Vec3d linv3(const cv::Vec3d &enc, double p)
	{
		return {linv(enc[0], p), linv(enc[1], p), linv(enc[2], p)};
	}

	// ---------- input parsing ----------

	bool readFile(const std::string &path, std::string &out)
	{
		std::ifstream f(path);
		if (!f)
			return false;
		std::stringstream ss;
		ss << f.rdbuf();
		out = ss.str();
		return true;
	}

	bool jsonTriple(const std::string &text, const char *key, cv::Vec3d &out)
	{
		const std::string needle = std::string("\"") + key + "\"";
		std::size_t k = text.find(needle);
		if (k == std::string::npos)
			return false;
		k = text.find('[', k);
		if (k == std::string::npos)
			return false;
		return std::sscanf(text.c_str() + k, "[ %lf , %lf , %lf", &out[0], &out[1], &out[2]) == 3;
	}

	bool jsonScalar(const std::string &text, const char *key, double &out)
	{
		const std::string needle = std::string("\"") + key + "\"";
		std::size_t k = text.find(needle);
		if (k == std::string::npos)
			return false;
		k = text.find(':', k);
		if (k == std::string::npos)
			return false;
		return std::sscanf(text.c_str() + k + 1, " %lf", &out) == 1;
	}

	bool loadLedSweepCsv(const std::string &path, std::vector<SweepSample> &out)
	{
		// Header-indexed: drive_* + led_* (ISP) are required; led_*_sat and the RAW
		// columns (raw_*, raw_*_sat; RAW+ISP dual-stream sweeps) are optional.
		CsvTable t;
		if (!loadCsvTable(path, t))
		{
			std::cout << "ERROR: cannot open " << path << std::endl;
			return false;
		}
		for (const char *n : {"drive_r", "drive_g", "drive_b", "led_r", "led_g", "led_b"})
			if (!t.has(n))
			{
				std::cout << "ERROR: " << path << " lacks column " << n << std::endl;
				return false;
			}
		const bool raw = t.has("raw_r") && t.has("raw_g") && t.has("raw_b");
		const char *ch[3] = {"r", "g", "b"};
		for (std::size_t i = 0; i < t.rows.size(); i++)
		{
			SweepSample s;
			for (int c = 0; c < 3; c++)
			{
				s.drive[c] = t.get(i, std::string("drive_") + ch[c]);
				s.led[c] = t.get(i, std::string("led_") + ch[c]);
				s.ledSat[c] = t.get(i, std::string("led_") + ch[c] + "_sat");
				if (raw)
				{
					s.raw[c] = t.get(i, std::string("raw_") + ch[c]);
					s.rawSat[c] = t.get(i, std::string("raw_") + ch[c] + "_sat");
				}
			}
			s.hasRaw = raw;
			out.push_back(s);
		}
		return !out.empty();
	}

	// Camera luma of the TV black patch (0,0,0) in tv_sweep.csv; -1 if absent.
	double loadTvBlackLuma(const std::string &path)
	{
		CsvTable t;
		if (!loadCsvTable(path, t) || !t.has("tv_r"))
			return -1.0;
		for (std::size_t i = 0; i < t.rows.size(); i++)
			if (t.get(i, "patch_r", 1) == 0 && t.get(i, "patch_g", 1) == 0 && t.get(i, "patch_b", 1) == 0)
				return 0.299 * t.get(i, "tv_r") + 0.587 * t.get(i, "tv_g") + 0.114 * t.get(i, "tv_b");
		return -1.0;
	}

	// All neutral (r==g==b) TV patches from tv_sweep.csv: (commanded level,
	// cam_tv). Parsed by header name. Returns false if the file/columns absent.
	bool loadTvNeutrals(const std::string &path, std::vector<std::pair<double, cv::Vec3d>> &out)
	{
		std::ifstream f(path);
		if (!f)
			return false;
		std::string line;
		if (!std::getline(f, line))
			return false;
		while (!line.empty() && line[0] == '#')
			if (!std::getline(f, line))
				return false;
		std::map<std::string, int> col;
		{
			std::stringstream ss(line);
			std::string cell;
			int i = 0;
			while (std::getline(ss, cell, ','))
				col[cell] = i++;
		}
		const char *need[] = {"patch_r", "patch_g", "patch_b", "tv_r", "tv_g", "tv_b"};
		for (const char *n : need)
			if (!col.count(n))
				return false;
		while (std::getline(f, line))
		{
			if (line.empty())
				continue;
			std::vector<double> v;
			std::stringstream ss(line);
			std::string cell;
			while (std::getline(ss, cell, ','))
				v.push_back(std::atof(cell.c_str()));
			if (static_cast<int>(v.size()) <= col["tv_b"])
				continue;
			const double pr = v[col["patch_r"]], pg = v[col["patch_g"]], pb = v[col["patch_b"]];
			if (std::fabs(pr - pg) > 1.0 || std::fabs(pg - pb) > 1.0)
				continue; // not neutral
			out.push_back({pr, cv::Vec3d{v[col["tv_r"]], v[col["tv_g"]], v[col["tv_b"]]}});
		}
		return !out.empty();
	}

	// Result of deriving the white anchor from the sweep neutrals.
	//   Derived    - found a clean neutral level; gains set.
	//   NoData     - no tv_sweep neutrals / no LED neutrals to derive from
	//                (fall back to neutral gains, just no metameric correction).
	//   AllClipped - neutrals exist but every usable level saturates the camera;
	//                a hard error - the LUT would be built on blown-out white.
	enum class AnchorResult
	{
		Derived,
		NoData,
		AllClipped
	};

	// HyperHDR-WB workflow: derive the von-Kries white from the neutral level
	// closest to d.anchorLevelTarget (the --ledcalib-white-level the sweeps ran
	// with, default mid-gray 127) at which NEITHER the TV nor the LED camera
	// reading clips. Anchoring at the eye-match level — rather than the brightest
	// clean level — computes the metameric gains where the WB eye-match is most
	// trustworthy and where clipping is least likely; the sweeps inject that exact
	// neutral so the TV side is a real patch, not interpolated. If the target level
	// itself is clipped/absent we take the nearest clean neutral instead.
	// doc/led-lut-calibration.md.
	AnchorResult deriveAnchorFromSweeps(const std::string &dir, CalibData &d, double clip)
	{
		std::vector<std::pair<double, cv::Vec3d>> tvN;
		if (!loadTvNeutrals(dir + "/tv_sweep.csv", tvN))
			return AnchorResult::NoData;

		// Neutral LED drives -> cam_led, sorted by level for interpolation.
		std::vector<std::pair<double, cv::Vec3d>> ledN;
		for (const auto &s : d.samples)
			if (std::fabs(s.drive[0] - s.drive[1]) < 1.0 && std::fabs(s.drive[1] - s.drive[2]) < 1.0)
				ledN.push_back({s.drive[0], s.led});
		if (ledN.empty())
			return AnchorResult::NoData;
		std::sort(ledN.begin(), ledN.end(), [](auto &a, auto &b) { return a.first < b.first; });
		auto ledAt = [&](double L) -> cv::Vec3d
		{
			if (L <= ledN.front().first)
				return ledN.front().second;
			if (L >= ledN.back().first)
				return ledN.back().second;
			for (std::size_t i = 1; i < ledN.size(); i++)
				if (ledN[i].first >= L)
				{
					const double t = (L - ledN[i - 1].first) / std::max(1e-9, ledN[i].first - ledN[i - 1].first);
					return ledN[i - 1].second + (ledN[i].second - ledN[i - 1].second) * t;
				}
			return ledN.back().second;
		};

		auto clipped = [&](const cv::Vec3d &v)
		{ return v[0] >= clip || v[1] >= clip || v[2] >= clip; };

		// Clean neutral (both sides unclipped) nearest the target level. A hard
		// floor of 32 keeps a noise-dominated dark patch from ever being the white
		// reference even if it is the closest clean level to a very dim target.
		constexpr double kMinAnchor = 32.0;
		const double target = d.anchorLevelTarget;
		bool found = false;
		double bestDist = 1e18, bestL = 0;
		cv::Vec3d bestTv, bestLed;
		for (const auto &[L, tv] : tvN)
		{
			if (L < kMinAnchor)
				continue; // too dark to be a usable white reference
			const cv::Vec3d led = ledAt(L);
			if (clipped(tv) || clipped(led))
				continue;
			const double dist = std::fabs(L - target);
			if (!found || dist < bestDist)
			{
				found = true;
				bestDist = dist;
				bestL = L;
				bestTv = tv;
				bestLed = led;
			}
		}
		if (found)
		{
			d.anchorDrive[0] = d.anchorDrive[1] = d.anchorDrive[2] = bestL;
			d.anchorTv = bestTv;
			d.anchorLed = bestLed;
			for (int c = 0; c < 3; c++)
				d.gains[c] = bestLed[c] > 1.0 ? bestTv[c] / bestLed[c] : 1.0;
			std::cout << "White anchor from sweeps at neutral level " << bestL
					  << " (target " << target << (bestL != target ? ", nearest clean neutral" : "") << "): "
					  << "TV " << bestTv[0] << "," << bestTv[1] << "," << bestTv[2]
					  << "  LED " << bestLed[0] << "," << bestLed[1] << "," << bestLed[2]
					  << "  gains " << d.gains[0] << "," << d.gains[1] << "," << d.gains[2] << std::endl;
			return AnchorResult::Derived;
		}
		std::cout << "ERROR: no unclipped neutral level found (every white patch saturates the camera "
					 "at >= " << clip << "). Lower the TV and/or LED brightness, or raise --calib-clip-level."
				  << std::endl;
		return AnchorResult::AllClipped;
	}

	// The sweeps a fit combines must come from the same camera setup: compare
	// the tuning file hash + camera args recorded in meta_<phase>.json. Mismatch
	// = WARNING (e.g. a tv sweep with the stock tuning and a led sweep with the
	// frozen one gives a LUT for neither).
	void checkPhaseMetas(const std::string &dir)
	{
		struct M
		{
			std::string phase, hash, camera;
		};
		std::vector<M> metas;
		for (const char *ph : {"response", "tv", "led", "shading"})
		{
			std::string txt;
			if (!readFile(dir + "/meta_" + ph + ".json", txt))
				continue;
			auto strField = [&](const char *key)
			{
				const std::string needle = std::string("\"") + key + "\": \"";
				const std::size_t k = txt.find(needle);
				if (k == std::string::npos)
					return std::string();
				const std::size_t e = txt.find('"', k + needle.size());
				return txt.substr(k + needle.size(), e == std::string::npos ? 0 : e - k - needle.size());
			};
			// camera args that must match (size/crop legitimately differ per phase)
			std::string cam;
			for (const char *key : {"shutter_us", "analogue_gain", "saturation", "contrast"})
			{
				double v = 0;
				if (jsonScalar(txt, key, v))
					cam += std::string(key) + "=" + std::to_string(v) + " ";
			}
			const std::size_t cgPos = txt.find("\"colour_gains\"", txt.find("\"camera\""));
			double g0 = 0, g1 = 0;
			if (cgPos != std::string::npos &&
				std::sscanf(txt.c_str() + txt.find('[', cgPos), "[ %lf , %lf", &g0, &g1) == 2)
				cam += "colour_gains=" + std::to_string(g0) + "," + std::to_string(g1);
			metas.push_back({ph, strField("tuning_hash"), cam});
		}
		for (std::size_t i = 1; i < metas.size(); i++)
		{
			if (metas[i].hash != metas[0].hash)
				std::cout << "WARNING: " << metas[i].phase << " was measured with a different tuning file than "
						  << metas[0].phase << " (hash '" << metas[i].hash << "' vs '" << metas[0].hash
						  << "'); re-measure with the same --tuning" << std::endl;
			if (metas[i].camera != metas[0].camera)
				std::cout << "WARNING: camera settings differ between " << metas[0].phase << " and " << metas[i].phase
						  << ": [" << metas[0].camera << "] vs [" << metas[i].camera << "]" << std::endl;
		}
	}

	bool loadCalibData(const std::string &dir, CalibData &d, double clip = 250.0, int anchorOverride = 0,
					   const std::string &responseMode = "auto")
	{
		if (!loadLedSweepCsv(dir + "/led_sweep.csv", d.samples))
			return false;
		checkPhaseMetas(dir);
		d.tvBlackLuma = loadTvBlackLuma(dir + "/tv_sweep.csv");

		// Camera model: the measured tone curve when the response phase ran
		// (--calib-response auto), else the legacy self-estimated power law.
		const std::string respPath = dir + "/camera_response.csv";
		if (responseMode != "off" && std::filesystem::exists(respPath))
		{
			if (!fitCameraResponse(respPath, d.cam.resp, false))
				return false;
			d.cam.measured = true;
			std::cout << "Camera response: measured curve from camera_response.csv (" << d.cam.resp.samples
					  << " samples, fit RMS " << (100.0 * d.cam.resp.rmsResid) << "%)" << std::endl;
			if (d.cam.resp.worstSeriesResid > 0.04)
				std::cout << "WARNING: camera_response.csv is not one consistent curve (worst series "
						  << (100.0 * d.cam.resp.worstSeriesResid) << "%): was the ISP frozen (--tuning)?" << std::endl;
		}
		else if (responseMode == "on")
		{
			std::cout << "ERROR: --calib-response on but no " << respPath << " (run --ledcalib response)" << std::endl;
			return false;
		}
		else
			std::cout << "Camera response: " << (responseMode == "off" ? "disabled" : "no camera_response.csv")
					  << " -> legacy power-law model (exponent self-estimated from LED additivity)" << std::endl;
		// LEDs on RAW: needs the measured curve to map RAW into the same linear
		// units as the TV readings (the legacy model keeps the old ISP-only path).
		const bool rawCols = !d.samples.empty() && d.samples.front().hasRaw;
		d.useRaw = d.cam.measured && rawCols;
		if (rawCols && !d.cam.measured)
			std::cout << "INFO: led_sweep.csv has RAW columns, unused by the legacy model" << std::endl;

		// White-anchor level: CLI override (>0) wins; else the level the sweeps
		// recorded in meta.json; else the mid-gray default already in CalibData.
		if (anchorOverride > 0)
			d.anchorLevelTarget = anchorOverride;
		else
		{
			std::string meta;
			double wl = 0;
			if (readFile(dir + "/meta.json", meta) && jsonScalar(meta, "white_level", wl) && wl > 0)
				d.anchorLevelTarget = wl;
		}

		std::string txt;
		if (readFile(dir + "/baseline.json", txt))
		{
			jsonTriple(txt, "led_mean", d.baselineLed);
			d.baselineHasRaw = jsonTriple(txt, "led_raw_mean", d.baselineRaw);
		}
		else
		{
			std::cout << "WARNING: no baseline.json; using the (0,0,0) sweep sample" << std::endl;
			for (const auto &s : d.samples)
				if (s.drive[0] == 0 && s.drive[1] == 0 && s.drive[2] == 0)
					d.baselineLed = s.led;
		}

		// Derive the von-Kries white from the sweep white points: cam_tv_white from
		// tv_sweep.csv vs cam_led_white from led_sweep.csv, at the brightest neutral
		// level where neither clips. Requires HyperHDR's white balance to have
		// eye-matched white before measuring (run --ledcalib white for that).
		switch (deriveAnchorFromSweeps(dir, d, clip))
		{
		case AnchorResult::Derived:
			return true;
		case AnchorResult::AllClipped:
			// Error already printed; abort rather than silently ship a LUT built on
			// a blown-out white point.
			return false;
		case AnchorResult::NoData:
			std::cout << "WARNING: could not derive white from tv_sweep.csv; assuming neutral gains "
						 "(no metameric correction - set HyperHDR white balance and include neutral "
						 "patches in the TV sweep)" << std::endl;
			return true;
		}
		return true;
	}

	// ---------- exponent estimation (additivity) ----------

	// Single-channel axis readings must sum (in linear light, baseline removed)
	// to the mixed-channel readings. Scan p for the best relative fit.
	double additivityResidual(const CalibData &d, double p)
	{
		std::map<int, cv::Vec3d> axis[3];
		cv::Vec3d black = d.baselineLed;
		for (const auto &s : d.samples)
		{
			const bool r = s.drive[0] > 0, g = s.drive[1] > 0, b = s.drive[2] > 0;
			if (r && !g && !b)
				axis[0][static_cast<int>(s.drive[0])] = s.led;
			else if (!r && g && !b)
				axis[1][static_cast<int>(s.drive[1])] = s.led;
			else if (!r && !g && b)
				axis[2][static_cast<int>(s.drive[2])] = s.led;
		}

		const cv::Vec3d blackLin = linv3(black, p);
		auto glow = [&](const cv::Vec3d &enc)
		{
			cv::Vec3d y = linv3(enc, p) - blackLin;
			return y;
		};

		double num = 0.0, den = 0.0;
		int used = 0;
		for (const auto &s : d.samples)
		{
			int on = (s.drive[0] > 0) + (s.drive[1] > 0) + (s.drive[2] > 0);
			if (on < 2)
				continue;
			auto itR = axis[0].find(static_cast<int>(s.drive[0]));
			auto itG = axis[1].find(static_cast<int>(s.drive[1]));
			auto itB = axis[2].find(static_cast<int>(s.drive[2]));
			cv::Vec3d pred{0, 0, 0};
			if (s.drive[0] > 0)
			{
				if (itR == axis[0].end())
					continue;
				pred += glow(itR->second);
			}
			if (s.drive[1] > 0)
			{
				if (itG == axis[1].end())
					continue;
				pred += glow(itG->second);
			}
			if (s.drive[2] > 0)
			{
				if (itB == axis[2].end())
					continue;
				pred += glow(itB->second);
			}
			const cv::Vec3d meas = glow(s.led);
			const cv::Vec3d diff = pred - meas;
			num += diff.dot(diff);
			den += meas.dot(meas);
			used++;
		}
		if (used == 0 || den <= 0.0)
			return 1e9;
		return num / den;
	}

	double estimateExponent(const CalibData &d, double &residOut)
	{
		// additivityResidual returns this sentinel when there are no usable
		// multi-channel samples to measure additivity against. Never let it win
		// the scan: bestR starts above it, so the first p (~1.2) would otherwise
		// be "best" and pin the exponent at 1.2 instead of the documented 2.2.
		constexpr double kNoData = 1e9;
		double bestP = 2.2, bestR = 1e18;
		bool measured = false;
		for (double p = 1.2; p <= 3.2 + 1e-9; p += 0.05)
		{
			const double r = additivityResidual(d, p);
			if (r < kNoData && r < bestR)
			{
				bestR = r;
				bestP = p;
				measured = true;
			}
		}
		if (!measured)
		{
			residOut = bestR; // sentinel; signals additivity was unmeasurable
			return 2.2;        // keep the neutral default rather than 1.2
		}
		for (double p = bestP - 0.05; p <= bestP + 0.05 + 1e-9; p += 0.005)
		{
			const double r = additivityResidual(d, p);
			if (r < kNoData && r < bestR)
			{
				bestR = r;
				bestP = p;
			}
		}
		residOut = bestR;
		return bestP;
	}

	// ---------- model fit (alternating least squares) ----------

	void monotonize(Curve &c)
	{
		double run = 0.0;
		for (std::size_t i = 0; i < c.value.size(); i++)
		{
			run = std::max(run, std::max(0.0, c.value[i]));
			c.value[i] = run;
		}
	}

	// Dark-end regularization. The camera cannot measure dim LED glow at the
	// locked exposure - a white-balance-attenuated channel (e.g. HyperHDR green at
	// 0.25) floors out first - so the bottom of a measured drive curve is noise
	// that monotonize() flattens into a plateau. Inverting a plateau makes that
	// channel switch on abruptly at a drive level unrelated to its neighbours,
	// producing dark-grey crush + colour cast (e.g. a green jump). Replace every
	// curve point below relFloor*full-output with a smooth power law
	// e = e(s*) * (s/s*)^gamma, anchored at the first reliable level s* and with
	// gamma fit log-log from the low reliable region. The result rises
	// monotonically from 0 with no plateau, so all channels turn on consistently.
	// Returns the (level, gamma) actually used for diagnostics; gamma<0 == no-op.
	std::pair<double, double> regularizeCurveBottom(Curve &c, double relFloor)
	{
		if (relFloor <= 0.0 || c.value.size() < 3)
			return {0.0, -1.0};
		const double vmax = c.value.back();
		if (vmax <= 0.0)
			return {0.0, -1.0};
		const double thr = relFloor * vmax;

		// First reliable level: lowest strictly-positive drive with value >= thr.
		std::size_t kstar = 0;
		while (kstar < c.value.size() && (c.level[kstar] <= 0.0 || c.value[kstar] < thr))
			kstar++;
		if (kstar == 0 || kstar >= c.value.size())
			return {0.0, -1.0}; // nothing below threshold, or no reliable point

		const double sStar = c.level[kstar], vStar = c.value[kstar];

		// Fit gamma over the low reliable region via log-log LS; keep it local to
		// the dark end (cap at half full output) but always take at least a few
		// points so a sparse low grid doesn't pin gamma on two noisy samples.
		double sx = 0, sy = 0, sxx = 0, sxy = 0;
		int n = 0;
		for (std::size_t k = kstar; k < c.value.size(); k++)
		{
			if ((c.value[k] > 6.0 * thr && n >= 3) || c.value[k] > 0.5 * vmax)
				break;
			if (c.level[k] <= 0.0 || c.value[k] <= 0.0)
				continue;
			const double x = std::log(c.level[k]), y = std::log(c.value[k]);
			sx += x; sy += y; sxx += x * x; sxy += x * y; n++;
		}
		double gamma = 1.0; // fallback: linear in drive (LED ~ linear hypothesis)
		if (n >= 3) // need >=3 points (matches the loop guard); 2 noisy samples -> fall back to linear
		{
			const double denom = n * sxx - sx * sx;
			if (std::fabs(denom) > 1e-9)
				gamma = (n * sxy - sx * sy) / denom;
		}
		gamma = std::clamp(gamma, 0.5, 4.0); // physical guard

		for (std::size_t k = 0; k < kstar; k++)
			c.value[k] = (c.level[k] <= 0.0) ? 0.0 : vStar * std::pow(c.level[k] / sStar, gamma);
		monotonize(c);
		return {sStar, gamma};
	}

	// Detect SENSOR clipping in the LED sweep. The camera's ADC clamps at a single
	// near-full-scale value (~254 for 8-bit), so a channel is saturated only where
	// its reading PINS at that one ceiling - its true light is then brighter than
	// measured, and those samples must not bias the curve fit or the e(255)=1
	// normalization. A channel that maxes BELOW the ceiling - soft LED compression
	// (still rising) or a WB-attenuated channel - is NOT clipped and is kept. Keying
	// off each channel's own max instead would wrongly exclude a soft-compressed
	// channel. Sets ceilOut[c] (a reading >= this is clipped; 0 == no clip) and marks
	// the clipped samples. Verified on real-rig data: blue pins at ~254 (clipped),
	// red maxes ~250 still rising (kept), green maxes ~200 (kept).
	void detectClipCeilings(const std::vector<SweepSample> &samples, cv::Vec3d &ceilOut,
							std::vector<char> &clipped)
	{
		ceilOut = cv::Vec3d{0, 0, 0};
		clipped.assign(samples.size(), 0);
		double maxAll = 0.0;
		for (const auto &s : samples)
			for (int c = 0; c < 3; c++)
				maxAll = std::max(maxAll, s.led[c]);
		if (maxAll < 245.0)
			return; // nothing reached the sensor ceiling -> no clipping anywhere
		constexpr double kEps = 2.5;	// readings within this of the ADC ceiling are pinned
		constexpr double kOwnEps = 1.0; // ...but only if the channel itself reaches the ceiling
		constexpr int kPlateauMin = 5;	// need a cluster at the ceiling, not a lone peak
		const double ceiling = maxAll - kEps;
		for (int c = 0; c < 3; c++)
		{
			// A clipped channel by definition attains the single ADC ceiling
			// (maxAll). A soft-compressed channel whose plateau merely lands
			// within kEps of another channel's pin (e.g. 252-253 vs a 254 pin)
			// is still rising and must be kept, so require the channel's own
			// max to reach the ceiling itself.
			double ownMax = 0.0;
			for (const auto &s : samples)
				ownMax = std::max(ownMax, s.led[c]);
			if (ownMax < maxAll - kOwnEps)
				continue;
			int nearCeil = 0;
			for (const auto &s : samples)
				if (s.led[c] >= ceiling)
					nearCeil++;
			if (nearCeil >= kPlateauMin)
				ceilOut[c] = ceiling; // this channel pins at the sensor ceiling
		}
		for (std::size_t i = 0; i < samples.size(); i++)
			for (int c = 0; c < 3; c++)
				if (ceilOut[c] > 0 && samples[i].led[c] >= ceilOut[c])
					clipped[i] = 1;
	}

	// RAW LED path: map the black-subtracted RAW means at the LED ROI into the
	// ISP-linear units the TV readings are linearized to (the measured camera
	// response), from the sweep's own samples where the ISP reading is clean
	// (above the floor, below the clip). Per ISP channel r:
	//   lin(isp_r) = sum_c M[r][c] * raw_c      (weighted for relative error)
	// M absorbs the colour matrix, WB gains, lens shading at the LED ROI and the
	// ISP's black/scale handling, so nothing about the ISP has to be known.
	bool fitRawToLin(const CalibData &d, FitModel &m)
	{
		double ssAll = 0.0;
		int nAll = 0;
		for (int r = 0; r < 3; r++)
		{
			std::vector<double> Ar, br;
			for (const auto &s : d.samples)
			{
				if (!s.hasRaw || std::max({s.rawSat[0], s.rawSat[1], s.rawSat[2]}) > 0.002)
					continue;
				if (s.ledSat[r] > 0.005 || s.led[r] > 245.0 || s.led[r] < 12.0)
					continue;
				const double y = d.cam.lin(s.led[r]);
				if (y <= 0.0)
					continue;
				const double w = 1.0 / y;
				for (int c = 0; c < 3; c++)
					Ar.push_back(w * s.raw[c]);
				br.push_back(1.0);
			}
			const int n = static_cast<int>(br.size());
			if (n < 10)
			{
				std::cout << "ERROR: too few clean LED samples (" << n << ") to map RAW -> ISP for channel " << "rgb"[r]
						  << std::endl;
				return false;
			}
			cv::Mat A(n, 3, CV_64F, Ar.data()), b(n, 1, CV_64F, br.data()), x;
			if (!cv::solve(A, b, x, cv::DECOMP_SVD))
				return false;
			for (int c = 0; c < 3; c++)
				m.rawToLin(r, c) = x.at<double>(c);
			cv::Mat res = A * x - b;
			for (int i = 0; i < n; i++)
				ssAll += res.at<double>(i) * res.at<double>(i);
			nAll += n;
		}
		m.rawMapResid = nAll ? std::sqrt(ssAll / nAll) : 0.0;
		m.rawLed = true;
		std::cout << "LED sweep on RAW: RAW->ISP-linear map fitted on the clean samples, relative RMS "
				  << (100.0 * m.rawMapResid) << "%" << std::endl;
		if (m.rawMapResid > 0.05)
			std::cout << "WARNING: RAW and ISP disagree by " << (100.0 * m.rawMapResid)
					  << "% RMS on the LED ROI (ISP not frozen, or a wrong camera response?)" << std::endl;
		return true;
	}

	bool fitModel(const CalibData &d, FitModel &m, bool verbose, double curveFloor)
	{
		m.cam = d.cam;
		if (!m.cam.measured)
		{
			m.p = estimateExponent(d, m.additivityResid);
			m.cam.p = m.p;
		}

		std::vector<cv::Vec3d> Y(d.samples.size());
		std::vector<char> clipped(d.samples.size(), 0);
		m.sweepFloor = d.baselineLed;
		if (d.useRaw)
		{
			// LEDs measured on RAW: no ISP clip, no tone-curve shoulder. Only a
			// saturated SENSOR excludes a sample.
			if (!fitRawToLin(d, m))
				return false;
			cv::Vec3d baseRaw = d.baselineRaw;
			if (!d.baselineHasRaw)
				for (const auto &s : d.samples)
					if (s.drive[0] == 0 && s.drive[1] == 0 && s.drive[2] == 0)
						baseRaw = s.raw;
			m.baselineLin = m.rawToLin * baseRaw;
			for (std::size_t i = 0; i < d.samples.size(); i++)
			{
				const SweepSample &s = d.samples[i];
				Y[i] = m.rawToLin * s.raw - m.baselineLin;
				for (int c = 0; c < 3; c++)
					if (s.rawSat[c] > 0.002)
					{
						clipped[i] = 1;
						m.clipExcluded[c]++;
					}
			}
			for (int c = 0; c < 3; c++)
				if (m.clipExcluded[c] > 0)
					std::cout << "WARNING: LED channel " << "rgb"[c] << " saturates the SENSOR in " << m.clipExcluded[c]
							  << " sweep samples (excluded). Lower --gain to calibrate its full range." << std::endl;
		}
		else
		{
			m.baselineLin = m.cam.lin3(d.baselineLed);
			for (std::size_t i = 0; i < d.samples.size(); i++)
				Y[i] = m.cam.lin3(d.samples[i].led) - m.baselineLin;

			// Sensor-clip detection: a channel saturating the camera at high drive reads
			// lower than the true light, biasing the curve top and the e(255)=1
			// normalization. Drop those samples from the fit; the curve then holds flat
			// above the last responsive drive (conservative). Per channel, since WB makes
			// R/G/B saturate at different encoded values.
			detectClipCeilings(d.samples, m.clipCeil, clipped);
			for (std::size_t i = 0; i < d.samples.size(); i++)
				if (clipped[i])
					for (int c = 0; c < 3; c++)
						if (m.clipCeil[c] > 0 && d.samples[i].led[c] >= m.clipCeil[c])
							m.clipExcluded[c]++;
			for (int c = 0; c < 3; c++)
				if (m.clipCeil[c] > 0)
					std::cout << "WARNING: LED channel " << "rgb"[c] << " saturates the camera (reading >= "
							  << m.clipCeil[c] << ") in " << m.clipExcluded[c] << " sweep samples; excluded from "
							  << "the fit (curve holds flat above the last responsive drive). Lower exposure or "
							  << "LED brightness on this channel to calibrate its full range." << std::endl;
		}

		// Initial A: columns from the strongest single-channel samples.
		cv::Matx33d A = cv::Matx33d::zeros();
		for (int c = 0; c < 3; c++)
		{
			double bestLvl = -1;
			cv::Vec3d bestY{0, 0, 0};
			for (std::size_t i = 0; i < d.samples.size(); i++)
			{
				if (clipped[i])
					continue;
				const auto &s = d.samples[i];
				const int onOther = (s.drive[(c + 1) % 3] > 0) + (s.drive[(c + 2) % 3] > 0);
				if (onOther == 0 && s.drive[c] > bestLvl)
				{
					bestLvl = s.drive[c];
					bestY = Y[i];
				}
			}
			if (bestLvl <= 0)
			{
				std::cout << "ERROR: led_sweep has no single-channel samples for channel " << c << std::endl;
				return false;
			}
			for (int r = 0; r < 3; r++)
				A(r, c) = bestY[r];
		}

		// Per-channel level tables (all unique drive levels in the sweep).
		for (int c = 0; c < 3; c++)
		{
			std::vector<double> lv;
			for (const auto &s : d.samples)
				lv.push_back(s.drive[c]);
			std::sort(lv.begin(), lv.end());
			lv.erase(std::unique(lv.begin(), lv.end()), lv.end());
			m.e[c].level = lv;
			m.e[c].value.assign(lv.size(), 0.0);
		}

		// ALS: z = A^-1 Y gives per-sample curve estimates; bucket by level,
		// then refit A against the tabulated curves.
		for (int iter = 0; iter < 6; iter++)
		{
			cv::Matx33d Ainv;
			if (cv::invert(A, Ainv, cv::DECOMP_LU) == 0)
			{
				std::cout << "ERROR: mixing matrix is singular (LED channels indistinguishable "
							 "at the camera)" << std::endl;
				return false;
			}

			for (int c = 0; c < 3; c++)
			{
				std::vector<double> sum(m.e[c].level.size(), 0.0);
				std::vector<int> cnt(m.e[c].level.size(), 0);
				for (std::size_t i = 0; i < d.samples.size(); i++)
				{
					if (clipped[i])
						continue;
					const cv::Vec3d z = Ainv * Y[i];
					const auto &lv = m.e[c].level;
					const auto it = std::lower_bound(lv.begin(), lv.end(), d.samples[i].drive[c]);
					if (it == lv.end() || *it != d.samples[i].drive[c])
						continue;
					const std::size_t k = static_cast<std::size_t>(it - lv.begin());
					sum[k] += z[c];
					cnt[k]++;
				}
				for (std::size_t k = 0; k < sum.size(); k++)
					if (cnt[k] > 0)
						m.e[c].value[k] = sum[k] / cnt[k];
				if (!m.e[c].level.empty() && m.e[c].level.front() == 0.0)
					m.e[c].value.front() = 0.0; // drive 0 -> LED dark
				monotonize(m.e[c]);
			}

			// Refit A: normal equations over all samples.
			cv::Matx33d YE = cv::Matx33d::zeros(), EE = cv::Matx33d::zeros();
			for (std::size_t i = 0; i < d.samples.size(); i++)
			{
				if (clipped[i])
					continue;
				cv::Vec3d e;
				for (int c = 0; c < 3; c++)
					e[c] = m.e[c].eval(d.samples[i].drive[c]);
				for (int r = 0; r < 3; r++)
					for (int c = 0; c < 3; c++)
					{
						YE(r, c) += Y[i][r] * e[c];
						EE(r, c) += e[r] * e[c];
					}
			}
			cv::Matx33d EEinv;
			if (cv::invert(EE, EEinv, cv::DECOMP_LU) == 0)
				break; // degenerate curves; keep previous A
			A = YE * EEinv;
		}

		// Regularize the unreliable dark end of each curve (camera floor noise)
		// with a smooth power law before normalization - removes the plateau-driven
		// crush + colour cast. Runs after the ALS so it reshapes only the bottom of
		// the converged curves; e(255) is untouched so the normalization scale below
		// (and A's columns) are unaffected.
		for (int c = 0; c < 3; c++)
		{
			const auto [sStar, gamma] = regularizeCurveBottom(m.e[c], curveFloor);
			if (verbose && gamma >= 0.0)
				std::cout << "Curve " << "rgb"[c] << ": regularized below drive " << sStar
						  << " (power-law gamma=" << gamma << ")" << std::endl;
		}

		// Normalize curves to e(255)=1 and fold the scale into A's columns.
		for (int c = 0; c < 3; c++)
		{
			const double s = m.e[c].value.empty() ? 0.0 : m.e[c].value.back();
			if (s <= 0.0)
			{
				std::cout << "ERROR: channel " << c << " produced no light in the sweep" << std::endl;
				return false;
			}
			for (auto &v : m.e[c].value)
				v /= s;
			for (int r = 0; r < 3; r++)
				A(r, c) *= s;
		}

		m.A = A;
		if (cv::invert(A, m.Ainv, cv::DECOMP_LU) == 0)
		{
			std::cout << "ERROR: mixing matrix is singular" << std::endl;
			return false;
		}

		// Fit residuals (relative, per sample).
		std::vector<double> rel;
		for (std::size_t i = 0; i < d.samples.size(); i++)
		{
			if (clipped[i])
				continue;
			cv::Vec3d e;
			for (int c = 0; c < 3; c++)
				e[c] = m.e[c].eval(d.samples[i].drive[c]);
			const cv::Vec3d pred = m.A * e;
			const cv::Vec3d diff = pred - Y[i];
			const double n = std::sqrt(Y[i].dot(Y[i]));
			if (n > 1e-6)
				rel.push_back(std::sqrt(diff.dot(diff)) / n);
		}
		std::sort(rel.begin(), rel.end());
		if (!rel.empty())
		{
			m.residMedian = rel[rel.size() / 2];
			m.residP95 = rel[static_cast<std::size_t>(static_cast<double>(rel.size() - 1) * 0.95)];
		}

		// Anchor: gains act in linear space as g^p. They were computed from the
		// TOTAL anchor reading (glow + ambient floor) and the viewer also sees
		// total light, so the renorm pins g^p * (A*e(S_w) + ambient) to the
		// eye-matched white. The inversion subtracts the ambient floor from
		// each target accordingly.
		cv::Vec3d corrLin; // fallback only (zero modelled white): encoded gains -> linear, ~power law
		for (int c = 0; c < 3; c++)
			corrLin[c] = std::pow(std::max(d.gains[c], 1e-6), m.cam.measured ? 2.2 : m.p);
		cv::Vec3d eW;
		for (int c = 0; c < 3; c++)
			eW[c] = m.e[c].eval(d.anchorDrive[c]);
		const cv::Vec3d totalW = m.A * eW + m.baselineLin;
		const cv::Vec3d tW = m.cam.lin3(d.anchorTv);
		for (int c = 0; c < 3; c++)
			m.corr[c] = (totalW[c] > 1e-12) ? tW[c] / totalW[c] : corrLin[c];

		// A channel whose modelled LED white reads near the camera floor (a
		// severely WB-attenuated channel) makes corr = tW/totalW explode and
		// crushes that channel across the LUT (colour cast). The fit can't repair
		// it - no anchor in the data reproduces that channel - so flag it loudly
		// instead of shipping a silent cast. Deferred "real" fix: clamp corr (see
		// CLAUDE.md, "Deferred fix - corr clamp").
		const double maxTotalW = std::max({totalW[0], totalW[1], totalW[2]});
		for (int c = 0; c < 3; c++)
			if (maxTotalW > 1e-9 && totalW[c] < 0.02 * maxTotalW)
				std::cout << "WARNING: LED white channel " << "rgb"[c]
						  << " reads near the camera floor (modelled white " << totalW[c]
						  << " vs brightest " << maxTotalW << ", corr=" << m.corr[c]
						  << "); the noise-dominated white correction will crush it across the LUT."
						  << " Reduce its HyperHDR WB attenuation or raise LED brightness, then recalibrate."
						  << std::endl;

		if (verbose)
		{
			std::cout << "Fit: p=" << m.p << " additivityResid=" << m.additivityResid
					  << " rel resid median=" << m.residMedian << " p95=" << m.residP95 << std::endl;
			std::cout << "A = " << cv::Mat(m.A) << std::endl;
		}
		return true;
	}

	// ---------- inversion ----------

	// Target LED light for LUT input vEnc: total corrected light minus the
	// ambient floor (the LEDs only control the part above it).
	cv::Vec3d nodeTarget(const FitModel &m, const cv::Vec3d &vEnc)
	{
		cv::Vec3d t = m.cam.lin3(vEnc);
		for (int c = 0; c < 3; c++)
			t[c] = t[c] / std::max(m.corr[c], 1e-12) - m.baselineLin[c];
		return t;
	}

	// Soft knee on the required max-drive fraction x: identity up to kneeStart,
	// then 1-exp roll-off with slope 1 at the start (C1, monotone) reaching 1 at
	// kneeMax (= what the brightest LUT input, camera white, needs).
	double kneeMap(const FitModel &m, double x)
	{
		if (!m.kneeOn || x <= m.kneeStart)
			return x;
		const double k0 = m.kneeStart, D = m.kneeMax - k0;
		return k0 + (1.0 - k0) * (1.0 - std::exp(-m.kneeA * (x - k0))) / (1.0 - std::exp(-m.kneeA * D));
	}

	// Required drive fractions e for input vEnc after the hue-true gamut mapping
	// (desaturation of negative components) — before knee/overflow handling.
	uint8_t solveE(const FitModel &m, const cv::Vec3d &vEnc, cv::Vec3d &e)
	{
		uint8_t flag = NodeInGamut;
		const cv::Vec3d t = nodeTarget(m, vEnc);
		e = m.Ainv * t;

		// Hue-true gamut mapping. Negative components: desaturate towards the
		// luminance-matched gray (linear blend keeps hue direction).
		if (e[0] < 0 || e[1] < 0 || e[2] < 0)
		{
			const cv::Vec3d lumW{0.299, 0.587, 0.114};
			const cv::Vec3d white = m.A * cv::Vec3d{1, 1, 1};
			const double lumE = (m.A * e).dot(lumW);
			const double lumWhite = white.dot(lumW);
			const double g = (lumWhite > 1e-12) ? lumE / lumWhite : 0.0;
			if (g <= 0.0)
			{
				e = {0, 0, 0};
				return NodeBlack;
			}
			const cv::Vec3d eGray{g, g, g};
			double beta = 1.0;
			for (int c = 0; c < 3; c++)
			{
				if (e[c] < 0.0)
					beta = std::min(beta, eGray[c] / (eGray[c] - e[c]));
			}
			e = beta * e + (1.0 - beta) * eGray;
			for (int c = 0; c < 3; c++)
				e[c] = std::max(e[c], 0.0);
			flag |= NodeDesaturated;
		}
		return flag;
	}

	uint8_t invertNode(const FitModel &m, const cv::Vec3d &vEnc, cv::Vec3d &driveOut)
	{
		cv::Vec3d e;
		uint8_t flag = solveE(m, vEnc, e);
		if (flag & NodeBlack)
		{
			driveOut = {0, 0, 0};
			return flag;
		}

		// Soft knee: compress brightness (all channels by one factor, so hue is
		// kept) where the LEDs would otherwise run out of headroom. The factor is
		// driven by the output LUMINANCE, expressed as the max-drive fraction white
		// would need at that luminance (m*): identical to a max-channel knee on the
		// gray axis (white still reaches full output), but smooth across hue, so
		// trilinear interpolation between gray nodes and their slightly tinted
		// neighbours doesn't dip in brightness near white. Saturated colours that
		// still exceed a channel fall to the overflow clip below.
		if (m.kneeOn)
		{
			const cv::Vec3d lumW{0.299, 0.587, 0.114};
			const double mStar = (m.A * e).dot(lumW) / m.kneeLumWhite * m.kneeMax;
			if (mStar > m.kneeStart)
			{
				e *= kneeMap(m, mStar) / mStar;
				flag |= NodeKnee;
			}
		}

		// Overflow. Saturated colours: scale down (keeps chromaticity, reduces
		// luminance). Near-neutral colours: keep the luminance and give up a little
		// saturation toward the eye-matched white instead — otherwise a slightly
		// tinted input next to white is dimmed as a whole and trilinear
		// interpolation between it and the gray nodes dips the gray axis' brightness
		// (~4.5% near white with the plain clip). Blended by saturation, continuous.
		const double mx = std::max({e[0], e[1], e[2]});
		if (mx > 1.0)
		{
			const cv::Vec3d eLum = e * (1.0 / mx);
			cv::Vec3d out = eLum;
			const cv::Vec3d &w = m.whiteDir;
			if (w[0] > 0 && w[1] > 0 && w[2] > 0)
			{
				const cv::Vec3d lumW{0.299, 0.587, 0.114};
				cv::Vec3d gray = w * ((m.A * e).dot(lumW) / (m.A * w).dot(lumW));
				const double gm = std::max({gray[0], gray[1], gray[2]});
				if (gm > 1.0)
					gray *= 1.0 / gm; // brighter than full white: the gray itself is dimmed
				double beta = 1.0;
				for (int c = 0; c < 3; c++)
					if (e[c] > 1.0 && e[c] > gray[c])
						beta = std::min(beta, (1.0 - gray[c]) / (e[c] - gray[c]));
				const cv::Vec3d eChroma = beta * e + (1.0 - beta) * gray;
				double rmin = 1e18, rmax = 0.0;
				for (int c = 0; c < 3; c++)
				{
					rmin = std::min(rmin, e[c] / w[c]);
					rmax = std::max(rmax, e[c] / w[c]);
				}
				const double sat = rmax > 0.0 ? 1.0 - rmin / rmax : 1.0; // 0 = eye-matched white hue
				const double u = std::clamp((sat - 0.1) / 0.2, 0.0, 1.0);
				const double wc = 1.0 - u * u * (3.0 - 2.0 * u); // 1 below 10% saturation, 0 above 30%
				out = wc * eChroma + (1.0 - wc) * eLum;
			}
			e = out;
			flag |= NodeLumClipped;
		}

		for (int c = 0; c < 3; c++)
			driveOut[c] = std::clamp(m.e[c].invert(e[c]), 0.0, 255.0);
		return flag;
	}

	// Equal R=G=B drive (HyperHDR's eye-matched white) whose modelled LED light
	// has the luminance of target t.
	double neutralDrive(const FitModel &m, const cv::Vec3d &t)
	{
		const cv::Vec3d lumW{0.299, 0.587, 0.114};
		const double yt = t.dot(lumW);
		if (yt <= 0.0)
			return 0.0;
		auto yAt = [&](double d)
		{ return (m.A * cv::Vec3d{m.e[0].eval(d), m.e[1].eval(d), m.e[2].eval(d)}).dot(lumW); };
		if (yt >= yAt(255.0))
			return 255.0;
		double lo = 0.0, hi = 255.0;
		for (int i = 0; i < 40; i++)
		{
			const double mid = 0.5 * (lo + hi);
			(yAt(mid) < yt ? lo : hi) = mid;
		}
		return 0.5 * (lo + hi);
	}

	void buildLut(const FitModel &m, FitResult &res)
	{
		const int n = res.n;
		res.lut.assign(static_cast<std::size_t>(n) * n * n, cv::Vec3d{0, 0, 0});
		res.flags.assign(res.lut.size(), 0);
		std::size_t idx = 0;
		int desat = 0, clip = 0, knee = 0, dark = 0;
		for (int b = 0; b < n; b++)
		{
			for (int g = 0; g < n; g++)
			{
				for (int r = 0; r < n; r++, idx++)
				{
					const cv::Vec3d v{255.0 * r / (n - 1), 255.0 * g / (n - 1), 255.0 * b / (n - 1)};
					res.flags[idx] = invertNode(m, v, res.lut[idx]);
					// Dark fallback: below the floor the measurements can't resolve
					// colour; fade to the neutral equal drive at the same luminance
					// (full neutral below half the floor, smoothstep up to the floor).
					const double luma = 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2];
					if (m.darkFloor > 0.0 && luma < m.darkFloor)
					{
						const double u = std::clamp((luma - 0.5 * m.darkFloor) / (0.5 * m.darkFloor), 0.0, 1.0);
						const double w = u * u * (3.0 - 2.0 * u);
						const double dN = neutralDrive(m, nodeTarget(m, v));
						res.lut[idx] = w * res.lut[idx] + (1.0 - w) * cv::Vec3d{dN, dN, dN};
						res.flags[idx] |= NodeDark;
					}
					if (res.flags[idx] & NodeDesaturated)
						desat++;
					if (res.flags[idx] & NodeLumClipped)
						clip++;
					if (res.flags[idx] & NodeKnee)
						knee++;
					if (res.flags[idx] & NodeDark)
						dark++;
				}
			}
		}
		const double N = static_cast<double>(res.lut.size());
		res.pctDesat = 100.0 * desat / N;
		res.pctClip = 100.0 * clip / N;
		res.pctKnee = 100.0 * knee / N;
		res.pctDark = 100.0 * dark / N;
	}

	// Eye-matched white direction (camera white's drive fractions, max 1).
	void setupWhiteDir(FitModel &m)
	{
		cv::Vec3d e;
		m.whiteDir = {0, 0, 0};
		if (solveE(m, cv::Vec3d{255, 255, 255}, e) & NodeBlack)
			return;
		const double mx = std::max({e[0], e[1], e[2]});
		if (mx > 0.0)
			m.whiteDir = e * (1.0 / mx);
	}

	// Knee: kneeMax = drive fraction camera white (LUT input 255,255,255) needs;
	// kneeA solves slope 1 at kneeStart. No headroom shortfall -> knee off.
	void setupKnee(FitModel &m, double startFrac)
	{
		m.kneeOn = false;
		if (startFrac <= 0.0)
			return;
		cv::Vec3d e;
		if (solveE(m, cv::Vec3d{255, 255, 255}, e) & NodeBlack)
			return;
		m.kneeStart = startFrac;
		m.kneeMax = std::max({e[0], e[1], e[2]});
		m.kneeLumWhite = (m.A * e).dot(cv::Vec3d{0.299, 0.587, 0.114});
		if (m.kneeLumWhite <= 0.0)
			return;
		const double D = m.kneeMax - m.kneeStart, room = 1.0 - m.kneeStart;
		if (D <= room * 1.0001)
		{
			std::cout << "Knee: not needed (camera white needs " << m.kneeMax << " of full LED output)" << std::endl;
			return;
		}
		// (1-k0)*a = 1 - exp(-a*D): f(a) < 0 just above 0, > 0 for large a.
		double lo = 1e-9, hi = 1.0;
		auto f = [&](double a) { return room * a - (1.0 - std::exp(-a * D)); };
		while (f(hi) < 0.0 && hi < 1e6)
			hi *= 2.0;
		for (int i = 0; i < 200; i++)
		{
			const double mid = 0.5 * (lo + hi);
			(f(mid) < 0.0 ? lo : hi) = mid;
		}
		m.kneeA = 0.5 * (lo + hi);
		m.kneeOn = true;
		std::cout << "Knee: camera white needs " << m.kneeMax << "x full LED output; brightness compressed above "
				  << (100.0 * m.kneeStart) << "% so white reaches full output (hue kept)" << std::endl;
	}

	// Automatic dark floor (encoded camera luma): one LUT grid step (8 codes)
	// above the TV's black reading. Below it a dark colour spans only a few
	// camera codes, so a 10-20% chroma difference is under one code - the
	// measurement can't resolve colour there (and the LUT has no node in
	// between). (The ROI's spatial std is NOT a usable noise measure: on the real
	// rig it is ~20% of the mean at every level - panel structure, not noise.)
	double autoDarkFloor(const CalibData &d)
	{
		const double black = d.tvBlackLuma >= 0.0 ? d.tvBlackLuma : 0.0;
		return std::clamp(black + 8.0, 4.0, 24.0);
	}

	// ---------- output ----------

	bool writeCube(const std::string &path, const FitResult &res, const std::string &srcDir)
	{
		std::ofstream f(path);
		if (!f)
		{
			std::cout << "ERROR: cannot write " << path << std::endl;
			return false;
		}
		char ts[64] = "";
		const std::time_t now = std::time(nullptr);
		std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
		f << "# hyperion-cam LED calibration LUT (doc/led-lut-calibration.md)\n";
		f << "# source: " << srcDir << "  generated: " << ts << "\n";
		f << "# fit: p=" << res.model.p << " relResidMedian=" << res.model.residMedian
		  << " relResidP95=" << res.model.residP95 << "\n";
		f << "# gamut: desaturated=" << res.pctDesat << "% lumClipped=" << res.pctClip << "%\n";
		f << "# sensor-clip ceilings (reading, 0=none): R=" << res.model.clipCeil[0]
		  << " G=" << res.model.clipCeil[1] << " B=" << res.model.clipCeil[2] << "\n";
		f << "LUT_3D_SIZE " << res.n << "\n";
		f << "DOMAIN_MIN 0.0 0.0 0.0\n";
		f << "DOMAIN_MAX 1.0 1.0 1.0\n";
		char line[96];
		for (const auto &s : res.lut)
		{
			std::snprintf(line, sizeof(line), "%.6f %.6f %.6f\n", s[0] / 255.0, s[1] / 255.0, s[2] / 255.0);
			f << line;
		}
		return true;
	}

	void writeReport(const std::string &dir, const FitResult &res)
	{
		std::ofstream f(dir + "/fit_report.json");
		if (!f)
		{
			std::cout << "WARNING: cannot write " << dir << "/fit_report.json" << std::endl;
			return;
		}
		const FitModel &m = res.model;
		f << "{\n";
		f << "  \"camera_model\": \"" << (m.cam.measured ? "measured" : "power") << "\",\n";
		if (m.cam.measured)
		{
			f << "  \"response_rms\": " << m.cam.resp.rmsResid << ",\n";
			f << "  \"response_worst_series\": " << m.cam.resp.worstSeriesResid << ",\n";
			f << "  \"response_lin\": [";
			for (int z = 0; z <= 255; z += 8)
				f << m.cam.resp.eval(z) << ", ";
			f << m.cam.resp.eval(255) << "],\n";
		}
		f << "  \"raw_led\": " << (m.rawLed ? "true" : "false") << ",\n";
		if (m.rawLed)
		{
			f << "  \"raw_to_lin\": [";
			for (int r = 0; r < 3; r++)
				f << "[" << m.rawToLin(r, 0) << ", " << m.rawToLin(r, 1) << ", " << m.rawToLin(r, 2) << "]" << (r < 2 ? ", " : "");
			f << "],\n  \"raw_map_resid\": " << m.rawMapResid << ",\n";
		}
		f << "  \"knee\": {\"on\": " << (m.kneeOn ? "true" : "false") << ", \"start\": " << m.kneeStart
		  << ", \"white_needs\": " << m.kneeMax << ", \"a\": " << m.kneeA << "},\n";
		f << "  \"dark_floor\": " << m.darkFloor << ",\n";
		f << "  \"pct_knee\": " << res.pctKnee << ",\n";
		f << "  \"pct_dark\": " << res.pctDark << ",\n";
		f << "  \"gray_axis\": [";
		for (std::size_t i = 0; i < res.grayAxis.size(); i++)
		{
			const auto &g = res.grayAxis[i];
			f << (i ? ", " : "") << "{\"tv_level\": " << g[0] << ", \"reading\": " << g[1] << ", \"drive\": [" << g[2]
			  << ", " << g[3] << ", " << g[4] << "]}";
		}
		f << "],\n";
		f << "  \"p\": " << m.p << ",\n";
		f << "  \"additivity_resid\": " << m.additivityResid << ",\n";
		f << "  \"rel_resid_median\": " << m.residMedian << ",\n";
		f << "  \"rel_resid_p95\": " << m.residP95 << ",\n";
		f << "  \"pct_desaturated\": " << res.pctDesat << ",\n";
		f << "  \"pct_lum_clipped\": " << res.pctClip << ",\n";
		f << "  \"A\": [";
		for (int r = 0; r < 3; r++)
		{
			f << "[" << m.A(r, 0) << ", " << m.A(r, 1) << ", " << m.A(r, 2) << "]" << (r < 2 ? ", " : "");
		}
		f << "],\n";
		f << "  \"corr\": [" << m.corr[0] << ", " << m.corr[1] << ", " << m.corr[2] << "],\n";
		f << "  \"clip_ceiling\": [" << m.clipCeil[0] << ", " << m.clipCeil[1] << ", " << m.clipCeil[2] << "],\n";
		f << "  \"clip_excluded\": [" << m.clipExcluded[0] << ", " << m.clipExcluded[1] << ", " << m.clipExcluded[2] << "],\n";
		f << "  \"sweep_floor\": [" << m.sweepFloor[0] << ", " << m.sweepFloor[1] << ", " << m.sweepFloor[2] << "],\n";
		for (int c = 0; c < 3; c++)
		{
			f << "  \"curve_" << "rgb"[c] << "\": {\"level\": [";
			for (std::size_t i = 0; i < m.e[c].level.size(); i++)
				f << m.e[c].level[i] << (i + 1 < m.e[c].level.size() ? ", " : "");
			f << "], \"value\": [";
			for (std::size_t i = 0; i < m.e[c].value.size(); i++)
				f << m.e[c].value[i] << (i + 1 < m.e[c].value.size() ? ", " : "");
			f << "]}" << (c < 2 ? ",\n" : "\n");
		}
		f << "}\n";
		std::cout << "Wrote " << dir << "/fit_report.json" << std::endl;
	}

	// "auto" | "off" | number; false + message on anything else.
	bool parseSwitch(const std::string &v, const char *name, double lo, double hi, double &num, bool &isAuto, bool &isOff)
	{
		isAuto = (v == "auto");
		isOff = (v == "off");
		if (isAuto || isOff)
			return true;
		char *end = nullptr;
		num = std::strtod(v.c_str(), &end);
		if (end && !*end && num >= lo && num <= hi)
			return true;
		std::cout << "ERROR: " << name << " must be auto, off or a number in [" << lo << "," << hi << "]" << std::endl;
		return false;
	}

	bool runFitCore(const LedCalibFitConfig &cfg, const CalibData &data, FitResult &res)
	{
		double kneePct = 0, floorV = 0;
		bool kAuto, kOff, fAuto, fOff;
		if (!parseSwitch(cfg.knee, "--calib-knee", 50.0, 99.0, kneePct, kAuto, kOff) ||
			!parseSwitch(cfg.darkFloor, "--calib-dark-floor", 0.0, 128.0, floorV, fAuto, fOff))
			return false;
		if (!fitModel(data, res.model, cfg.verbose, cfg.curveFloor))
			return false;
		setupWhiteDir(res.model);
		setupKnee(res.model, kOff ? 0.0 : (kAuto ? 0.85 : kneePct / 100.0));
		res.model.darkFloor = fOff ? 0.0 : (fAuto ? autoDarkFloor(data) : floorV);
		if (res.model.darkFloor > 0.0)
			std::cout << "Dark floor: camera luma < " << res.model.darkFloor << (fAuto ? " (auto: TV black + 8 codes)" : "")
					  << " fades to the neutral equal drive" << std::endl;
		res.n = std::clamp(cfg.lutSize, 2, 65);
		buildLut(res.model, res);
		return true;
	}

	// ---------- self-test ----------

	struct TruthChain
	{
		// Ground-truth LED/wall/camera chain, deliberately different from the
		// model's parametrization defaults. The camera encodes with the STOCK
		// imx708 tone curve (local exponent 1.6..4.2), not a power law: on it the
		// legacy single-exponent model reproduces the real rig's gray drift away
		// from the white anchor; the measured-response model must not.
		cv::Matx33d A = cv::Matx33d{0.72, 0.10, 0.034, 0.068, 0.66, 0.085, 0.026, 0.13, 0.68} * 1.15;
		double exps[3] = {2.0, 1.9, 2.1};
		cv::Vec3d ambient{0.004, 0.005, 0.006};
		// Sensor: camera-linear light -> black-subtracted RAW counts (the ISP's
		// colour matrix/WB inverse); clips at 1023 only above ~1.7x the ISP clip.
		cv::Matx33d rawM{380, 95, 45, 110, 470, 95, 28, 140, 420};
		// TV gray light per channel (camera-linear): dimmer than the LEDs, own 2.2 EOTF.
		cv::Vec3d tvWhite{0.55, 0.60, 0.58};
		cv::Vec3d tvAmbient{0.001, 0.001, 0.0012};

		static double curve(double x, bool inverse)
		{
			static const double k[66] = {0, 0, 1024, 5040, 2048, 9338, 3072, 12356, 4096, 15312, 5120, 18051, 6144,
										 20790, 7168, 23193, 8192, 25744, 9216, 27942, 10240, 30035, 11264, 32005,
										 12288, 33975, 13312, 35815, 14336, 37600, 15360, 39168, 16384, 40642, 18432,
										 43379, 20480, 45749, 22528, 47753, 24576, 49621, 26624, 51253, 28672, 52698,
										 30720, 53796, 32768, 54876, 36864, 57012, 40960, 58656, 45056, 59954, 49152,
										 61183, 53248, 62355, 57344, 63419, 61440, 64476, 65535, 65535};
			const int a = inverse ? 1 : 0, b = inverse ? 0 : 1; // interpolate b as a function of a
			x = std::clamp(x, 0.0, 1.0) * 65535.0;
			for (int i = 1; i < 33; i++)
				if (k[2 * i + a] >= x)
				{
					const double x0 = k[2 * (i - 1) + a], x1 = k[2 * i + a];
					const double y0 = k[2 * (i - 1) + b], y1 = k[2 * i + b];
					return (y0 + (x1 > x0 ? (x - x0) / (x1 - x0) : 0.0) * (y1 - y0)) / 65535.0;
				}
			return 1.0;
		}

		cv::Vec3d glowLin(const cv::Vec3d &drive) const
		{
			cv::Vec3d e;
			for (int c = 0; c < 3; c++)
				e[c] = std::pow(std::clamp(drive[c], 0.0, 255.0) / 255.0, exps[c]);
			return A * e + ambient;
		}
		cv::Vec3d encode(const cv::Vec3d &lin) const
		{
			return {255.0 * curve(lin[0], false), 255.0 * curve(lin[1], false), 255.0 * curve(lin[2], false)};
		}
		double linOf(double enc) const { return curve(enc / 255.0, true); }
		cv::Vec3d reading(const cv::Vec3d &drive) const { return encode(glowLin(drive)); }
		cv::Vec3d rawOf(const cv::Vec3d &lin) const { return rawM * lin; }
		cv::Vec3d tvLin(double L) const
		{
			const double f = std::pow(std::clamp(L, 0.0, 255.0) / 255.0, 2.2);
			return tvWhite * f + tvAmbient;
		}
	};

	// Edge-dense level (mirrors ledCalib.cpp edgeLevel) for the synthetic axis ramp.
	double edgeLevelSelftest(int i, int n)
	{
		const double pi = std::acos(-1.0);
		return std::lround(255.0 * 0.5 * (1.0 - std::cos(pi * i / (n - 1))));
	}

	cv::Vec3d sampleLutTrilinear(const FitResult &res, const cv::Vec3d &vEnc)
	{
		const int n = res.n;
		auto at = [&](int r, int g, int b) -> const cv::Vec3d &
		{
			return res.lut[(static_cast<std::size_t>(b) * n + g) * n + r];
		};
		double f[3];
		int i0[3];
		for (int c = 0; c < 3; c++)
		{
			const double x = std::clamp(vEnc[c], 0.0, 255.0) / 255.0 * (n - 1);
			i0[c] = std::min(static_cast<int>(x), n - 2);
			f[c] = x - i0[c];
		}
		cv::Vec3d out{0, 0, 0};
		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++)
				{
					const double w = (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]);
					out += w * at(i0[0] + dx, i0[1] + dy, i0[2] + dz);
				}
		return out;
	}

	bool nodeUsable(const FitResult &res, const cv::Vec3d &vEnc)
	{
		// All 8 surrounding nodes strictly in gamut -> interpolated point is a
		// fair accuracy test (clipped regions are policy, not error).
		const int n = res.n;
		int i0[3];
		for (int c = 0; c < 3; c++)
		{
			const double x = std::clamp(vEnc[c], 0.0, 255.0) / 255.0 * (n - 1);
			i0[c] = std::min(static_cast<int>(x), n - 2);
		}
		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++)
				{
					const std::size_t idx = (static_cast<std::size_t>(i0[2] + dz) * n + (i0[1] + dy)) * n + (i0[0] + dx);
					if (res.flags[idx] != NodeInGamut)
						return false;
				}
		return true;
	}

} // namespace

int runLedCalibFit(const LedCalibFitConfig &cfg)
{
	if (cfg.response != "auto" && cfg.response != "on" && cfg.response != "off")
	{
		std::cout << "ERROR: --calib-response must be auto, on or off" << std::endl;
		return EXIT_FAILURE;
	}
	CalibData data;
	if (!loadCalibData(cfg.calibDir, data, cfg.clipLevel, cfg.whiteAnchorLevel, cfg.response))
		return EXIT_FAILURE;
	std::cout << "Loaded " << data.samples.size() << " LED sweep samples from " << cfg.calibDir
			  << (data.useRaw ? " (LEDs measured on RAW)" : "") << std::endl;

	FitResult res;
	if (!runFitCore(cfg, data, res))
		return EXIT_FAILURE;

	std::cout << "Fit: camera " << (res.model.cam.measured ? "measured response" : "power law p=" + std::to_string(res.model.p))
			  << ", rel resid median=" << res.model.residMedian << " p95=" << res.model.residP95
			  << "; gamut: " << res.pctDesat << "% desaturated, " << res.pctClip << "% luminance-clipped, "
			  << res.pctKnee << "% knee, " << res.pctDark << "% dark-blended" << std::endl;

	// Gray-axis diagnostics: what the LUT drives for each measured TV gray.
	{
		std::vector<std::pair<double, cv::Vec3d>> tvN;
		if (loadTvNeutrals(cfg.calibDir + "/tv_sweep.csv", tvN))
		{
			std::sort(tvN.begin(), tvN.end(), [](auto &a, auto &b) { return a.first < b.first; });
			for (const auto &[L, tv] : tvN)
			{
				const cv::Vec3d d = sampleLutTrilinear(res, tv);
				res.grayAxis.push_back({L, tv[1], d[0], d[1], d[2]});
			}
		}
	}
	if (res.model.residP95 > 0.10)
		std::cout << "WARNING: fit residual p95 > 10% - measurement noisy or model poor; "
					 "inspect fit_report.json before trusting the LUT" << std::endl;

	const std::string lutPath = cfg.lutOut.empty() ? cfg.calibDir + "/lut.cube" : cfg.lutOut;
	if (!writeCube(lutPath, res, cfg.calibDir))
		return EXIT_FAILURE;
	std::cout << "Wrote " << lutPath << " (" << res.n << "^3)" << std::endl;
	writeReport(cfg.calibDir, res);
	return EXIT_SUCCESS;
}

int runLedCalibFitSelftest(const LedCalibFitConfig &cfgIn)
{
	const TruthChain truth;
	std::mt19937 rng(12345);
	std::normal_distribution<double> noise(0.0, 0.3); // encoded-units measurement noise
	std::normal_distribution<double> rawNoise(0.0, 0.5);

	const std::string dir = "/tmp/ledcalib-selftest";
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	for (const char *f : {"meta.json", "meta_tv.json", "meta_led.json", "meta_response.json"})
		std::filesystem::remove(dir + "/" + f, ec);

	// One camera observation of linear light: ISP reading (stock curve, 8-bit,
	// clip at 1.0) and the RAW counts of the same light (sensor clip at 1023).
	struct Obs
	{
		cv::Vec3d isp, ispSat, raw, rawSat;
	};
	auto observe = [&](const cv::Vec3d &lin, double nIsp)
	{
		Obs o;
		o.isp = truth.encode(lin);
		o.raw = truth.rawOf(lin);
		for (int c = 0; c < 3; c++)
		{
			o.ispSat[c] = lin[c] >= 1.0 ? 1.0 : 0.0;
			o.isp[c] = std::clamp(o.isp[c] + nIsp * noise(rng), 0.0, 255.0);
			o.raw[c] = std::clamp(o.raw[c] + rawNoise(rng), -2.0, 1023.0 - 64.0);
			o.rawSat[c] = o.raw[c] >= 1022.0 - 64.0 ? 1.0 : 0.0;
		}
		return o;
	};

	// Synthetic LED sweep (dual-stream format): cube + gray ramp + axis ramps.
	{
		std::ofstream f(dir + "/led_sweep.csv");
		f << "drive_r,drive_g,drive_b,led_r,led_g,led_b,led_r_tstd,led_g_tstd,led_b_tstd,led_r_sstd,led_g_sstd,"
			 "led_b_sstd,frames,led_r_sat,led_g_sat,led_b_sat,raw_r,raw_g,raw_b,raw_r_tstd,raw_g_tstd,raw_b_tstd,"
			 "raw_r_sat,raw_g_sat,raw_b_sat,exposure_us,analogue_gain\n";
		auto emit = [&](int r, int g, int b)
		{
			const Obs o = observe(truth.glowLin({double(r), double(g), double(b)}), 1.0 / std::sqrt(20.0));
			f << r << "," << g << "," << b << "," << o.isp[0] << "," << o.isp[1] << "," << o.isp[2]
			  << ",0.3,0.3,0.3,1,1,1,20," << o.ispSat[0] << "," << o.ispSat[1] << "," << o.ispSat[2] << ","
			  << o.raw[0] << "," << o.raw[1] << "," << o.raw[2] << ",0.5,0.5,0.5," << o.rawSat[0] << ","
			  << o.rawSat[1] << "," << o.rawSat[2] << ",13600,1\n";
		};
		for (int r = 0; r < 8; r++)
			for (int g = 0; g < 8; g++)
				for (int b = 0; b < 8; b++)
					emit(r * 255 / 7, g * 255 / 7, b * 255 / 7);
		for (int i = 0; i < 17; i++)
			emit(i * 255 / 16, i * 255 / 16, i * 255 / 16);
		for (int i = 1; i < 24; i++)
		{
			const int v = static_cast<int>(edgeLevelSelftest(i, 24));
			emit(v, 0, 0);
			emit(0, v, 0);
			emit(0, 0, v);
		}
	}
	{
		std::ofstream f(dir + "/baseline.json");
		const cv::Vec3d b = truth.reading({0, 0, 0}), br = truth.rawOf(truth.glowLin({0, 0, 0}));
		f << "{\n  \"led_mean\": [" << b[0] << ", " << b[1] << ", " << b[2] << "],\n  \"led_raw_mean\": [" << br[0]
		  << ", " << br[1] << ", " << br[2] << "]\n}\n";
	}

	// TV gray sweep: the TV's own light (camera-linear) through the true curve;
	// per-pixel noise (sstd) for the automatic dark floor.
	{
		std::ofstream f(dir + "/tv_sweep.csv");
		f << "index,patch_r,patch_g,patch_b,tv_r,tv_g,tv_b,tv_r_tstd,tv_g_tstd,tv_b_tstd,tv_r_sstd,tv_g_sstd,"
			 "tv_b_sstd,frames\n";
		for (int i = 0; i < 17; i++)
		{
			const int L = i * 255 / 16;
			const Obs o = observe(truth.tvLin(L), 0.05);
			f << i << "," << L << "," << L << "," << L << "," << o.isp[0] << "," << o.isp[1] << "," << o.isp[2]
			  << ",0.1,0.1,0.1";
			for (int c = 0; c < 3; c++)
				f << "," << (0.6 + 0.012 * o.isp[c]);
			f << ",10\n";
		}
	}

	// Camera response shutter sweep over the LED white and a TV gray.
	{
		std::ofstream f(dir + "/camera_response.csv");
		f << "# synthetic\nstep,t_ms,target_exposure_us,exposure_us,analogue_gain,frames";
		for (const char *roi : {"led", "tv"})
			for (const char *what : {"isp", "isp_tstd", "isp_sat", "raw", "raw_tstd", "raw_sat"})
				for (const char *c : {"r", "g", "b"})
					f << "," << roi << "_" << what << "_" << c;
		f << "\n";
		const cv::Vec3d L0 = truth.glowLin({255, 255, 255}), T0 = truth.tvLin(180);
		int step = 0;
		for (int j = -24; j <= 8; j++)
		{
			const double k = std::pow(2.0, j / 4.0);
			f << step << "," << step * 100 << "," << std::lround(13600 * k) << "," << std::lround(13600 * k) << ",1,10";
			step++;
			for (const cv::Vec3d &L : {L0, T0})
			{
				const Obs o = observe(L * k, 0.3 / std::sqrt(10.0));
				const cv::Vec3d *cols[6] = {&o.isp, nullptr, &o.ispSat, &o.raw, nullptr, &o.rawSat};
				for (const cv::Vec3d *v : cols)
					for (int c = 0; c < 3; c++)
						f << "," << (v ? (*v)[c] : 0.1);
			}
			f << "\n";
		}
	}

	LedCalibFitConfig cfg = cfgIn;
	cfg.calibDir = dir;
	cfg.lutOut = dir + "/lut.cube";
	cfg.knee = "off";
	cfg.darkFloor = "off";

	// No meta.json: the white anchor falls to the default mid-gray target (127),
	// a measured neutral in both sweeps. Lower clip so the neutrals stay interior.
	CalibData data;
	if (!loadCalibData(cfg.calibDir, data, 200.0, 0, "auto") || !data.cam.measured || !data.useRaw)
	{
		std::cout << "SELFTEST FAIL: measured response / RAW LED path not selected" << std::endl;
		return EXIT_FAILURE;
	}

	// 1. Recovered tone curve vs truth (both normalised to lin(255) = 1).
	double respErr = 0.0;
	int respErrAt = 0;
	for (int z = 16; z <= 250; z += 2)
	{
		const double e = std::fabs(data.cam.resp.eval(z) / truth.linOf(z) - 1.0);
		if (e > respErr)
		{
			respErr = e;
			respErrAt = z;
		}
	}
	const bool respOk = respErr < 0.04;
	std::cout << "Selftest response recovery: max relative error " << (100.0 * respErr) << "% (at code " << respErrAt
			  << ") over codes 16..250 -> "
			  << (respOk ? "PASS" : "FAIL") << std::endl;

	const cv::Vec3d anchorTv = data.anchorTv;
	const cv::Vec3d anchorDrive{data.anchorDrive[0], data.anchorDrive[1], data.anchorDrive[2]};
	// True white correction: TV light / LED light at the anchor (the "everything
	// linear" world the LUT must reproduce at every level).
	const cv::Vec3d tvA = truth.tvLin(anchorDrive[0]), ledA = truth.glowLin(anchorDrive);
	const cv::Vec3d corrTrue{tvA[0] / ledA[0], tvA[1] / ledA[1], tvA[2] / ledA[2]};

	// Closed loop on the TRUE chain: input v -> LUT -> drive -> LED light; the
	// camera, after the true white correction, must read v back.
	auto closedLoop = [&](const FitResult &res, double &p95, double &mx, int &tested)
	{
		std::mt19937 r2(777);
		std::uniform_real_distribution<double> uni(0.0, 255.0);
		std::vector<double> errs;
		tested = 0;
		for (int i = 0; i < 4000; i++)
		{
			const cv::Vec3d v{uni(r2), uni(r2), uni(r2)};
			if (!nodeUsable(res, v))
				continue;
			const cv::Vec3d l = truth.glowLin(sampleLutTrilinear(res, v));
			const cv::Vec3d enc = truth.encode({corrTrue[0] * l[0], corrTrue[1] * l[1], corrTrue[2] * l[2]});
			for (int c = 0; c < 3; c++)
				errs.push_back(std::fabs(enc[c] - v[c]));
			tested++;
		}
		std::sort(errs.begin(), errs.end());
		p95 = errs.empty() ? 1e9 : errs[static_cast<std::size_t>(static_cast<double>(errs.size() - 1) * 0.95)];
		mx = errs.empty() ? 1e9 : errs.back();
	};

	// 2. New model: measured response + LEDs on RAW.
	FitResult res;
	if (!runFitCore(cfg, data, res))
		return EXIT_FAILURE;
	if (!writeCube(cfg.lutOut, res, dir))
		return EXIT_FAILURE;
	writeReport(dir, res);
	double p95 = 0, mx = 0;
	int tested = 0;
	closedLoop(res, p95, mx, tested);
	const cv::Vec3d sW = sampleLutTrilinear(res, anchorTv);
	double anchorErr = 0.0;
	for (int c = 0; c < 3; c++)
		anchorErr = std::max(anchorErr, std::fabs(sW[c] - anchorDrive[c]));
	std::cout << "Selftest closed loop, measured model (" << tested << " in-gamut pts): err p95=" << p95
			  << " max=" << mx << " (encoded units); anchor drive err=" << anchorErr << std::endl;

	// 3. Legacy power-law model on the same data: must be visibly worse (this
	// is the real rig's drift; proves the test exercises the failure).
	double lp95 = 0, lmx = 0;
	int ltested = 0;
	{
		CalibData legacy;
		FitResult lres;
		if (loadCalibData(cfg.calibDir, legacy, 200.0, 0, "off") && runFitCore(cfg, legacy, lres))
			closedLoop(lres, lp95, lmx, ltested);
	}
	const bool legacyWorse = lp95 > 2.0 * std::max(p95, 0.25);
	std::cout << "Selftest closed loop, legacy power-law model: err p95=" << lp95 << " max=" << lmx << " -> "
			  << (legacyWorse ? "worse, as on the real rig (test exercises the drift)" : "NOT worse (test too weak?)")
			  << std::endl;

	// 4. Knee + dark floor (policy): camera white reaches ~full output, the gray
	// axis stays monotone, dark nodes are neutral.
	bool kneeOk = false, darkOk = false;
	{
		LedCalibFitConfig c2 = cfg;
		c2.knee = "auto";
		c2.darkFloor = "auto";
		FitResult pres;
		if (runFitCore(c2, data, pres))
		{
			writeCube(dir + "/lut_policy.cube", pres, dir);
			const cv::Vec3d dW = sampleLutTrilinear(pres, {255, 255, 255});
			// Brightness must rise along the gray axis. Trilinear interpolation
			// between the diagonal nodes and their off-diagonal neighbours (which the
			// knee / gamut clip scales down by their brightest channel) can dip it
			// slightly between nodes near white: allow 2% linear, <= 2.5 codes/channel.
			bool mono = true;
			cv::Vec3d prev{-1, -1, -1};
			double prevLum = -1.0;
			for (int L = 0; L <= 255; L += 5)
			{
				const cv::Vec3d d = sampleLutTrilinear(pres, {double(L), double(L), double(L)});
				const cv::Vec3d lin = truth.glowLin(d);
				const double lum = 0.299 * lin[0] + 0.587 * lin[1] + 0.114 * lin[2];
				bool bad = lum < prevLum * 0.98;
				for (int c = 0; c < 3; c++)
					bad = bad || d[c] < prev[c] - 2.5;
				if (bad && mono)
					std::cout << "  gray axis not monotone at input " << L << ": drive " << prev[0] << "," << prev[1] << ","
							  << prev[2] << " -> " << d[0] << "," << d[1] << "," << d[2] << std::endl;
				mono = mono && !bad;
				prev = d;
				prevLum = lum;
			}
			kneeOk = (!pres.model.kneeOn || std::max({dW[0], dW[1], dW[2]}) >= 250.0) && mono;
			const double F = pres.model.darkFloor;
			double maxSpread = 0.0;
			const int n = pres.n;
			for (int b = 0; b < n; b++)
				for (int g = 0; g < n; g++)
					for (int r = 0; r < n; r++)
					{
						const double vr = 255.0 * r / (n - 1), vg = 255.0 * g / (n - 1), vb = 255.0 * b / (n - 1);
						if (0.299 * vr + 0.587 * vg + 0.114 * vb >= 0.5 * F)
							continue;
						const cv::Vec3d &d = pres.lut[(static_cast<std::size_t>(b) * n + g) * n + r];
						maxSpread = std::max({maxSpread, std::fabs(d[0] - d[1]), std::fabs(d[1] - d[2])});
					}
			darkOk = F > 0.0 && maxSpread < 0.5;
			std::cout << "Selftest knee: " << (pres.model.kneeOn ? "on" : "off") << ", camera-white drive max "
					  << std::max({dW[0], dW[1], dW[2]}) << ", gray axis " << (mono ? "monotone" : "NOT monotone")
					  << " -> " << (kneeOk ? "PASS" : "FAIL") << std::endl;
			std::cout << "Selftest dark floor " << F << ": max R/G/B drive spread below half of it " << maxSpread
					  << " -> " << (darkOk ? "PASS" : "FAIL") << std::endl;
		}
	}

	// 5. Clip-detection unit checks (legacy ISP path): channels pinned near full
	// scale across many samples are flagged; one maxing well below is kept.
	bool clipOk;
	{
		std::vector<SweepSample> cs(40);
		for (int i = 0; i < 40; i++)
		{
			cs[i].drive[0] = cs[i].drive[1] = cs[i].drive[2] = i * 255.0 / 39;
			// blue pins at the ADC ceiling (clipped); red maxes ~250, still below it
			// (soft compression -> kept); green maxes ~200 (kept).
			cs[i].led = cv::Vec3d(std::min(250.0, i * 7.2), std::min(200.0, i * 6.0), std::min(254.0, i * 9.0));
		}
		cv::Vec3d ceil;
		std::vector<char> clp;
		detectClipCeilings(cs, ceil, clp);
		clipOk = (ceil[2] > 0 && ceil[0] == 0 && ceil[1] == 0); // only blue pins at the sensor ceiling
		std::cout << "Clip-detect selftest: ceilings R=" << ceil[0] << " G=" << ceil[1]
				  << " B=" << ceil[2] << " -> " << (clipOk ? "PASS" : "FAIL") << std::endl;

		// A soft plateau just under the pinned ceiling (many samples within kEps
		// of blue's 254 pin, but never reaching it) must also be kept.
		for (int i = 0; i < 40; i++)
			cs[i].led[0] = std::min(252.5, i * 7.2);
		detectClipCeilings(cs, ceil, clp);
		const bool softOk = (ceil[0] == 0 && ceil[2] > 0);
		clipOk = clipOk && softOk;
		std::cout << "Clip-detect selftest (soft plateau near pin): R=" << ceil[0]
				  << " B=" << ceil[2] << " -> " << (softOk ? "PASS" : "FAIL") << std::endl;
	}

	const bool pass = tested >= 100 && (p95 <= 2.0) && (mx <= 6.0) && (anchorErr <= 3.0) && respOk && legacyWorse &&
					  kneeOk && darkOk && clipOk;
	std::cout << (pass ? "SELFTEST PASS" : "SELFTEST FAIL") << std::endl;
	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
