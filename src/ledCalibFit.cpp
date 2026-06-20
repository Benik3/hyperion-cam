#include "ledCalibFit.h"

#include <opencv2/core.hpp>

#include <algorithm>
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
		cv::Vec3d led; // encoded LED-ROI mean
		cv::Vec3d tv;  // encoded TV-ROI mean (darkness sanity, unused by fit)
	};

	struct CalibData
	{
		std::vector<SweepSample> samples;
		cv::Vec3d baselineLed{0, 0, 0};
		double anchorDrive[3] = {255, 255, 255};
		cv::Vec3d anchorTv{255, 255, 255};
		cv::Vec3d anchorLed{255, 255, 255};
		cv::Vec3d gains{1, 1, 1};
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
		double p = 2.2;
		cv::Matx33d A;
		cv::Matx33d Ainv;
		Curve e[3];
		cv::Vec3d corr{1, 1, 1}; // g^p * anchor renorm, applied to A*e
		cv::Vec3d baselineLin{0, 0, 0};
		// diagnostics
		double additivityResid = 0.0;
		double residMedian = 0.0;
		double residP95 = 0.0;
	};

	enum NodeFlag : uint8_t
	{
		NodeInGamut = 0,
		NodeDesaturated = 1,
		NodeLumClipped = 2,
		NodeBlack = 4
	};

	struct FitResult
	{
		FitModel model;
		int n = 33;
		std::vector<cv::Vec3d> lut; // drive 0..255, red index fastest
		std::vector<uint8_t> flags;
		double pctDesat = 0.0, pctClip = 0.0;
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

	bool loadLedSweepCsv(const std::string &path, std::vector<SweepSample> &out)
	{
		std::ifstream f(path);
		if (!f)
		{
			std::cout << "ERROR: cannot open " << path << std::endl;
			return false;
		}
		std::string line;
		std::getline(f, line); // header
		while (std::getline(f, line))
		{
			if (line.empty())
				continue;
			std::vector<double> col;
			std::stringstream ss(line);
			std::string cell;
			while (std::getline(ss, cell, ','))
				col.push_back(std::atof(cell.c_str()));
			// drive(3), led mean(3), led tstd(3), led sstd(3), tv mean(3), frames
			if (col.size() < 16)
				continue;
			SweepSample s;
			s.drive[0] = col[0];
			s.drive[1] = col[1];
			s.drive[2] = col[2];
			s.led = {col[3], col[4], col[5]};
			s.tv = {col[12], col[13], col[14]};
			out.push_back(s);
		}
		return !out.empty();
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

	// HyperHDR-WB workflow: derive the von-Kries white from the brightest neutral
	// level where NEITHER the TV nor the LED camera reading clips. The camera
	// maxes below 255, and the TV or the LEDs can be the brighter side, so the
	// true-white (255) point is often saturated -> use a dimmer clean white.
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

		// Brightest neutral level (from the TV patches) with both sides clean.
		std::sort(tvN.begin(), tvN.end(), [](auto &a, auto &b) { return a.first > b.first; });
		for (const auto &[L, tv] : tvN)
		{
			if (L < 64.0)
				break; // too dark to be a usable white reference
			const cv::Vec3d led = ledAt(L);
			if (clipped(tv) || clipped(led))
				continue;
			d.anchorDrive[0] = d.anchorDrive[1] = d.anchorDrive[2] = L;
			d.anchorTv = tv;
			d.anchorLed = led;
			for (int c = 0; c < 3; c++)
				d.gains[c] = led[c] > 1.0 ? tv[c] / led[c] : 1.0;
			std::cout << "White anchor from sweeps at neutral level " << L
					  << (L < 254.0 ? " (dimmer than 255 to avoid camera clipping)" : "") << ": "
					  << "TV " << tv[0] << "," << tv[1] << "," << tv[2]
					  << "  LED " << led[0] << "," << led[1] << "," << led[2]
					  << "  gains " << d.gains[0] << "," << d.gains[1] << "," << d.gains[2] << std::endl;
			return AnchorResult::Derived;
		}
		std::cout << "ERROR: no unclipped neutral level found (every white patch saturates the camera "
					 "at >= " << clip << "). Lower the TV and/or LED brightness, or raise --calib-clip-level."
				  << std::endl;
		return AnchorResult::AllClipped;
	}

	bool loadCalibData(const std::string &dir, CalibData &d, double clip = 250.0)
	{
		if (!loadLedSweepCsv(dir + "/led_sweep.csv", d.samples))
			return false;

		std::string txt;
		if (readFile(dir + "/baseline.json", txt))
		{
			jsonTriple(txt, "led_mean", d.baselineLed);
		}
		else
		{
			std::cout << "WARNING: no baseline.json; using the (0,0,0) sweep sample" << std::endl;
			for (const auto &s : d.samples)
				if (s.drive[0] == 0 && s.drive[1] == 0 && s.drive[2] == 0)
					d.baselineLed = s.led;
		}

		if (!readFile(dir + "/white_anchor.json", txt))
		{
			// No eye-anchor phase: try to derive the white correction from the
			// sweep white points (requires HyperHDR WB to have eye-matched white).
			switch (deriveAnchorFromSweeps(dir, d, clip))
			{
			case AnchorResult::Derived:
				return true;
			case AnchorResult::AllClipped:
				// Error already printed; abort rather than silently ship a LUT
				// built on a blown-out white point.
				return false;
			case AnchorResult::NoData:
				std::cout << "WARNING: no white_anchor.json and could not derive white from tv_sweep.csv; "
							 "assuming neutral gains (no metameric correction - run --ledcalib white, "
							 "or set HyperHDR white balance and include a white patch in the TV sweep)" << std::endl;
				return true;
			}
			return true;
		}
		cv::Vec3d drv;
		if (jsonTriple(txt, "drive", drv))
		{
			d.anchorDrive[0] = drv[0];
			d.anchorDrive[1] = drv[1];
			d.anchorDrive[2] = drv[2];
		}
		jsonTriple(txt, "tv_mean", d.anchorTv);
		jsonTriple(txt, "led_mean", d.anchorLed);
		if (!jsonTriple(txt, "gains", d.gains))
		{
			for (int c = 0; c < 3; c++)
				d.gains[c] = d.anchorLed[c] > 1.0 ? d.anchorTv[c] / d.anchorLed[c] : 1.0;
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
		double bestP = 2.2, bestR = 1e18;
		for (double p = 1.2; p <= 3.2 + 1e-9; p += 0.05)
		{
			const double r = additivityResidual(d, p);
			if (r < bestR)
			{
				bestR = r;
				bestP = p;
			}
		}
		for (double p = bestP - 0.05; p <= bestP + 0.05 + 1e-9; p += 0.005)
		{
			const double r = additivityResidual(d, p);
			if (r < bestR)
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

	bool fitModel(const CalibData &d, FitModel &m, bool verbose)
	{
		m.p = estimateExponent(d, m.additivityResid);
		m.baselineLin = linv3(d.baselineLed, m.p);

		std::vector<cv::Vec3d> Y(d.samples.size());
		for (std::size_t i = 0; i < d.samples.size(); i++)
			Y[i] = linv3(d.samples[i].led, m.p) - m.baselineLin;

		// Initial A: columns from the strongest single-channel samples.
		cv::Matx33d A = cv::Matx33d::zeros();
		for (int c = 0; c < 3; c++)
		{
			double bestLvl = -1;
			cv::Vec3d bestY{0, 0, 0};
			for (std::size_t i = 0; i < d.samples.size(); i++)
			{
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
		cv::Vec3d corrLin;
		for (int c = 0; c < 3; c++)
			corrLin[c] = std::pow(std::max(d.gains[c], 1e-6), m.p);
		cv::Vec3d eW;
		for (int c = 0; c < 3; c++)
			eW[c] = m.e[c].eval(d.anchorDrive[c]);
		const cv::Vec3d totalW = m.A * eW + m.baselineLin;
		const cv::Vec3d tW = linv3(d.anchorTv, m.p);
		for (int c = 0; c < 3; c++)
			m.corr[c] = (totalW[c] > 1e-12) ? tW[c] / totalW[c] : corrLin[c];

		if (verbose)
		{
			std::cout << "Fit: p=" << m.p << " additivityResid=" << m.additivityResid
					  << " rel resid median=" << m.residMedian << " p95=" << m.residP95 << std::endl;
			std::cout << "A = " << cv::Mat(m.A) << std::endl;
		}
		return true;
	}

	// ---------- inversion ----------

	uint8_t invertNode(const FitModel &m, const cv::Vec3d &vEnc, cv::Vec3d &driveOut)
	{
		uint8_t flag = NodeInGamut;
		// Target is total corrected light; the LEDs only control the part
		// above the ambient floor.
		cv::Vec3d t = linv3(vEnc, m.p);
		for (int c = 0; c < 3; c++)
			t[c] = t[c] / std::max(m.corr[c], 1e-12) - m.baselineLin[c];

		cv::Vec3d e = m.Ainv * t;

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
				driveOut = {0, 0, 0};
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

		// Overflow: scale down (keeps chromaticity, reduces luminance).
		const double mx = std::max({e[0], e[1], e[2]});
		if (mx > 1.0)
		{
			e *= 1.0 / mx;
			flag |= NodeLumClipped;
		}

		for (int c = 0; c < 3; c++)
			driveOut[c] = std::clamp(m.e[c].invert(e[c]), 0.0, 255.0);
		return flag;
	}

	void buildLut(const FitModel &m, FitResult &res)
	{
		const int n = res.n;
		res.lut.assign(static_cast<std::size_t>(n) * n * n, cv::Vec3d{0, 0, 0});
		res.flags.assign(res.lut.size(), 0);
		std::size_t idx = 0;
		int desat = 0, clip = 0;
		for (int b = 0; b < n; b++)
		{
			for (int g = 0; g < n; g++)
			{
				for (int r = 0; r < n; r++, idx++)
				{
					const cv::Vec3d v{255.0 * r / (n - 1), 255.0 * g / (n - 1), 255.0 * b / (n - 1)};
					res.flags[idx] = invertNode(m, v, res.lut[idx]);
					if (res.flags[idx] & NodeDesaturated)
						desat++;
					if (res.flags[idx] & NodeLumClipped)
						clip++;
				}
			}
		}
		res.pctDesat = 100.0 * desat / static_cast<double>(res.lut.size());
		res.pctClip = 100.0 * clip / static_cast<double>(res.lut.size());
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
		const FitModel &m = res.model;
		f << "{\n";
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

	bool runFitCore(const LedCalibFitConfig &cfg, const CalibData &data, FitResult &res)
	{
		if (!fitModel(data, res.model, cfg.verbose))
			return false;
		res.n = std::clamp(cfg.lutSize, 2, 65);
		buildLut(res.model, res);
		return true;
	}

	// ---------- self-test ----------

	struct TruthChain
	{
		// Ground-truth LED/wall/camera chain, deliberately different from the
		// model's parametrization defaults.
		cv::Matx33d A{0.72, 0.10, 0.034,
					  0.068, 0.66, 0.085,
					  0.026, 0.13, 0.68};
		double exps[3] = {2.0, 1.9, 2.1};
		double pEnc = 2.2; // camera encodes with x^(1/pEnc)
		cv::Vec3d ambient{0.004, 0.005, 0.006};

		cv::Vec3d glowLin(const cv::Vec3d &drive) const
		{
			cv::Vec3d e;
			for (int c = 0; c < 3; c++)
				e[c] = std::pow(std::clamp(drive[c], 0.0, 255.0) / 255.0, exps[c]);
			return A * e + ambient;
		}

		cv::Vec3d encode(const cv::Vec3d &lin) const
		{
			cv::Vec3d out;
			for (int c = 0; c < 3; c++)
				out[c] = 255.0 * std::pow(std::clamp(lin[c], 0.0, 1.0), 1.0 / pEnc);
			return out;
		}

		cv::Vec3d reading(const cv::Vec3d &drive) const
		{
			return encode(glowLin(drive));
		}
	};

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
	CalibData data;
	if (!loadCalibData(cfg.calibDir, data, cfg.clipLevel))
		return EXIT_FAILURE;
	std::cout << "Loaded " << data.samples.size() << " LED sweep samples from " << cfg.calibDir << std::endl;

	FitResult res;
	if (!runFitCore(cfg, data, res))
		return EXIT_FAILURE;

	std::cout << "Fit: p=" << res.model.p
			  << " rel resid median=" << res.model.residMedian << " p95=" << res.model.residP95
			  << "; gamut: " << res.pctDesat << "% desaturated, " << res.pctClip << "% luminance-clipped"
			  << std::endl;
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

	const std::string dir = "/tmp/ledcalib-selftest";
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);

	// Synthetic sweep: same plan as --ledcalib (8^3 cube + 17 grays).
	{
		std::ofstream f(dir + "/led_sweep.csv");
		f << "drive_r,drive_g,drive_b,led_r,led_g,led_b,led_r_tstd,led_g_tstd,led_b_tstd,"
			 "led_r_sstd,led_g_sstd,led_b_sstd,tv_r,tv_g,tv_b,frames\n";
		auto emit = [&](int r, int g, int b)
		{
			cv::Vec3d m = truth.reading({static_cast<double>(r), static_cast<double>(g), static_cast<double>(b)});
			for (int c = 0; c < 3; c++)
				m[c] = std::clamp(m[c] + noise(rng) / std::sqrt(20.0), 0.0, 255.0);
			f << r << "," << g << "," << b << "," << m[0] << "," << m[1] << "," << m[2]
			  << ",0.3,0.3,0.3,1,1,1,2,2,2,20\n";
		};
		for (int r = 0; r < 8; r++)
			for (int g = 0; g < 8; g++)
				for (int b = 0; b < 8; b++)
					emit(r * 255 / 7, g * 255 / 7, b * 255 / 7);
		for (int i = 0; i < 17; i++)
		{
			const int v = i * 255 / 16;
			emit(v, v, v);
		}
	}

	// Baseline + anchor. The anchor models the user's eye-match: TV white read
	// by the camera vs the glow reading at the matched drive.
	const cv::Vec3d anchorDrive{235, 240, 228};
	const cv::Vec3d anchorLed = truth.reading(anchorDrive);
	const cv::Vec3d anchorTv{224.0, 217.0, 209.0};
	{
		std::ofstream f(dir + "/baseline.json");
		const cv::Vec3d b = truth.reading({0, 0, 0});
		f << "{\n  \"led_mean\": [" << b[0] << ", " << b[1] << ", " << b[2] << "]\n}\n";
	}
	{
		std::ofstream f(dir + "/white_anchor.json");
		f << "{\n  \"drive\": [" << anchorDrive[0] << ", " << anchorDrive[1] << ", " << anchorDrive[2] << "],\n";
		f << "  \"tv_mean\": [" << anchorTv[0] << ", " << anchorTv[1] << ", " << anchorTv[2] << "],\n";
		f << "  \"led_mean\": [" << anchorLed[0] << ", " << anchorLed[1] << ", " << anchorLed[2] << "],\n";
		f << "  \"gains\": [" << anchorTv[0] / anchorLed[0] << ", " << anchorTv[1] / anchorLed[1]
		  << ", " << anchorTv[2] / anchorLed[2] << "]\n}\n";
	}

	LedCalibFitConfig cfg = cfgIn;
	cfg.calibDir = dir;
	cfg.lutOut = dir + "/lut.cube";

	CalibData data;
	if (!loadCalibData(cfg.calibDir, data))
		return EXIT_FAILURE;
	FitResult res;
	if (!runFitCore(cfg, data, res))
		return EXIT_FAILURE;
	if (!writeCube(cfg.lutOut, res, dir))
		return EXIT_FAILURE;
	writeReport(dir, res);

	std::cout << "Selftest fit: p=" << res.model.p << " (truth " << truth.pEnc << ")"
			  << " resid median=" << res.model.residMedian << " p95=" << res.model.residP95
			  << " gamut desat=" << res.pctDesat << "% clip=" << res.pctClip << "%" << std::endl;

	// Closed loop: target v -> LUT -> drive -> truth chain -> anchor-corrected
	// encoded reading; must reproduce v on in-gamut points.
	const cv::Vec3d gains{anchorTv[0] / anchorLed[0], anchorTv[1] / anchorLed[1], anchorTv[2] / anchorLed[2]};
	std::uniform_real_distribution<double> uni(0.0, 255.0);
	std::vector<double> errs;
	int tested = 0, skipped = 0;
	for (int i = 0; i < 4000; i++)
	{
		const cv::Vec3d v{uni(rng), uni(rng), uni(rng)};
		if (!nodeUsable(res, v))
		{
			skipped++;
			continue;
		}
		const cv::Vec3d drive = sampleLutTrilinear(res, v);
		const cv::Vec3d enc = truth.reading(drive);
		for (int c = 0; c < 3; c++)
			errs.push_back(std::fabs(gains[c] * enc[c] - v[c]));
		tested++;
	}
	if (tested < 100)
	{
		std::cout << "SELFTEST FAIL: too few in-gamut test points (" << tested << ")" << std::endl;
		return EXIT_FAILURE;
	}
	std::sort(errs.begin(), errs.end());
	const double med = errs[errs.size() / 2];
	const double p95 = errs[static_cast<std::size_t>(static_cast<double>(errs.size() - 1) * 0.95)];
	const double mx = errs.back();

	// Anchor must round-trip to its own drive value.
	const cv::Vec3d sW = sampleLutTrilinear(res, anchorTv);
	double anchorErr = 0.0;
	for (int c = 0; c < 3; c++)
		anchorErr = std::max(anchorErr, std::fabs(sW[c] - anchorDrive[c]));

	std::cout << "Selftest closed loop (" << tested << " pts, " << skipped << " out-of-gamut skipped): "
			  << "err median=" << med << " p95=" << p95 << " max=" << mx
			  << " (encoded 0-255 units); anchor drive err=" << anchorErr << std::endl;

	const bool pass = (p95 <= 2.0) && (mx <= 6.0) && (anchorErr <= 3.0);
	std::cout << (pass ? "SELFTEST PASS" : "SELFTEST FAIL") << std::endl;
	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
