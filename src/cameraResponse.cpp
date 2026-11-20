#include "cameraResponse.h"
#include "tuning.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

double CsvTable::get(std::size_t row, const std::string &name, double def) const
{
	auto it = col.find(name);
	if (it == col.end() || row >= rows.size())
		return def;
	const auto &r = rows[row];
	return (it->second < static_cast<int>(r.size())) ? r[it->second] : def;
}

bool loadCsvTable(const std::string &path, CsvTable &out)
{
	std::ifstream f(path);
	if (!f)
		return false;
	std::string line;
	bool haveHeader = false;
	while (std::getline(f, line))
	{
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		if (line.empty())
			continue;
		if (line[0] == '#')
		{
			out.comments.push_back(line);
			continue;
		}
		std::stringstream ss(line);
		std::string cell;
		if (!haveHeader)
		{
			int i = 0;
			while (std::getline(ss, cell, ','))
				out.col[cell] = i++;
			haveHeader = true;
			continue;
		}
		std::vector<double> v;
		while (std::getline(ss, cell, ','))
			v.push_back(std::atof(cell.c_str()));
		out.rows.push_back(std::move(v));
	}
	return haveHeader;
}

double CameraResponse::eval(double z) const
{
	if (lin.size() < 2)
		return 0.0;
	z = std::clamp(z, 0.0, static_cast<double>(lin.size() - 1));
	const std::size_t i = std::min(static_cast<std::size_t>(z), lin.size() - 2);
	const double t = z - static_cast<double>(i);
	return lin[i] + t * (lin[i + 1] - lin[i]);
}

double CameraResponse::invert(double l) const
{
	if (lin.size() < 2)
		return 0.0;
	if (l <= lin.front())
		return 0.0;
	if (l >= lin.back())
		return static_cast<double>(lin.size() - 1);
	const auto it = std::lower_bound(lin.begin(), lin.end(), l);
	const std::size_t hi = static_cast<std::size_t>(it - lin.begin());
	const std::size_t lo = hi - 1;
	const double d = lin[hi] - lin[lo];
	return static_cast<double>(lo) + (d > 0 ? (l - lin[lo]) / d : 0.0);
}

namespace
{
	struct Sample
	{
		int series;
		double z;	// ISP code
		double lnE; // ln RAW light
		double w;	// hat weight
	};

	// Nominal exponent the smoothness prior is taken relative to: g(z) minus
	// kGamma0*ln(z) is smooth for both the stock (S-ish) and a power-law curve,
	// whereas g itself has huge curvature near z -> 0.
	constexpr double kGamma0 = 2.2;
	constexpr double kLambda = 10.0; // smoothness weight vs data; 10 follows the ISP's hardware-knot corners (selftest: 40 rounds them off >4%, 4 fits noise)

	double hat(double z) { return 0.05 + std::min(z, 255.0 - z) / 127.5; }

	double tableAt(const std::vector<double> &t, double z)
	{
		z = std::clamp(z, 0.0, static_cast<double>(t.size() - 1));
		const std::size_t i = std::min(static_cast<std::size_t>(z), t.size() - 2);
		return t[i] + (z - static_cast<double>(i)) * (t[i + 1] - t[i]);
	}

	bool readText(const std::string &path, std::string &out)
	{
		std::ifstream f(path);
		if (!f)
			return false;
		std::stringstream ss;
		ss << f.rdbuf();
		out = ss.str();
		return true;
	}
}

bool loadResponseHints(const std::string &metaPath, ResponseHints &out)
{
	out = ResponseHints{};
	std::string txt;
	if (!readText(metaPath, txt))
		return false;
	// "tuning_file": "path"
	if (auto k = txt.find("\"tuning_file\""); k != std::string::npos)
	{
		const auto q0 = txt.find('"', txt.find(':', k) + 1);
		const auto q1 = (q0 == std::string::npos) ? q0 : txt.find('"', q0 + 1);
		if (q1 != std::string::npos)
			out.tuningFile = txt.substr(q0 + 1, q1 - q0 - 1);
	}
	if (!out.tuningFile.empty() && !loadTuningToneCurve(out.tuningFile, out.priorLin))
		std::cout << "WARNING: cannot read the tone curve of " << out.tuningFile
				  << " (recorded in " << metaPath << "); using a power-law prior" << std::endl;
	// "ccm": [9 numbers] (frame metadata, row-major)
	if (auto k = txt.find("\"ccm\""); k != std::string::npos)
	{
		std::size_t pos = txt.find('[', k);
		cv::Matx33d m;
		int n = 0;
		while (pos != std::string::npos && n < 9)
		{
			char *end = nullptr;
			const double v = std::strtod(txt.c_str() + pos + 1, &end);
			if (end == txt.c_str() + pos + 1)
				break;
			m.val[n++] = v;
			pos = txt.find_first_of(",]", static_cast<std::size_t>(end - txt.c_str()));
			if (pos == std::string::npos || txt[pos] == ']')
				break;
		}
		cv::Matx33d inv;
		if (n == 9 && cv::invert(m, inv, cv::DECOMP_LU) != 0)
		{
			for (int i = 0; i < 9; i++)
				out.ccmInv[i] = inv.val[i];
			out.haveCcm = true;
		}
	}
	return true;
}

bool fitCameraResponse(const std::string &csvPath, CameraResponse &out, bool verbose, const ResponseHints *hints)
{
	const bool havePrior = hints && hints->priorLin.size() == 256 && hints->priorLin[255] > 0.0;
	const bool preCheck = havePrior && hints->haveCcm;
	int preDropped = 0;
	out = CameraResponse{};
	CsvTable t;
	if (!loadCsvTable(csvPath, t))
	{
		std::cout << "ERROR: cannot read " << csvPath << std::endl;
		return false;
	}

	// Series = (ROI, ISP channel). Light = sum of the black-subtracted RAW planes
	// of the same ROI (fixed-spectrum source: every plane is proportional to it).
	std::vector<std::string> rois;
	for (const char *r : {"led", "tv"})
		if (t.has(std::string(r) + "_isp_r") && t.has(std::string(r) + "_raw_r"))
			rois.push_back(r);
	if (rois.empty())
	{
		std::cout << "ERROR: " << csvPath << " has no isp/raw columns" << std::endl;
		return false;
	}
	const char *ch[3] = {"r", "g", "b"};
	std::vector<Sample> samples;
	std::vector<int> seriesCount(rois.size() * 3, 0);
	for (std::size_t j = 0; j < t.rows.size(); j++)
		for (std::size_t k = 0; k < rois.size(); k++)
		{
			const std::string &R = rois[k];
			double light = 0.0, rawSat = 0.0;
			for (const char *c : ch)
			{
				light += t.get(j, R + "_raw_" + c);
				rawSat = std::max(rawSat, t.get(j, R + "_raw_sat_" + c));
			}
			if (rawSat > 0.002 || light < 3.0)
				continue; // sensor clipped, or light in the noise
			// The ISP's digital gain (libcamera's compensation of the shutter's line
			// quantisation, up to a few % at short exposures) reaches the ISP output
			// but not the RAW frame.
			const double dg = t.get(j, "digital_gain", 0.0);
			if (dg > 0.0)
				light *= dg;
			if (preCheck)
			{
				// Pre-matrix value = CCM^-1 * (post-matrix linear): >= full scale means
				// the colour-gain stage clipped this ROI (only reconstructable while no
				// post-matrix channel is itself clipped).
				double post[3];
				bool inRange = true;
				for (int c = 0; c < 3; c++)
				{
					const double z = t.get(j, R + "_isp_" + ch[c]);
					inRange = inRange && z >= 1.0 && z <= 253.0;
					post[c] = tableAt(hints->priorLin, z);
				}
				if (inRange)
				{
					double mx = 0.0;
					for (int r = 0; r < 3; r++)
						mx = std::max(mx, hints->ccmInv[3 * r] * post[0] + hints->ccmInv[3 * r + 1] * post[1] +
											  hints->ccmInv[3 * r + 2] * post[2]);
					if (mx > 0.97)
					{
						preDropped++;
						continue;
					}
				}
			}
			for (int c = 0; c < 3; c++)
			{
				const double z = t.get(j, R + "_isp_" + ch[c]);
				const double sat = t.get(j, R + "_isp_sat_" + ch[c]);
				if (sat > 0.01 || z < 1.0 || z > 253.0)
					continue; // ISP clipped, or at the floor
				const int s = static_cast<int>(k) * 3 + c;
				samples.push_back({s, z, std::log(light), hat(z)});
				seriesCount[s]++;
			}
		}
	// Keep series with enough points to pin their scale.
	std::vector<int> seriesIdx(seriesCount.size(), -1);
	int nSeries = 0;
	for (std::size_t s = 0; s < seriesCount.size(); s++)
		if (seriesCount[s] >= 4)
			seriesIdx[s] = nSeries++;
	samples.erase(std::remove_if(samples.begin(), samples.end(), [&](const Sample &s)
								 { return seriesIdx[s.series] < 0; }),
				  samples.end());
	if (nSeries < 1 || samples.size() < 12)
	{
		std::cout << "ERROR: camera response: not enough usable samples (" << samples.size()
				  << ") - is the scene lit and still?" << std::endl;
		return false;
	}

	// Unknowns: g[1..255] (index z-1) and b_s = ln(scale) per series.
	const int nG = 255;
	const int N = nG + nSeries;
	const int nSmooth = 253 - 1; // z = 2..253
	const int M = static_cast<int>(samples.size()) + nSmooth + 1;
	cv::Mat A = cv::Mat::zeros(M, N, CV_64F), b = cv::Mat::zeros(M, 1, CV_64F);
	int row = 0;
	for (const Sample &s : samples)
	{
		int i = static_cast<int>(std::floor(s.z));
		i = std::clamp(i, 1, 254);
		const double f = std::clamp(s.z - i, 0.0, 1.0);
		A.at<double>(row, i - 1) += s.w * (1.0 - f);
		A.at<double>(row, i) += s.w * f;
		A.at<double>(row, nG + seriesIdx[s.series]) = -s.w;
		b.at<double>(row) = s.w * s.lnE;
		row++;
	}
	for (int z = 2; z <= 253; z++)
	{
		const double w = kLambda * hat(z);
		A.at<double>(row, z - 2) = w;
		A.at<double>(row, z - 1) = -2.0 * w;
		A.at<double>(row, z) = w;
		if (havePrior)
		{
			auto g0 = [&](int zz) { return std::log(std::max(hints->priorLin[zz], 1e-9)); };
			b.at<double>(row) = w * (g0(z - 1) - 2.0 * g0(z) + g0(z + 1));
		}
		else
			b.at<double>(row) =
				w * kGamma0 * (std::log(z - 1.0) - 2.0 * std::log(static_cast<double>(z)) + std::log(z + 1.0));
		row++;
	}
	A.at<double>(row, 128 - 1) = 100.0; // gauge: g(128) = 0
	row++;
	cv::Mat x;
	if (!cv::solve(A, b, x, cv::DECOMP_SVD))
	{
		std::cout << "ERROR: camera response solve failed" << std::endl;
		return false;
	}

	auto gAt = [&](double z)
	{
		int i = std::clamp(static_cast<int>(std::floor(z)), 1, 254);
		const double f = std::clamp(z - i, 0.0, 1.0);
		return (1.0 - f) * x.at<double>(i - 1) + f * x.at<double>(i);
	};

	// Residuals (mid-range samples dominate the verdict; the ends are noisy /
	// compressed by design).
	std::vector<double> ss(nSeries, 0.0);
	std::vector<int> sn(nSeries, 0);
	double sAll = 0.0;
	int nAll = 0;
	out.zMin = 255;
	out.zMax = 0;
	for (const Sample &s : samples)
	{
		const int si = seriesIdx[s.series];
		const double r = gAt(s.z) - x.at<double>(nG + si) - s.lnE;
		out.zMin = std::min(out.zMin, static_cast<int>(s.z));
		out.zMax = std::max(out.zMax, static_cast<int>(std::ceil(s.z)));
		if (s.w < 0.3)
			continue;
		ss[si] += r * r;
		sn[si]++;
		sAll += r * r;
		nAll++;
	}
	out.rmsResid = nAll ? std::sqrt(sAll / nAll) : 0.0;
	for (int i = 0; i < nSeries; i++)
		if (sn[i])
			out.worstSeriesResid = std::max(out.worstSeriesResid, std::sqrt(ss[i] / sn[i]));

	// Curve: lin = exp(g), lin[0] = 0, monotone, normalised to lin[255] = 1.
	out.lin.assign(256, 0.0);
	for (int z = 1; z <= 255; z++)
		out.lin[z] = std::exp(x.at<double>(z - 1));
	// Below the data the fit can only extrapolate, and the smoothness prior pins
	// the shape, not the slope: a slight tilt from the noisiest (darkest) samples
	// grows to ~10% at code 4. With the tuning's own curve known, splice its
	// exact shape below the lowest well-measured code instead.
	if (havePrior)
	{
		std::vector<int> cnt(256, 0);
		for (const Sample &sm : samples)
			if (sm.w >= 0.1)
				cnt[std::clamp(static_cast<int>(std::lround(sm.z)), 0, 255)]++;
		int zLow = 255;
		for (int z = 1; z <= 247; z++)
		{
			int n = 0;
			for (int k = z; k <= z + 8; k++)
				n += cnt[k];
			if (n >= 6)
			{
				zLow = z + 4;
				break;
			}
		}
		if (zLow < 255 && hints->priorLin[zLow] > 0.0)
		{
			const double scale = out.lin[zLow] / hints->priorLin[zLow];
			for (int z = 1; z < zLow; z++)
				out.lin[z] = scale * hints->priorLin[z];
			out.spliceBelow = zLow;
		}
	}
	for (int z = 2; z <= 255; z++)
		out.lin[z] = std::max(out.lin[z], out.lin[z - 1] * (1.0 + 1e-9));
	const double top = out.lin[255];
	for (double &v : out.lin)
		v /= top;

	// Source drift over the reference steps (the most common target exposure):
	// RAW light per unit exposure should not move. Informational - the curve
	// itself is drift-proof because it is referenced to RAW, not to the shutter.
	if (t.has("target_exposure_us") && t.has("exposure_us"))
	{
		std::map<long long, int> freq;
		for (std::size_t j = 0; j < t.rows.size(); j++)
			freq[std::llround(t.get(j, "target_exposure_us"))]++;
		long long ref = 0;
		int best = 0;
		for (const auto &[e, n] : freq)
			if (n > best)
			{
				best = n;
				ref = e;
			}
		for (const std::string &R : rois)
		{
			double lo = 1e300, hi = 0.0;
			for (std::size_t j = 0; j < t.rows.size(); j++)
			{
				if (std::llround(t.get(j, "target_exposure_us")) != ref)
					continue;
				const double e = t.get(j, "exposure_us");
				double light = 0.0, sat = 0.0;
				for (const char *c : ch)
				{
					light += t.get(j, R + "_raw_" + c);
					sat = std::max(sat, t.get(j, R + "_raw_sat_" + c));
				}
				if (e <= 0 || light < 3.0 || sat > 0.002)
					continue;
				lo = std::min(lo, light / e);
				hi = std::max(hi, light / e);
			}
			if (hi > 0 && lo < 1e300)
				out.sourceDrift = std::max(out.sourceDrift, hi / lo - 1.0);
		}
	}

	out.series = nSeries;
	out.samples = static_cast<int>(samples.size());
	out.ok = true;
	if (verbose && havePrior)
		std::cout << "Camera response: prior = tone curve of " << hints->tuningFile
				  << (out.spliceBelow > 0 ? "; its shape used below code " + std::to_string(out.spliceBelow) : "")
				  << std::endl;
	if (preDropped > 0)
		std::cout << "Camera response: dropped " << preDropped
				  << " ROI samples clipped at the ISP's colour-gain stage (before the matrix; use -b 1,1)" << std::endl;
	if (verbose)
		std::cout << "Camera response fit: " << nSeries << " series, " << samples.size() << " samples, codes "
				  << out.zMin << ".." << out.zMax << ", RMS " << (100.0 * out.rmsResid) << "%" << std::endl;
	return true;
}

bool ledCalibCheckResponse(const std::string &csvPath, bool verbose, const ResponseHints *hints)
{
	CameraResponse r;
	if (!fitCameraResponse(csvPath, r, verbose, hints))
		return false;
	std::cout << "Camera response: " << r.series << " series, " << r.samples << " samples over codes " << r.zMin
			  << ".." << r.zMax << "; fit RMS " << (100.0 * r.rmsResid) << "% (worst series "
			  << (100.0 * r.worstSeriesResid) << "%)" << std::endl;
	std::cout << "  local exponent (d ln light / d ln code):";
	for (int z : {16, 32, 64, 128, 192, 240})
	{
		const double a = z - 4.0, c = z + 4.0;
		const double la = r.eval(a), lc = r.eval(c);
		if (la > 0 && lc > 0)
			std::cout << "  " << z << ":" << std::log(lc / la) / std::log(c / a);
	}
	std::cout << std::endl;
	if (r.worstSeriesResid > 0.04)
		std::cout << "WARNING: the camera response is NOT one consistent curve (worst series "
				  << (100.0 * r.worstSeriesResid) << "% off). The ISP is probably still scene-adaptive "
				  << "(contrast enhancement / adaptive lens shading) - run with the frozen tuning file "
				  << "(--make-tuning / --tuning) - or the scene moved during the sweep." << std::endl;
	else
		std::cout << "  -> consistent single curve: the ISP behaves as frozen." << std::endl;
	if (r.sourceDrift > 0.02)
		std::cout << "  note: the light sources drifted " << (100.0 * r.sourceDrift)
				  << "% over the sweep (TV dimming / LED warm-up); the curve is referenced to RAW, so it "
				  << "is unaffected, but keep the room/TV static during the sweeps." << std::endl;
	if (r.zMax < 230 || r.zMin > 20)
		std::cout << "  note: data covered codes " << r.zMin << ".." << r.zMax
				  << " only; the curve outside is extrapolated (check the LED/TV ROIs are lit)." << std::endl;
	return true;
}
