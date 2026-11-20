#include "tuning.h"
#include "cameraResponse.h"

// Header-only Boost.JSON (Boost is already a dependency; this TU compiles the
// implementation, so no extra library is linked).
#include <boost/json/src.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace json = boost::json;

namespace
{
	bool readText(const std::string &path, std::string &out)
	{
		std::ifstream f(path, std::ios::binary);
		if (!f)
			return false;
		std::stringstream ss;
		ss << f.rdbuf();
		out = ss.str();
		return true;
	}

	bool parseJsonFile(const std::string &path, json::value &out)
	{
		std::string text;
		if (!readText(path, text))
		{
			std::cout << "ERROR: cannot read " << path << std::endl;
			return false;
		}
		boost::system::error_code ec;
		out = json::parse(text, ec);
		if (ec)
		{
			std::cout << "ERROR: " << path << " is not valid JSON: " << ec.message() << std::endl;
			return false;
		}
		return true;
	}

	// Pretty printer close to the stock files' style: objects one key per line,
	// arrays of numbers on wrapped lines.
	void writeJson(std::ostream &os, const json::value &v, int ind)
	{
		const std::string pad(ind, ' '), pad2(ind + 4, ' ');
		switch (v.kind())
		{
		case json::kind::object:
		{
			const json::object &o = v.get_object();
			if (o.empty())
			{
				os << "{}";
				return;
			}
			os << "{\n";
			std::size_t i = 0;
			for (const auto &kv : o)
			{
				os << pad2 << json::serialize(json::value(kv.key())) << ": ";
				writeJson(os, kv.value(), ind + 4);
				os << (++i < o.size() ? ",\n" : "\n");
			}
			os << pad << "}";
			return;
		}
		case json::kind::array:
		{
			const json::array &a = v.get_array();
			bool scalars = true;
			for (const auto &e : a)
				scalars = scalars && !e.is_object() && !e.is_array();
			if (a.empty())
			{
				os << "[]";
				return;
			}
			if (scalars)
			{
				os << "[";
				for (std::size_t i = 0; i < a.size(); i++)
				{
					if (i % 12 == 0)
						os << "\n" << pad2;
					os << json::serialize(a[i]) << (i + 1 < a.size() ? ", " : "");
				}
				os << "\n" << pad << "]";
				return;
			}
			os << "[\n";
			for (std::size_t i = 0; i < a.size(); i++)
			{
				os << pad2;
				writeJson(os, a[i], ind + 4);
				os << (i + 1 < a.size() ? ",\n" : "\n");
			}
			os << pad << "]";
			return;
		}
		default:
			os << json::serialize(v);
		}
	}

	// The algorithm block "rpi.<name>" in a version-2 tuning file, or null.
	json::object *findAlgo(json::value &root, const char *name)
	{
		if (!root.is_object())
			return nullptr;
		json::object &r = root.get_object();
		auto it = r.find("algorithms");
		if (it == r.end() || !it->value().is_array())
			return nullptr;
		for (json::value &a : it->value().get_array())
			if (a.is_object())
			{
				auto f = a.get_object().find(name);
				if (f != a.get_object().end() && f->value().is_object())
					return &f->value().get_object();
			}
		return nullptr;
	}

	double num(const json::value &v)
	{
		if (v.is_double())
			return v.get_double();
		if (v.is_int64())
			return static_cast<double>(v.get_int64());
		if (v.is_uint64())
			return static_cast<double>(v.get_uint64());
		return 0.0;
	}

	// Hardware tone-curve knots of the vc4 ISP (the stock curves use exactly
	// these 33 x positions; pisp resamples any curve).
	std::vector<int> curveKnots()
	{
		std::vector<int> x;
		for (int i = 0; i < 32; i++)
			x.push_back(i < 16 ? i * 1024 : (i < 24 ? (i - 16) * 2048 + 16384 : (i - 24) * 4096 + 32768));
		x.push_back(65535);
		return x;
	}

	// Adaptive ISP stages still enabled in a tuning (empty = frozen): contrast
	// enhancement and ALSC iterations change the camera's response with the scene.
	std::string adaptiveStagesOf(const json::value &root)
	{
		json::value &r = const_cast<json::value &>(root);
		std::string what;
		if (const json::object *c = findAlgo(r, "rpi.contrast"))
		{
			auto it = c->find("ce_enable");
			if (it == c->end() || num(it->value()) != 0.0) // libcamera default: enabled
				what += "contrast enhancement (ce_enable)";
		}
		if (const json::object *a = findAlgo(r, "rpi.alsc"))
		{
			auto it = a->find("n_iter");
			if (it == a->end() || num(it->value()) != 0.0) // libcamera default: 8
				what += std::string(what.empty() ? "" : " + ") + "adaptive lens shading (alsc n_iter)";
		}
		return what;
	}

	// Piecewise-linear tone curve (x,y in 0..1) from a gamma_curve array.
	struct ToneCurve
	{
		std::vector<double> x, y;
		bool load(const json::object *contrast)
		{
			x.clear();
			y.clear();
			if (!contrast)
				return false;
			auto it = contrast->find("gamma_curve");
			if (it == contrast->end() || !it->value().is_array())
				return false;
			const json::array &a = it->value().get_array();
			for (std::size_t i = 0; i + 1 < a.size(); i += 2)
			{
				x.push_back(num(a[i]) / 65535.0);
				y.push_back(num(a[i + 1]) / 65535.0);
			}
			return x.size() >= 2;
		}
		double inverse(double yy) const // encoded 0..1 -> linear 0..1
		{
			if (yy <= y.front())
				return x.front();
			for (std::size_t i = 1; i < y.size(); i++)
				if (y[i] >= yy)
				{
					const double d = y[i] - y[i - 1];
					return x[i - 1] + (d > 0 ? (yy - y[i - 1]) / d : 0.0) * (x[i] - x[i - 1]);
				}
			return x.back();
		}
		double forward(double xx) const
		{
			if (xx <= x.front())
				return y.front();
			for (std::size_t i = 1; i < x.size(); i++)
				if (x[i] >= xx)
				{
					const double d = x[i] - x[i - 1];
					return y[i - 1] + (d > 0 ? (xx - x[i - 1]) / d : 0.0) * (y[i] - y[i - 1]);
				}
			return y.back();
		}
	};

	// ---- TV-fitted colour matrix (--from-calib) ----

	struct TvPatch
	{
		int rgb[3];
		bool peak;
		cv::Vec3d raw; // black-subtracted, normalised to the 10-bit full scale
		cv::Vec3d rawSat;
		cv::Vec3d isp; // 0..255, tuning in effect during tv-raw
	};

	struct CcmFit
	{
		cv::Matx33d ccm;	  // the matrix to write (applies to WB-gained RAW)
		double eps = 0.0;	  // safety desaturation used
		double scale = 0.0;	  // exposure scale (peak -> target)
		cv::Vec3d alsc{1, 1, 1}; // per-raw-channel ISP gain at the TV ROI (lens shading etc.)
		double alscResid = 0.0;
		cv::Vec3d peakOut, whiteOut; // predicted ISP-linear outputs
		double wb[2] = {1.0, 1.0};
		double dgain = 1.0;
		int negBefore = 0; // bright patches with a channel < 0 under the CURRENT matrix
	};

	bool fitTvMatrix(const MakeTuningConfig &cfg, const json::value &baseRoot, CcmFit &out)
	{
		const std::string dir = cfg.fromCalib;
		CsvTable t;
		if (!loadCsvTable(dir + "/tv_raw.csv", t))
		{
			std::cout << "ERROR: --from-calib: cannot read " << dir << "/tv_raw.csv (run --ledcalib tv-raw first)"
					  << std::endl;
			return false;
		}
		json::value meta;
		if (!parseJsonFile(dir + "/meta_tv-raw.json", meta) || !meta.is_object())
			return false;
		const json::object &mo = meta.get_object();
		auto fr = mo.find("frame");
		if (fr == mo.end() || !fr->value().is_object())
		{
			std::cout << "ERROR: meta_tv-raw.json has no frame metadata" << std::endl;
			return false;
		}
		const json::object &fo = fr->value().get_object();
		cv::Matx33d ccmNow = cv::Matx33d::eye();
		bool haveCcm = false;
		if (auto c = fo.find("ccm"); c != fo.end() && c->value().is_array() && c->value().get_array().size() == 9)
		{
			for (int i = 0; i < 9; i++)
				ccmNow.val[i] = num(c->value().get_array()[i]);
			haveCcm = true;
		}
		if (auto g = fo.find("colour_gains"); g != fo.end() && g->value().is_array() && g->value().get_array().size() == 2)
		{
			out.wb[0] = num(g->value().get_array()[0]);
			out.wb[1] = num(g->value().get_array()[1]);
		}
		if (auto d = fo.find("digital_gain"); d != fo.end())
			out.dgain = std::max(1e-3, num(d->value()));
		if (!haveCcm || out.wb[0] <= 0 || out.wb[1] <= 0)
		{
			std::cout << "ERROR: meta_tv-raw.json lacks the CCM / colour gains in effect" << std::endl;
			return false;
		}

		// The tone curve in effect during tv-raw: the tuning file it recorded,
		// else the stock base.
		ToneCurve curve;
		{
			std::string tf;
			if (auto it = mo.find("tuning_file"); it != mo.end() && it->value().is_string())
				tf = std::string(it->value().get_string());
			json::value tr;
			json::value *src = const_cast<json::value *>(&baseRoot);
			if (!tf.empty() && parseJsonFile(tf, tr))
				src = &tr;
			if (!curve.load(findAlgo(*src, "rpi.contrast")))
			{
				std::cout << "ERROR: no gamma_curve in the tuning used for tv-raw" << std::endl;
				return false;
			}
			// The per-channel ISP gain below is fitted from the tv-raw ISP readings
			// decoded with this static curve: an adaptive tuning (stock: contrast
			// enhancement + ALSC iterations) biases it by a few %, and production
			// (frozen) will not reproduce that state.
			const std::string adaptive = adaptiveStagesOf(*src);
			if (!adaptive.empty())
				std::cout << "WARNING: tv-raw ran under a tuning with " << adaptive
						  << " enabled (" << (tf.empty() ? "the stock file" : tf)
						  << "); the ISP readings the matrix is fitted to were scene-adaptive. Rerun tv-raw under "
							 "an interim frozen tuning (--make-tuning interim.json without --from-calib, then "
							 "--tuning interim.json --saturation 1)."
						  << std::endl;
		}

		std::vector<TvPatch> patches;
		const char *ch[3] = {"r", "g", "b"};
		for (std::size_t i = 0; i < t.rows.size(); i++)
		{
			TvPatch p;
			p.rgb[0] = static_cast<int>(t.get(i, "patch_r"));
			p.rgb[1] = static_cast<int>(t.get(i, "patch_g"));
			p.rgb[2] = static_cast<int>(t.get(i, "patch_b"));
			p.peak = t.get(i, "peak") > 0.5;
			for (int c = 0; c < 3; c++)
			{
				p.raw[c] = t.get(i, std::string("raw_") + ch[c]) / 1023.0;
				p.rawSat[c] = t.get(i, std::string("raw_") + ch[c] + "_sat");
				p.isp[c] = t.get(i, std::string("isp_") + ch[c]);
			}
			patches.push_back(p);
		}
		auto find = [&](int r, int g, int b, bool peak) -> const TvPatch *
		{
			for (const auto &p : patches)
				if (p.rgb[0] == r && p.rgb[1] == g && p.rgb[2] == b && p.peak == peak)
					return &p;
			return nullptr;
		};
		const TvPatch *pR = find(255, 0, 0, false), *pG = find(0, 255, 0, false), *pB = find(0, 0, 255, false);
		const TvPatch *pW = find(255, 255, 255, false), *pPk = find(255, 255, 255, true);
		if (!pR || !pG || !pB || !pW)
		{
			std::cout << "ERROR: tv_raw.csv lacks the full-level primaries / white" << std::endl;
			return false;
		}
		if (std::max({pW->raw[0], pW->raw[1], pW->raw[2]}) < 30.0 / 1023.0)
		{
			std::cout << "ERROR: the TV white patch is too dark in RAW (" << pW->raw[0] * 1023 << ","
					  << pW->raw[1] * 1023 << "," << pW->raw[2] * 1023
					  << " counts): is the TV on and showing the page, and the TV ROI on the patch?" << std::endl;
			return false;
		}
		for (const TvPatch *p : {pR, pG, pB, pW})
			for (int c = 0; c < 3; c++)
				if (p->rawSat[c] > 0.01)
				{
					std::cout << "ERROR: the TV patch " << p->rgb[0] << "," << p->rgb[1] << "," << p->rgb[2]
							  << " clips the SENSOR (RAW). Shorten the shutter (-s; the gain is already at the "
								 "sensor minimum) and rerun tv-raw."
							  << std::endl;
					return false;
				}
		// The matrix is fitted on the light exactly as the ISP will see it (TV
		// black / ambient included: ~0 on OLED, and the ISP has no way to remove
		// it either), so every check below is on what production will output.
		const cv::Vec3d black(0.0, 0.0, 0.0);

		// Primaries as columns.
		cv::Matx33d P;
		const TvPatch *prim[3] = {pR, pG, pB};
		for (int j = 0; j < 3; j++)
			for (int c = 0; c < 3; c++)
				P(c, j) = prim[j]->raw[c] - black[c];
		if (std::fabs(cv::determinant(P)) < 1e-12)
		{
			std::cout << "ERROR: TV primaries are degenerate in RAW (is the TV ROI on the TV?)" << std::endl;
			return false;
		}
		const cv::Matx33d M0 = P.inv();
		const cv::Vec3d w0 = M0 * (pW->raw - black); // WOLED white subpixel: not exactly (1,1,1)
		cv::Matx33d D = cv::Matx33d::zeros();
		for (int c = 0; c < 3; c++)
			D(c, c) = 1.0 / std::max(w0[c], 1e-6);
		const cv::Matx33d M1 = D * M0; // TV white -> (1,1,1), primaries on the axes

		// How many bright patches the CURRENT matrix pushes below zero (the
		// motivation; reported for the record).
		for (const auto &p : patches)
			if (!p.peak && std::max({p.rgb[0], p.rgb[1], p.rgb[2]}) >= 64 && std::min({p.isp[0], p.isp[1], p.isp[2]}) <= 0.6)
				out.negBefore++;

		// Safety desaturation: smallest eps >= cfg.desat with every bright patch's
		// channels >= 0.5% of its brightest channel (noise / non-additivity margin).
		auto desatM = [&](double e)
		{
			cv::Matx33d J;
			for (int i = 0; i < 9; i++)
				J.val[i] = 1.0 / 3.0;
			return ((1.0 - e) * cv::Matx33d::eye() + e * J) * M1;
		};
		double eps = cfg.desat;
		for (; eps <= 0.30001; eps += 0.005)
		{
			const cv::Matx33d M = desatM(eps);
			bool okAll = true;
			for (const auto &p : patches)
			{
				if (p.peak || std::max({p.rgb[0], p.rgb[1], p.rgb[2]}) < 64)
					continue;
				const cv::Vec3d o = M * (p.raw - black);
				const double mx = std::max({o[0], o[1], o[2]});
				if (mx > 0 && std::min({o[0], o[1], o[2]}) < 0.005 * mx)
				{
					okAll = false;
					break;
				}
			}
			if (okAll)
				break;
		}
		if (eps > 0.30001)
		{
			std::cout << "WARNING: some TV patches stay below zero even at 30% desaturation (TV not additive in "
						 "RAW?); using 30%." << std::endl;
			eps = 0.30;
		}
		out.eps = eps;
		const cv::Matx33d M2 = desatM(eps);

		// Per-raw-channel ISP gain at the TV ROI (black-level rescale, lens
		// shading, ...): fit  f^-1(isp) = CCMnow * diag(wb*dg) * diag(a) * raw
		// over mid-range patches (where the tv-raw tone curve is trustworthy).
		const cv::Vec3d wbv{out.wb[0] * out.dgain, out.dgain, out.wb[1] * out.dgain};
		cv::Matx33d Q = ccmNow;
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				Q(r, c) *= wbv[c];
		// The ISP applies the colour gains (x digital gain x lens shading) BEFORE
		// the matrix and clips there at full scale: with -b 2.6,1.6 a TV red
		// saturates the ISP's red at ~38% of the sensor range, invisible to the
		// sensor-clip check and to the post-matrix readings. Pre-matrix value of
		// raw channel c: wb*dg*a*raw. Patches clipped there are dropped from the
		// fit (one refit), and a clipped peak / bright patch is reported.
		auto preCcm = [&](const TvPatch &p, int c) { return wbv[c] * out.alsc[c] * p.raw[c]; };
		auto preClipped = [&](const TvPatch &p)
		{ return std::max({preCcm(p, 0), preCcm(p, 1), preCcm(p, 2)}) > 0.97; };
		bool haveA = false;
		for (int pass = 0; pass < 2; pass++)
		{
			std::vector<double> Arows, brows;
			int nEq = 0;
			for (const auto &p : patches)
			{
				if (p.peak || (haveA && preClipped(p)))
					continue;
				for (int r = 0; r < 3; r++)
				{
					if (p.isp[r] < 40.0 || p.isp[r] > 200.0)
						continue;
					for (int c = 0; c < 3; c++)
						Arows.push_back(Q(r, c) * p.raw[c]);
					brows.push_back(curve.inverse(p.isp[r] / 255.0));
					nEq++;
				}
			}
			if (nEq < 6)
			{
				std::cout << "WARNING: too few mid-range TV patches to measure the ISP gain; assuming 1" << std::endl;
				break;
			}
			cv::Mat A(nEq, 3, CV_64F, Arows.data()), b(nEq, 1, CV_64F, brows.data()), a;
			if (!cv::solve(A, b, a, cv::DECOMP_SVD))
				break;
			for (int c = 0; c < 3; c++)
				out.alsc[c] = a.at<double>(c);
			cv::Mat r = A * a - b;
			double ss = 0.0, sb = 0.0;
			for (int i = 0; i < nEq; i++)
			{
				ss += r.at<double>(i) * r.at<double>(i);
				sb += b.at<double>(i) * b.at<double>(i);
			}
			out.alscResid = std::sqrt(ss / std::max(sb, 1e-12));
			haveA = true;
			bool any = false;
			for (const auto &p : patches)
				any = any || (!p.peak && preClipped(p));
			if (!any)
				break; // nothing to drop: the first fit stands
		}
		if (haveA)
		{
			int nPre = 0;
			for (const auto &p : patches)
				if (!p.peak && preClipped(p))
					nPre++;
			if (nPre > 0)
				std::cout << "WARNING: " << nPre << " TV patches clip the ISP's colour-gain stage (before the "
						  << "matrix; colour gains " << out.wb[0] << "," << out.wb[1]
						  << "). Rerun tv-raw with -b 1,1: the fitted matrix absorbs the gains, so the ISP then "
							 "clips only where the sensor does."
						  << std::endl;
		}
		for (int c = 0; c < 3; c++)
			if (!(out.alsc[c] > 0.05 && out.alsc[c] < 50.0))
			{
				std::cout << "ERROR: implausible ISP gain at the TV ROI (" << out.alsc[0] << "," << out.alsc[1] << ","
						  << out.alsc[2] << "); is tv_raw.csv from this camera setup?" << std::endl;
				return false;
			}

		// Exposure scale: the peak-white window lands at cfg.peakTarget.
		const TvPatch *pk = pPk ? pPk : pW;
		if (!pPk)
			std::cout << "WARNING: no peak-white window patch; scaling to full-field white instead" << std::endl;
		const cv::Vec3d peakM = M2 * (pk->raw - black);
		out.scale = cfg.peakTarget / std::max({peakM[0], peakM[1], peakM[2], 1e-9});
		if (pk->rawSat[0] > 0.01 || pk->rawSat[1] > 0.01 || pk->rawSat[2] > 0.01)
			std::cout << "WARNING: the peak-white window clips the SENSOR; highlights will still clip. Shorten the "
						 "shutter (-s) and rerun tv-raw."
					  << std::endl;
		else if (haveA && preClipped(*pk))
			std::cout << "WARNING: the peak-white window clips the ISP's colour-gain stage (pre-matrix "
					  << preCcm(*pk, 0) << "," << preCcm(*pk, 1) << "," << preCcm(*pk, 2)
					  << " of full scale): the 95% peak target is unreachable. Rerun tv-raw with -b 1,1." << std::endl;

		// ISP-linear = CCMnew * diag(wb*dg) * diag(a) * raw  ==  scale * M2 * raw
		cv::Matx33d S = cv::Matx33d::zeros();
		for (int c = 0; c < 3; c++)
			S(c, c) = 1.0 / (wbv[c] * out.alsc[c]);
		out.ccm = out.scale * M2 * S;
		out.peakOut = out.scale * M2 * pk->raw;
		out.whiteOut = out.scale * M2 * pW->raw;
		return true;
	}

	std::string platformDir()
	{
		std::string compat;
		readText("/proc/device-tree/compatible", compat);
		return (compat.find("bcm2712") != std::string::npos) ? "pisp" : "vc4";
	}
}

std::string findSystemTuningFile(const std::string &sensorModel)
{
	const std::string plat = platformDir();
	for (const char *root : {"/usr/local/share/libcamera/ipa/rpi/", "/usr/share/libcamera/ipa/rpi/"})
	{
		const std::string p = std::string(root) + plat + "/" + sensorModel + ".json";
		if (std::filesystem::exists(p))
			return p;
	}
	return "";
}

std::string fileHash(const std::string &path)
{
	std::string text;
	if (path.empty() || !readText(path, text))
		return "";
	uint64_t h = 1469598103934665603ull;
	for (unsigned char c : text)
	{
		h ^= c;
		h *= 1099511628211ull;
	}
	char buf[17];
	std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
	return buf;
}

bool loadTuningToneCurve(const std::string &path, std::vector<double> &linOfCode)
{
	linOfCode.clear();
	json::value root;
	if (path.empty() || !parseJsonFile(path, root))
		return false;
	ToneCurve tc;
	if (!tc.load(findAlgo(root, "rpi.contrast")))
		return false;
	// vc4 evaluates the curve at its 33 hardware knots and interpolates linearly
	// between them; pisp resamples finely (~ the curve's own PWL).
	ToneCurve hw;
	if (platformDir() == "vc4")
		for (int k : curveKnots())
		{
			hw.x.push_back(k / 65535.0);
			hw.y.push_back(tc.forward(k / 65535.0));
		}
	else
		hw = tc;
	linOfCode.resize(256);
	for (int z = 0; z < 256; z++)
		linOfCode[z] = std::clamp(hw.inverse(z / 255.0), 0.0, 1.0);
	return linOfCode[255] > 0.0;
}

int runMakeTuning(const MakeTuningConfig &cfg)
{
	std::string base = cfg.basePath.empty() ? findSystemTuningFile(cfg.sensorModel) : cfg.basePath;
	if (base.empty())
	{
		std::cout << "ERROR: no stock tuning file found for sensor '" << cfg.sensorModel << "' (platform "
				  << platformDir() << "); pass --tuning-base FILE" << std::endl;
		return EXIT_FAILURE;
	}
	if (std::filesystem::exists(cfg.outPath) && std::filesystem::equivalent(cfg.outPath, base))
	{
		std::cout << "ERROR: --make-tuning would overwrite its own base file" << std::endl;
		return EXIT_FAILURE;
	}
	json::value root, pristine; // pristine: the unedited base (its tone curve decodes tv-raw readings)
	if (!parseJsonFile(base, root) || !parseJsonFile(base, pristine))
		return EXIT_FAILURE;
	if (root.is_object())
		if (auto h = root.get_object().find("hyperion_cam"); h != root.get_object().end())
			std::cout << "WARNING: the base tuning " << base << " is already a hyperion-cam frozen file" << std::endl;

	json::object *contrast = findAlgo(root, "rpi.contrast");
	json::object *alsc = findAlgo(root, "rpi.alsc");
	json::object *ccm = findAlgo(root, "rpi.ccm");
	if (!contrast)
	{
		std::cout << "ERROR: " << base << " has no rpi.contrast block (unsupported tuning format)" << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "Base tuning: " << base << std::endl;

	// 1. Contrast: no adaptive enhancement; power-law tone curve.
	(*contrast)["ce_enable"] = 0;
	std::string curveDesc = "stock";
	if (cfg.curve != "stock")
	{
		char *end = nullptr;
		const double g = std::strtod(cfg.curve.c_str(), &end);
		if (!end || *end || !(g >= 1.0 && g <= 4.0))
		{
			std::cout << "ERROR: --tuning-curve must be 'stock' or an exponent in [1,4]" << std::endl;
			return EXIT_FAILURE;
		}
		json::array gc;
		for (int x : curveKnots())
		{
			gc.push_back(x);
			gc.push_back(static_cast<int64_t>(std::lround(65535.0 * std::pow(x / 65535.0, 1.0 / g))));
		}
		(*contrast)["gamma_curve"] = std::move(gc);
		curveDesc = "power " + cfg.curve;
	}
	std::cout << "  rpi.contrast: ce_enable 0, tone curve " << curveDesc << std::endl;

	// 2. Lens shading: calibrated tables only, no per-frame adaptation.
	if (alsc)
	{
		(*alsc)["n_iter"] = 0;
		std::cout << "  rpi.alsc: n_iter 0 (static calibrated tables)" << std::endl;
	}
	else
		std::cout << "  rpi.alsc: absent (nothing to freeze)" << std::endl;

	// 3. Colour matrix fitted to the TV.
	std::string ccmDesc = "stock";
	CcmFit fit;
	if (!cfg.fromCalib.empty())
	{
		if (!ccm)
		{
			std::cout << "ERROR: " << base << " has no rpi.ccm block" << std::endl;
			return EXIT_FAILURE;
		}
		if (!fitTvMatrix(cfg, pristine, fit))
			return EXIT_FAILURE;
		double ct = 5000.0;
		if (auto it = ccm->find("ccms"); it != ccm->end() && it->value().is_array() && !it->value().get_array().empty())
			if (const json::value &e0 = it->value().get_array()[0]; e0.is_object() && e0.get_object().contains("ct"))
				ct = num(e0.get_object().at("ct"));
		json::array m;
		for (int i = 0; i < 9; i++)
			m.push_back(std::round(fit.ccm.val[i] * 1e5) / 1e5);
		json::object entry;
		entry["ct"] = ct;
		entry["ccm"] = std::move(m);
		json::array ccms;
		ccms.push_back(std::move(entry));
		(*ccm)["ccms"] = std::move(ccms);
		ccmDesc = "TV-fitted (" + cfg.fromCalib + ")";
		double maxCoef = 0.0;
		for (double v : fit.ccm.val)
			maxCoef = std::max(maxCoef, std::fabs(v));
		std::cout << "  rpi.ccm: single TV-fitted matrix (safety desaturation " << (100.0 * fit.eps)
				  << "%, peak-white -> " << cfg.peakTarget << ", exposure scale " << fit.scale << ")\n"
				  << "    ISP gain at the TV ROI " << fit.alsc[0] << "," << fit.alsc[1] << "," << fit.alsc[2]
				  << " (model residual " << (100.0 * fit.alscResid) << "%)\n"
				  << "    stock matrix pushed " << fit.negBefore << " bright TV patches to 0 in some channel; the new "
				  << "matrix keeps every measured patch >= 0.5% of its brightest channel" << std::endl;
		if (maxCoef > 7.9)
			std::cout << "WARNING: matrix coefficient " << maxCoef << " is beyond the ISP's range (+-8); lengthen "
						 "the shutter instead of relying on matrix gain." << std::endl;
	}
	std::cout << "  rpi.ccm: " << ccmDesc << std::endl;

	// Provenance (ignored by libcamera: it only reads version/target/algorithms).
	json::object prov;
	prov["frozen"] = true;
	prov["base"] = base;
	prov["base_hash"] = fileHash(base);
	prov["curve"] = curveDesc;
	prov["ccm"] = ccmDesc;
	if (!cfg.fromCalib.empty())
	{
		prov["colour_gains"] = json::array{fit.wb[0], fit.wb[1]};
		prov["desat"] = fit.eps;
		prov["exposure_scale"] = fit.scale;
	}
	root.get_object()["hyperion_cam"] = std::move(prov);

	{
		std::ofstream f(cfg.outPath);
		if (!f)
		{
			std::cout << "ERROR: cannot write " << cfg.outPath << std::endl;
			return EXIT_FAILURE;
		}
		writeJson(f, root, 0);
		f << "\n";
	}
	std::cout << "Wrote " << cfg.outPath << " (hash " << fileHash(cfg.outPath) << ")" << std::endl;

	if (!cfg.fromCalib.empty())
	{
		// Predicted readings with the new file (TV-linear, then encoded).
		const double g = (cfg.curve == "stock") ? 0.0 : std::strtod(cfg.curve.c_str(), nullptr);
		auto enc = [&](double l)
		{ return g > 0 ? 255.0 * std::pow(std::clamp(l, 0.0, 1.0), 1.0 / g) : -1.0; };
		std::ostringstream pr;
		pr << std::fixed << std::setprecision(3) << "  predicted ISP-linear: full-field white " << fit.whiteOut[0] << ","
		   << fit.whiteOut[1] << "," << fit.whiteOut[2] << "   peak window " << fit.peakOut[0] << ","
		   << fit.peakOut[1] << "," << fit.peakOut[2];
		std::cout << pr.str() << std::endl;
		if (g > 0)
			std::cout << "  -> encoded: full-field white ~" << std::lround(enc(fit.whiteOut[1])) << ", peak ~"
					  << std::lround(enc(fit.peakOut[1])) << std::endl;
		std::cout << "\nUse it everywhere from now on (production service AND every calibration phase):\n"
				  << "  --tuning " << cfg.outPath << " -b " << fit.wb[0] << "," << fit.wb[1]
				  << " --saturation 1\n"
				  << "(the matrix already contains the colour gains and the saturation; a different -b or\n"
				  << " --saturation would be applied ON TOP of it). Then re-measure shading, response, tv, led."
				  << std::endl;
		if (std::fabs(fit.wb[0] - 1.0) > 1e-3 || std::fabs(fit.wb[1] - 1.0) > 1e-3)
			std::cout << "NOTE: tv-raw ran with colour gains " << fit.wb[0] << "," << fit.wb[1]
					  << ". The ISP applies them BEFORE the matrix and clips there, so anything redder/bluer than "
						 "white (e.g. the LED glow) saturates long before the sensor does. Prefer rerunning tv-raw "
						 "with -b 1,1 (the matrix then carries the gains)."
					  << std::endl;
	}
	else
		std::cout << "\nUse it with --tuning " << cfg.outPath
				  << " in the production service and every calibration phase." << std::endl;
	return EXIT_SUCCESS;
}
