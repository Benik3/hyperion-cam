#include <iostream>
#include <pthread.h>
#include <sys/mman.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <boost/program_options.hpp>
#include <libcamera/libcamera.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "event_loop.h"
#include "FlatBufferConnection.h"
#include "opencvTransform.h"

#ifdef HAVE_GPU_TRANSFORM
#include "gpuNv12Transform.h"
#include "gpuRgbTransform.h"
#endif

using namespace libcamera;
namespace po = boost::program_options;

static std::shared_ptr<Camera> _camera;
bool _useNv12 = false;
bool _useGpu = false;

static int _strideY = 0;
static int _strideUV = 0;
static int _strideBgr = 0;
static int _bppBgr = 3;
static bool _swapRb4 = false;

struct Nv12RingBuffer
{
	std::vector<std::vector<uint8_t>> y;
	std::vector<std::vector<uint8_t>> uv;
	std::size_t index{0};
	int width{0};
	int height{0};

	void reset(int w, int h, std::size_t ringSize = 3)
	{
		width = w;
		height = h;
		index = 0;
		y.assign(ringSize, std::vector<uint8_t>(static_cast<std::size_t>(w) * static_cast<std::size_t>(h)));
		uv.assign(ringSize, std::vector<uint8_t>(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) / 2u));
	}

	struct View
	{
		uint8_t *y;
		uint8_t *uv;
		int strideY;
		int strideUV;
		int width;
		int height;
	};

	View next()
	{
		std::size_t idx = index++ % y.size();
		return View{y[idx].data(), uv[idx].data(), width, width, width, height};
	}
};

static Nv12RingBuffer _nv12Out;

struct RgbRingBuffer
{
	std::vector<std::vector<uint8_t>> rgb;
	std::size_t index{0};
	int width{0};
	int height{0};

	void reset(int w, int h, std::size_t ringSize = 3)
	{
		width = w;
		height = h;
		index = 0;
		rgb.assign(ringSize, std::vector<uint8_t>(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u));
	}

	struct View
	{
		uint8_t *rgb;
		int width;
		int height;
	};

	View next()
	{
		std::size_t idx = index++ % rgb.size();
		return View{rgb[idx].data(), width, height};
	}
};

static RgbRingBuffer _rgbOut;

std::map<libcamera::FrameBuffer *, std::vector<libcamera::Span<uint8_t>>> _mappedBuffers;
static EventLoop _loop;
static EventLoop _loopFb;
static EventLoop _loopTr;
FlatBufferConnection _flatbuf;
OpencvTransform *_transformation;

#ifdef HAVE_GPU_TRANSFORM
GpuNv12Transform *_gpuNv12 = nullptr;
GpuRgbTransform *_gpuRgb = nullptr;
#endif
cv::Point2f _imgQuad[4];
int _imgWidth = 0;
int _imgHeight = 0;
int _imgWidthToSend;
int _imgHeightToSend;

int _width;
int _height;
int32_t _shutter;
float _gain;
std::string _wbgainsStr;
float _rGain;
float _bGain;
std::string _origin;
std::string _conn;
int _port;
int _priority;
std::string _transformStr;
bool _transformEn = false;
bool _verbose = false;
bool _help = false;

static bool envEnabled(const char *name)
{
	const char *v = std::getenv(name);
	if (!v)
		return false;
	return (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
}

static inline int64_t nowNs()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
			   std::chrono::high_resolution_clock::now().time_since_epoch())
		.count();
}

struct CpuProfileState
{
	bool enabled{false};
	uint64_t lastPrintNs{0};
	uint64_t frames{0};
	uint64_t accXformNs{0};
};

static inline void cpuProfileTick(CpuProfileState &s, const char *tag, uint64_t xformNs)
{
	if (!s.enabled)
		return;

	s.accXformNs += xformNs;
	s.frames++;

	uint64_t now = static_cast<uint64_t>(nowNs());
	if (s.lastPrintNs == 0)
		s.lastPrintNs = now;

	if (now - s.lastPrintNs > 1000000000ull)
	{
		double denom = s.frames ? static_cast<double>(s.frames) : 1.0;
		double avgMs = (static_cast<double>(s.accXformNs) / denom) / 1e6;
		std::cerr << tag << " avg ms: xform=" << avgMs << std::endl;
		s.frames = 0;
		s.accXformNs = 0;
		s.lastPrintNs = now;
	}
}

static CpuProfileState _cpuRgbProf{envEnabled("HYPERCAM_CPU_PROFILE")};
static CpuProfileState _cpuNv12Prof{envEnabled("HYPERCAM_CPU_PROFILE")};

static std::atomic<int> _pendingProcess{0};
static std::atomic<int> _pendingSend{0};

void *flatLoop(void *arg)
{
	_loopFb.exec();
	return 0;
}

static void flatBufSend(const uint8_t *img)
{
	const int64_t startNs = nowNs();
	if (_flatbuf.connectToServer(_origin.c_str(), _conn.c_str(), _port, _priority, !_verbose))
	{
		if (_flatbuf.setImage(img, _imgWidthToSend, _imgHeightToSend) < 0)
		{
			std::cout << "ERROR: Flatbuffer write timeout" << std::endl;
		}
	}
	else
		std::cout << "ERROR: Flatbuffer connection timeout" << std::endl;

	int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;

		char strTime[100];
		time_t t = time(NULL);
		struct tm *p = localtime(&t);
		strftime(strTime, 100, "%F %T", p);

		std::cout << strTime << " - WARNING: Frame sending was too slow: " << elapsedMicros << " us" << std::endl;
	}
}

static void flatBufSendNv12(const uint8_t *y, const uint8_t *uv, int strideY, int strideUV, int width, int height)
{
	const int64_t startNs = nowNs();
	if (_flatbuf.connectToServer(_origin.c_str(), _conn.c_str(), _port, _priority, !_verbose))
	{
		if (_flatbuf.setNV12Image(y, uv, width, height, strideY, strideUV) < 0)
		{
			std::cout << "ERROR: Flatbuffer write timeout" << std::endl;
		}
	}
	else
		std::cout << "ERROR: Flatbuffer connection timeout" << std::endl;

	int before = _pendingSend.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;

		char strTime[100];
		time_t t = time(NULL);
		struct tm *p = localtime(&t);
		strftime(strTime, 100, "%F %T", p);

		std::cout << strTime << " - WARNING: Frame sending was too slow: " << elapsedMicros << " us" << std::endl;
	}
}
void *transformLoop(void *arg)
{
	_transformation = new OpencvTransform(_imgWidth, _imgHeight, _imgQuad, &_imgWidthToSend, &_imgHeightToSend, _useNv12);
	if (_useNv12)
		_nv12Out.reset(_imgWidthToSend, _imgHeightToSend);
	else
		_rgbOut.reset(_imgWidthToSend, _imgHeightToSend);

#ifdef HAVE_GPU_TRANSFORM
	if (_useNv12 && _useGpu)
		_gpuNv12 = new GpuNv12Transform(_imgWidth, _imgHeight, _imgQuad, _imgWidthToSend, _imgHeightToSend);
	if (!_useNv12 && _useGpu)
		_gpuRgb = new GpuRgbTransform(_imgWidth, _imgHeight, _imgQuad, _imgWidthToSend, _imgHeightToSend);
#endif
	_loopTr.exec();
	return 0;
}

static void transformImg(uint8_t *img)
{
	const int64_t startNs = nowNs();
	uint8_t *imgToSend = _transformation->transform(img);
	cpuProfileTick(_cpuRgbProf, "[cpuRgb]", static_cast<uint64_t>(nowNs() - startNs));

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;

		char strTime[100];
		time_t t = time(NULL);
		struct tm *p = localtime(&t);
		strftime(strTime, 100, "%F %T", p);

		std::cout << strTime << " - WARNING: Frame transformation was too slow: " << elapsedMicros << " us" << std::endl;
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	_loopFb.callLater([imgToSend]() { flatBufSend(imgToSend); });
}

static void transformImgNv12(uint8_t *y, uint8_t *uv)
{
	const int64_t startNs = nowNs();
	// Avoid NV12 <-> BGR conversions: warp Y and UV planes directly.
	// UV is half-resolution, so use maps computed in UV coordinate space.
	static thread_local bool inited = false;
	static thread_local cv::Mat mapYx, mapYy;
	static thread_local cv::Mat mapUVx, mapUVy;
	if (!inited)
	{
		cv::Point2f dstQuadY[4] = {
			cv::Point2f(0.f, 0.f),
			cv::Point2f(static_cast<float>(_imgWidthToSend), 0.f),
			cv::Point2f(0.f, static_cast<float>(_imgHeightToSend)),
			cv::Point2f(static_cast<float>(_imgWidthToSend), static_cast<float>(_imgHeightToSend))};
		cv::Mat warpY = cv::getPerspectiveTransform(_imgQuad, dstQuadY);
		cv::Mat warpYInv;
		cv::invert(warpY, warpYInv);
		mapYx.create(_imgHeightToSend, _imgWidthToSend, CV_32FC1);
		mapYy.create(_imgHeightToSend, _imgWidthToSend, CV_32FC1);

		const double y00 = warpYInv.at<double>(0, 0);
		const double y01 = warpYInv.at<double>(0, 1);
		const double y02 = warpYInv.at<double>(0, 2);
		const double y10 = warpYInv.at<double>(1, 0);
		const double y11 = warpYInv.at<double>(1, 1);
		const double y12 = warpYInv.at<double>(1, 2);
		const double y20 = warpYInv.at<double>(2, 0);
		const double y21 = warpYInv.at<double>(2, 1);
		const double y22 = warpYInv.at<double>(2, 2);

		for (int dy = 0; dy < _imgHeightToSend; dy++)
		{
			float *mx = mapYx.ptr<float>(dy);
			float *my = mapYy.ptr<float>(dy);
			for (int dx = 0; dx < _imgWidthToSend; dx++)
			{
				const double denom = y20 * dx + y21 * dy + y22;
				if (denom == 0.0)
				{
					mx[dx] = -1.f;
					my[dx] = -1.f;
					continue;
				}
				mx[dx] = static_cast<float>((y00 * dx + y01 * dy + y02) / denom);
				my[dx] = static_cast<float>((y10 * dx + y11 * dy + y12) / denom);
			}
		}

		cv::Point2f srcQuadUV[4];
		cv::Point2f dstQuadUV[4] = {
			cv::Point2f(0.f, 0.f),
			cv::Point2f(static_cast<float>(_imgWidthToSend) / 2.f, 0.f),
			cv::Point2f(0.f, static_cast<float>(_imgHeightToSend) / 2.f),
			cv::Point2f(static_cast<float>(_imgWidthToSend) / 2.f, static_cast<float>(_imgHeightToSend) / 2.f)};
		for (int i = 0; i < 4; i++)
			srcQuadUV[i] = cv::Point2f(_imgQuad[i].x / 2.f, _imgQuad[i].y / 2.f);
		cv::Mat warpUV = cv::getPerspectiveTransform(srcQuadUV, dstQuadUV);
		cv::Mat warpUVInv;
		cv::invert(warpUV, warpUVInv);

		const int uvW = _imgWidthToSend / 2;
		const int uvH = _imgHeightToSend / 2;
		mapUVx.create(uvH, uvW, CV_32FC1);
		mapUVy.create(uvH, uvW, CV_32FC1);

		const double u00 = warpUVInv.at<double>(0, 0);
		const double u01 = warpUVInv.at<double>(0, 1);
		const double u02 = warpUVInv.at<double>(0, 2);
		const double u10 = warpUVInv.at<double>(1, 0);
		const double u11 = warpUVInv.at<double>(1, 1);
		const double u12 = warpUVInv.at<double>(1, 2);
		const double u20 = warpUVInv.at<double>(2, 0);
		const double u21 = warpUVInv.at<double>(2, 1);
		const double u22 = warpUVInv.at<double>(2, 2);

		for (int dy = 0; dy < uvH; dy++)
		{
			float *mx = mapUVx.ptr<float>(dy);
			float *my = mapUVy.ptr<float>(dy);
			for (int dx = 0; dx < uvW; dx++)
			{
				const double denom = u20 * dx + u21 * dy + u22;
				if (denom == 0.0)
				{
					mx[dx] = -1.f;
					my[dx] = -1.f;
					continue;
				}
				mx[dx] = static_cast<float>((u00 * dx + u01 * dy + u02) / denom);
				my[dx] = static_cast<float>((u10 * dx + u11 * dy + u12) / denom);
			}
		}

		inited = true;
	}

	cv::Mat yIn(_imgHeight, _imgWidth, CV_8UC1, y, _strideY);
	cv::Mat uvIn(_imgHeight / 2, _imgWidth / 2, CV_8UC2, uv, _strideUV);

	auto view = _nv12Out.next();
	cv::Mat yOut(_imgHeightToSend, _imgWidthToSend, CV_8UC1, view.y, view.strideY);
	cv::Mat uvOut(_imgHeightToSend / 2, _imgWidthToSend / 2, CV_8UC2, view.uv, view.strideUV);

	cv::remap(yIn, yOut, mapYx, mapYy, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::remap(uvIn, uvOut, mapUVx, mapUVy, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cpuProfileTick(_cpuNv12Prof, "[cpuNv12]", static_cast<uint64_t>(nowNs() - startNs));

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;

		char strTime[100];
		time_t t = time(NULL);
		struct tm *p = localtime(&t);
		strftime(strTime, 100, "%F %T", p);

		std::cout << strTime << " - WARNING: Frame transformation was too slow: " << elapsedMicros << " us" << std::endl;
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	_loopFb.callLater([view]() { flatBufSendNv12(view.y, view.uv, view.strideY, view.strideUV, view.width, view.height); });
}

static void transformImgNv12Gpu(uint8_t *y, uint8_t *uv)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)y;
	(void)uv;
	std::cout << "ERROR: GPU transform not compiled in" << std::endl;
	_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	return;
#else
	const int64_t startNs = nowNs();

	if (!_gpuNv12)
	{
		std::cout << "ERROR: GPU transform not initialized" << std::endl;
		_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
		return;
	}

	auto view = _nv12Out.next();
	_gpuNv12->transform(y, uv, _strideY, _strideUV, view.y, view.uv);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;

		char strTime[100];
		time_t t = time(NULL);
		struct tm *p = localtime(&t);
		strftime(strTime, 100, "%F %T", p);

		std::cout << strTime << " - WARNING: Frame transformation was too slow: " << elapsedMicros << " us" << std::endl;
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	_loopFb.callLater([view]() { flatBufSendNv12(view.y, view.uv, view.strideY, view.strideUV, view.width, view.height); });
#endif
}

static void transformImgRgbGpu(uint8_t *img)
{
#ifndef HAVE_GPU_TRANSFORM
	(void)img;
	std::cout << "ERROR: GPU transform not compiled in" << std::endl;
	_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	return;
#else
	const int64_t startNs = nowNs();

	if (!_gpuRgb)
	{
		std::cout << "ERROR: GPU transform not initialized" << std::endl;
		_pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
		return;
	}

	auto view = _rgbOut.next();
	GpuRgbTransform::InputFormat fmt = GpuRgbTransform::InputFormat::Packed3;
	if (_bppBgr == 4)
		fmt = _swapRb4 ? GpuRgbTransform::InputFormat::Xrgb8888 : GpuRgbTransform::InputFormat::Xbgr8888;
	_gpuRgb->transform(img, _strideBgr, fmt, view.rgb);

	int before = _pendingProcess.fetch_sub(1, std::memory_order_acq_rel);
	if (before > 1 && _verbose)
	{
		int64_t elapsedMicros = (nowNs() - startNs) / 1000;

		char strTime[100];
		time_t t = time(NULL);
		struct tm *p = localtime(&t);
		strftime(strTime, 100, "%F %T", p);

		std::cout << strTime << " - WARNING: Frame transformation was too slow: " << elapsedMicros << " us" << std::endl;
	}

	_pendingSend.fetch_add(1, std::memory_order_acq_rel);
	_loopFb.callLater([view]() { flatBufSend(view.rgb); });
#endif
}

static void processRequest(Request *request);

static void requestComplete(Request *request)
{
	if (request->status() == Request::RequestCancelled)
		return;

	_loop.callLater(std::bind(&processRequest, request));
}

static void processRequest(Request *request)
{
	const Request::BufferMap &buffers = request->buffers();
	for (auto bufferPair : buffers)
	{
		FrameBuffer *buffer = bufferPair.second;
		auto it = _mappedBuffers.find(buffer);
		if (it == _mappedBuffers.end() || it->second.empty())
			continue;

		if (_transformEn)
		{
			_pendingProcess.fetch_add(1, std::memory_order_acq_rel);

			if (_useNv12)
			{
				if (it->second.size() < 2)
					continue;
				uint8_t *y = it->second[0].data();
				uint8_t *uv = it->second[1].data();
				if (_useGpu)
					_loopTr.callLater([y, uv]() { transformImgNv12Gpu(y, uv); });
				else
					_loopTr.callLater([y, uv]() { transformImgNv12(y, uv); });
			}
			else
			{
				uint8_t *img = it->second[0].data();
				if (_useGpu)
					_loopTr.callLater([img]() { transformImgRgbGpu(img); });
				else
					_loopTr.callLater([img]() { transformImg(img); });
			}
		}
		else
		{
			_pendingSend.fetch_add(1, std::memory_order_acq_rel);

			if (_useNv12)
			{
				if (it->second.size() < 2)
					continue;
				uint8_t *y = it->second[0].data();
				uint8_t *uv = it->second[1].data();
				_loopFb.callLater([y, uv]() { flatBufSendNv12(y, uv, _strideY, _strideUV, _imgWidthToSend, _imgHeightToSend); });
			}
			else
			{
				uint8_t *img = it->second[0].data();
				_loopFb.callLater([img]() { flatBufSend(img); });
			}
		}
	}

	request->reuse(Request::ReuseBuffers);
	_camera->queueRequest(request);
}
// Main loop
int main(int argc, char *argv[])
{
	po::options_description options("Options");
	options.add_options()																										   //
		("width,w", po::value<int>(&_width)->default_value(640), "Width")														   //
		("height,h", po::value<int>(&_height)->default_value(480), "Height")													   //
		("shutter,s", po::value<int32_t>(&_shutter)->default_value(16666), "Shutter speed in us")								   //
		("gain,g", po::value<float>(&_gain)->default_value(1.0), "Camera gain")													   //
		("wbgain,b", po::value<std::string>(&_wbgainsStr)->default_value("1.0,1.0"), "White balance gains")						   //
		("origin,o", po::value<std::string>(&_origin)->default_value("camera"), "Name of the origin")							   //
		("connection,c", po::value<std::string>(&_conn)->default_value("127.0.0.1"), "Socket address (IP or /tmp/hyperhdr-domain") //
		("port,p", po::value<int>(&_port)->default_value(19400), "TCP port")													   //
		("priority,r", po::value<int>(&_priority)->default_value(120), "Hyperion priority")										   //
		("transform,t", po::value<std::string>(&_transformStr), "Picture transformation in format x1,y1:x2,y2:x3,y3:x4,y4")		   //
		("nv12", po::bool_switch(&_useNv12)->default_value(false), "Enable NV12 format output (default is BGR888)")			   //
		("gpu", po::bool_switch(&_useGpu)->default_value(false), "Use GPU for transform (requires --transform)")			   //
		("verbose,v", po::bool_switch(&_verbose)->default_value(false), "Verbose output")										   //
		("help", po::bool_switch(&_help)->default_value(false), "Print help")													   //
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, options), vm);
	po::notify(vm);

	if (_help)
	{
		std::cout << std::endl
				  << options << std::endl;
		return 0;
	}

	if (vm.count("transform"))
	{
		if (sscanf(_transformStr.c_str(), "%f,%f:%f,%f:%f,%f:%f,%f",
				   &_imgQuad[0].x, &_imgQuad[0].y, &_imgQuad[1].x, &_imgQuad[1].y, &_imgQuad[2].x, &_imgQuad[2].y, &_imgQuad[3].x, &_imgQuad[3].y) == 8)
		{
			_transformEn = true;
		}
		else
		{
			std::cout << "ERROR: Invalid value for transformation" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (_useGpu)
	{
		if (!_transformEn)
		{
			std::cout << "ERROR: --gpu requires --transform" << std::endl;
			return EXIT_FAILURE;
		}
#ifndef HAVE_GPU_TRANSFORM
		std::cout << "ERROR: --gpu requested but GPU transform support was not compiled in" << std::endl;
		return EXIT_FAILURE;
#endif
	}

	if (sscanf(_wbgainsStr.c_str(), "%f,%f", &_rGain, &_bGain) != 2)
	{
		std::cout << "ERROR: wrong white balance gains!" << std::endl;
		return EXIT_FAILURE;
	}

	// Create a Camera Manager.
	std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
	cm->start();

	if (_verbose)
	{
		std::cout << "Camera Id list: " << std::endl;
		for (auto const &camera : cm->cameras())
			std::cout << "\t" << camera->id() << std::endl;
	}

	// Camera
	if (cm->cameras().empty())
	{
		std::cout << "ERROR: No cameras were identified on the system." << std::endl;
		cm->stop();
		return EXIT_FAILURE;
	}

	std::string cameraId = cm->cameras()[0]->id();
	_camera = cm->get(cameraId);
	_camera->acquire();

	// Stream
	std::unique_ptr<CameraConfiguration> config =
		_camera->generateConfiguration({StreamRole::Viewfinder});

	// Stream configuration
	StreamConfiguration &streamConfig = config->at(0);

	if (_verbose)
	{
		std::cout << "Supported pixel formats: ";
		std::vector<PixelFormat> formats = streamConfig.formats().pixelformats();
		for (const PixelFormat format : formats)
		{
			std::cout << format.toString() << ", ";
		}
		std::cout << std::endl;
	}

	if (_useNv12)
	{
		streamConfig.pixelFormat = libcamera::formats::NV12;
	}
	else
	{
		// In GPU mode, try to get a 4 bytes/pixel format to reduce CPU packing costs.
		// libcamera pixel formats map to DRM fourcc:
		// - XBGR8888: memory is R,G,B,X (no swap needed)
		// - XRGB8888: memory is B,G,R,X (swap R/B)
		_bppBgr = 3;
		_swapRb4 = false;
		if (_useGpu)
		{
			// Empirically on some Pi/Mesa stacks, BGR888 + NEON pack is faster than
			// pulling 4 Bpp from the camera. Allow opting into 4 Bpp via env var.
			const bool prefer4bpp = envEnabled("HYPERCAM_GPU_PREFER_4BPP");
			if (!prefer4bpp)
			{
				streamConfig.pixelFormat = libcamera::formats::BGR888;
				_bppBgr = 3;
				_swapRb4 = false;
			}
			else
			{
				bool hasXbgr = false;
				bool hasXrgb = false;
				std::vector<PixelFormat> fmts = streamConfig.formats().pixelformats();
				for (const PixelFormat &f : fmts)
				{
					if (f == libcamera::formats::XBGR8888)
						hasXbgr = true;
					else if (f == libcamera::formats::XRGB8888)
						hasXrgb = true;
				}

				if (hasXbgr)
				{
					streamConfig.pixelFormat = libcamera::formats::XBGR8888;
					_bppBgr = 4;
					_swapRb4 = false;
				}
				else if (hasXrgb)
				{
					streamConfig.pixelFormat = libcamera::formats::XRGB8888;
					_bppBgr = 4;
					_swapRb4 = true;
				}
				else
				{
					streamConfig.pixelFormat = libcamera::formats::BGR888;
					_bppBgr = 3;
					_swapRb4 = false;
				}
			}
		}
		else
		{
			streamConfig.pixelFormat = libcamera::formats::BGR888;
			_bppBgr = 3;
			_swapRb4 = false;
		}
	}
	streamConfig.colorSpace = libcamera::ColorSpace::fromString("Rec709/Rec709/None/Full");

	libcamera::Size mySize(_width, _height);
	streamConfig.size = mySize;

	auto valRet = config->validate();
	if (valRet == CameraConfiguration::Status::Adjusted)
	{
		std::cout << "WARNING: Configuration Adjusted" << std::endl;
		std::cout << "Color space: " << libcamera::ColorSpace::toString(streamConfig.colorSpace) << std::endl;
		std::cout << "Configuration: " << streamConfig.toString() << std::endl;
	}
	else if (valRet == CameraConfiguration::Status::Invalid)
	{
		std::cout << "ERROR: Configuration invalid" << std::endl;
		return EXIT_FAILURE;
	}

	if (_verbose)
	{
		std::cout << "Validated viewfinder configuration is: "
				  << streamConfig.toString() << std::endl;
	}

	if (_useNv12)
	{
		if (streamConfig.pixelFormat != libcamera::formats::NV12)
		{
			std::cout << "ERROR: NV12 format is not supported" << std::endl;
			return EXIT_FAILURE;
		}
		if ((streamConfig.size.width % 2) != 0 || (streamConfig.size.height % 2) != 0)
		{
			std::cout << "ERROR: NV12 requires even width and height" << std::endl;
			return EXIT_FAILURE;
		}
		_strideY = static_cast<int>(streamConfig.stride);
		_strideUV = static_cast<int>(streamConfig.stride);
		if (_strideY <= 0)
			_strideY = streamConfig.size.width;
		if (_strideUV <= 0)
			_strideUV = streamConfig.size.width;
	}
	else
	{
		if (_useGpu && _bppBgr == 4)
		{
			if (streamConfig.pixelFormat != libcamera::formats::XBGR8888 && streamConfig.pixelFormat != libcamera::formats::XRGB8888)
			{
				std::cout << "ERROR: 4 Bpp X*RGB8888 format is not supported" << std::endl;
				return EXIT_FAILURE;
			}
			// If libcamera adjusted the format away from our preference, update swap flag.
			_swapRb4 = (streamConfig.pixelFormat == libcamera::formats::XRGB8888);
		}
		else if (streamConfig.pixelFormat != libcamera::formats::BGR888)
		{
			std::cout << "ERROR: BGR888 format is not supported" << std::endl;
			return EXIT_FAILURE;
		}
		_strideBgr = static_cast<int>(streamConfig.stride);
		if (_strideBgr <= 0)
			_strideBgr = static_cast<int>(streamConfig.size.width) * _bppBgr;
		if (_verbose)
		{
			std::cout << "Using RGB pixel format: " << streamConfig.pixelFormat.toString()
					  << " (bpp=" << _bppBgr << ", swapRB4=" << (_swapRb4 ? 1 : 0)
					  << ")" << std::endl;
		}
	}

	_imgHeight = streamConfig.size.height;
	_imgWidth = streamConfig.size.width;

	if (!_transformEn)
	{
		_imgHeightToSend = _imgHeight;
		_imgWidthToSend = _imgWidth;
	}

	_camera->configure(config.get());

	// Buffer Allocation
	FrameBufferAllocator *allocator = new FrameBufferAllocator(_camera);

	for (StreamConfiguration &cfg : *config)
	{
		int ret = allocator->allocate(cfg.stream());
		if (ret < 0)
		{
			std::cout << "ERROR: Can't allocate buffers" << std::endl;
			return EXIT_FAILURE;
		}

		size_t allocated = allocator->buffers(cfg.stream()).size();

		if (_verbose)
			std::cout << "Allocated " << allocated << " buffers for stream" << std::endl;
	}

	// Frame Capture
	Stream *stream = streamConfig.stream();
	const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);
	std::vector<std::unique_ptr<Request>> requests;
	for (unsigned int i = 0; i < buffers.size(); ++i)
	{
		std::unique_ptr<Request> request = _camera->createRequest();
		if (!request)
		{
			std::cout << "ERROR: Can't create request" << std::endl;
			return EXIT_FAILURE;
		}

		const std::unique_ptr<FrameBuffer> &buffer = buffers[i];
		int ret = request->addBuffer(stream, buffer.get());
		if (ret < 0)
		{
			std::cout << "ERROR: Can't set buffer for request"
					  << std::endl;
			return EXIT_FAILURE;
		}

		ControlList &controls = request->controls();
		controls.set(controls::AwbEnable, false);
		controls.set(controls::ColourGains, libcamera::Span<const float, 2>({_rGain, _bGain}));

		requests.push_back(std::move(request));
	}

	_camera->requestCompleted.connect(requestComplete);

	// Buffers memory mapping
	for (const std::unique_ptr<libcamera::FrameBuffer> &buffer : buffers)
	{
		for (const FrameBuffer::Plane &plane : buffer->planes())
		{
			size_t mapLen = plane.offset + plane.length;
			void *memory = mmap(NULL, mapLen, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
			uint8_t *planePtr = static_cast<uint8_t *>(memory) + plane.offset;
			_mappedBuffers[buffer.get()].push_back(libcamera::Span<uint8_t>(planePtr, plane.length));
		}
	}

	// controls the setting of camera
	ControlList controls;
	controls.set(controls::AeEnable, false);
	controls.set(controls::ExposureTime, _shutter);
	controls.set(controls::AnalogueGain, _gain);
	controls.set(controls::Brightness, 0.0);
	controls.set(controls::Contrast, 1.0);
	controls.set(controls::AwbEnable, false);
	controls.set(controls::ColourGains, libcamera::Span<const float, 2>({_rGain, _bGain}));
	controls.set(controls::Saturation, 1.0);
	controls.set(controls::Sharpness, 0.0);
	controls.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({INT64_C(100), INT64_C(1000000)}));

	// Start Capture
	_camera->start(&controls);
	for (std::unique_ptr<Request> &request : requests)
		_camera->queueRequest(request.get());

	// Run thread for flatbuffer
	pthread_t ptidFb;
	pthread_create(&ptidFb, NULL, &flatLoop, NULL);

	// Run thread for opencv transformation
	if (_transformEn)
	{
		pthread_t ptidCv;
		pthread_create(&ptidCv, NULL, &transformLoop, NULL);
	}

	// Run an EventLoop for camera
	_loop.exec();

	// Clean Up
	_camera->stop();
	allocator->free(stream);
	delete allocator;
	_camera->release();
	_camera.reset();
	cm->stop();

	return EXIT_SUCCESS;
}
