#include <iostream>
#include <pthread.h>
#include <sys/mman.h>
#include <atomic>
#include <boost/program_options.hpp>
#include <libcamera/libcamera.h>
#include <opencv2/core/types.hpp>

#include "event_loop.h"
#include "FlatBufferConnection.h"
#include "opencvTransform.h"

using namespace libcamera;
namespace po = boost::program_options;

static std::shared_ptr<Camera> _camera;
std::map<libcamera::FrameBuffer *, std::vector<libcamera::Span<uint8_t>>> _mappedBuffers;
static EventLoop _loop;
static EventLoop _loopFb;
static EventLoop _loopTr;
FlatBufferConnection _flatbuf;
OpencvTransform *_transformation;
cv::Point2f _imgQuad[4];

void *_imgMem = nullptr;
uint8_t *_imgMemToSend = nullptr;
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

void *flatLoop(void *arg)
{
	_loopFb.exec();
	return 0;
}

void flatBufSend()
{
	if (_flatbuf.connectToServer(_origin.c_str(), _conn.c_str(), _port, _priority, !_verbose))
	{
		if (_flatbuf.setImage(_imgMemToSend, _imgWidthToSend, _imgHeightToSend) < 0)
		{
			std::cout << "ERROR: Flatbuffer write timeout" << std::endl;
		}
	}
	else
		std::cout << "ERROR: Flatbuffer connection timeout" << std::endl;
}

void *transformLoop(void *arg)
{
	_transformation = new OpencvTransform(_imgWidth, _imgHeight, _imgQuad, &_imgWidthToSend, &_imgHeightToSend);
	_loopTr.exec();
	return 0;
}

void transformImg()
{
	_imgMemToSend = _transformation->transform((uint8_t *)_imgMem);
	_loopFb.callLater(std::bind(&flatBufSend));
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

		if (_transformEn)
		{
			_imgMem = _mappedBuffers.find(buffer)->second[0].data();
			_loopFb.callLater(std::bind(&transformImg));
		}
		else
		{
			_imgMemToSend = _mappedBuffers.find(buffer)->second[0].data();
			_loopFb.callLater(std::bind(&flatBufSend));
		}
	}

	/* Re-queue the Request to the camera. */
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

	streamConfig.pixelFormat = libcamera::formats::BGR888;
	streamConfig.colorSpace = libcamera::ColorSpace::Sycc;

	libcamera::Size mySize(_width, _height);
	streamConfig.size = mySize;

	auto valRet = config->validate();
	if (valRet == CameraConfiguration::Status::Adjusted)
	{
		std::cout << "WARNING: Configuration Adjusted" << std::endl;
	}
	else if (valRet == CameraConfiguration::Status::Invalid)
	{
		std::cout << "ERROR: Configuration invalid" << std::endl;
		return EXIT_FAILURE;
	}

	if (streamConfig.colorSpace == libcamera::ColorSpace::Raw)
		std::cout << "Color space: Raw" << std::endl;
	else if (streamConfig.colorSpace == libcamera::ColorSpace::Srgb)
		std::cout << "Color space: Srgb" << std::endl;
	else if (streamConfig.colorSpace == libcamera::ColorSpace::Sycc)
		std::cout << "Color space: Sycc" << std::endl;
	else if (streamConfig.colorSpace == libcamera::ColorSpace::Smpte170m)
		std::cout << "Color space: Smpte170m" << std::endl;
	else if (streamConfig.colorSpace == libcamera::ColorSpace::Rec709)
		std::cout << "Color space: Rec709" << std::endl;
	else if (streamConfig.colorSpace == libcamera::ColorSpace::Rec2020)
		std::cout << "Color space: Rec2020" << std::endl;
	else
		std::cout << "Color space: Unknown" << std::endl;

	if (_verbose)
	{
		std::cout << "Validated viewfinder configuration is: "
				  << streamConfig.toString() << std::endl;
	}

	if (streamConfig.pixelFormat != libcamera::formats::BGR888)
	{
		std::cout << "ERROR: BGR888 format is not supported" << std::endl;
		return EXIT_FAILURE;
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
		const FrameBuffer::Plane &plane = buffer->planes().front();
		void *memory = mmap(NULL, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
		_mappedBuffers[buffer.get()].push_back(libcamera::Span<uint8_t>(static_cast<uint8_t *>(memory), plane.length));
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
