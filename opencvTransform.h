
#include <cstddef>
#include <cstdint>
#include <vector>

#include "opencv2/imgproc.hpp"

class OpencvTransform
{
public:
	/// @brief Constructor
	OpencvTransform(int width, int height, cv::Point2f *points, int *widthOut, int *heightOut, bool alignEvenOutput = false);

	/// @brief Destructor
	~OpencvTransform();

	uint8_t *transform(uint8_t *img);
	uint8_t *transform(uint8_t *img, int srcStrideBytes);
	void transformInto(const uint8_t *img, int srcStrideBytes, uint8_t *dst, int dstStrideBytes);

private:
	cv::Point2f *_srcQuad;
	cv::Point2f _dstQuad[4];
	cv::Mat _warp;
	cv::Mat _mapX;
	cv::Mat _mapY;
	std::vector<cv::Mat> _dstBuffers;
	std::size_t _dstIndex = 0;
	int _srcWidth = 0;
	int _srcHeight = 0;
};