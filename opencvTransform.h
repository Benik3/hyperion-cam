#include "opencv2/imgproc.hpp"

using namespace cv;

class OpencvTransform
{
public:
	/// @brief Constructor
	OpencvTransform(int width, int height, Point2f *points, int *widthOut, int *heightOut);

	/// @brief Destructor
	~OpencvTransform();

	uint8_t *transform(uint8_t *img);

private:
	Point2f *_srcQuad;

	Point2f _dstQuad[4];
	Mat *_src;
	Mat *_dst;
	Mat _warp;
};