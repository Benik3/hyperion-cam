#pragma once

#include <cstdint>

#include <opencv2/core/types.hpp>

class GpuRgbTransform
{
public:
	enum class InputFormat : int
	{
		// 3 bytes per pixel. Bytes are copied into (R,G,B) channels as provided.
		Packed3 = 3,
		// 4 bytes per pixel in memory order R,G,B,X (DRM fourcc XBGR8888).
		Xbgr8888 = 4,
		// 4 bytes per pixel in memory order B,G,R,X (DRM fourcc XRGB8888).
		Xrgb8888 = 5,
	};

	GpuRgbTransform(int srcWidth,
			 int srcHeight,
			 const cv::Point2f *srcQuad,
			 int dstWidth,
			 int dstHeight);
	~GpuRgbTransform();

	GpuRgbTransform(const GpuRgbTransform &) = delete;
	GpuRgbTransform &operator=(const GpuRgbTransform &) = delete;

	// Treats input as tightly packed 3 bytes per pixel (BGR888/RGB888 in memory).
	// strideBytes is the number of bytes per input row.
	// outRgb must be dstWidth*dstHeight*3, tightly packed.
	void transform(const uint8_t *inRgb, int strideBytes, uint8_t *outRgb);

	// Like transform(), but allows 4 bytes per pixel input (e.g. XBGR8888/BGRX8888).
	void transform(const uint8_t *inPixels, int strideBytes, InputFormat fmt, uint8_t *outRgb);

private:
	struct Impl;
	Impl *_impl;
};
