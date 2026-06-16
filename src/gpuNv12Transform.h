#pragma once

#include <cstdint>

#include <opencv2/core/types.hpp>

class GpuNv12Transform
{
public:
	// shadeGains (optional): per-output-position RGB gain map, shadeU*shadeV*3
	// floats in [0,1], row-major (cell (i,j) at (j*shadeU+i)*3). Applied in the
	// warp shader (--shading-map / ALSC correction). nullptr = disabled.
	GpuNv12Transform(int srcWidth,
					 int srcHeight,
					 const cv::Point2f *srcQuad,
					 int dstWidth,
					 int dstHeight,
					 bool highQuality = false,
					 const float *shadeGains = nullptr,
					 int shadeU = 0,
					 int shadeV = 0);
	~GpuNv12Transform();

	GpuNv12Transform(const GpuNv12Transform &) = delete;
	GpuNv12Transform &operator=(const GpuNv12Transform &) = delete;

	// Transforms NV12 planes into tightly packed output planes (stride == width).
	// yPlane: pointer to Y (strideY bytes per row)
	// uvPlane: pointer to UV (strideUV bytes per row), size height/2 x width/2 with 2 channels interleaved
	// outY: must be dstWidth*dstHeight
	// outUV: must be dstWidth*dstHeight/2
	void transform(const uint8_t *yPlane,
				   const uint8_t *uvPlane,
				   int strideY,
				   int strideUV,
				   uint8_t *outY,
				   uint8_t *outUV);

	// DMA-BUF import path: avoids CPU upload by importing the input planes via EGLImage.
	// Returns false if DMA-BUF import isn't available/enabled (caller may fall back to transform()).
	bool transformDmabuf(int fdY,
						 int offsetYBytes,
						 int strideY,
						 int fdUV,
						 int offsetUVBytes,
						 int strideUV,
						 uint8_t *outY,
						 uint8_t *outUV);

private:
	struct Impl;
	Impl *_impl;
};
