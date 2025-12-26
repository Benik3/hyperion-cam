#include "opencvTransform.h"

OpencvTransform::OpencvTransform(int width, int height, cv::Point2f *points, int *widthOut, int *heightOut)
    : _srcQuad(points), _srcWidth(width), _srcHeight(height)
{
    int x = std::max(_srcQuad[1].x - _srcQuad[0].x, _srcQuad[3].x - _srcQuad[2].x);
    int y = std::max(_srcQuad[2].y - _srcQuad[0].y, _srcQuad[3].y - _srcQuad[1].y);

    _dstQuad[0] = cv::Point2f(0.f, 0.f);
    _dstQuad[1] = cv::Point2f((float)x, 0.f);
    _dstQuad[2] = cv::Point2f(0.f, (float)y);
    _dstQuad[3] = cv::Point2f((float)x, (float)y);

    _warp = cv::getPerspectiveTransform(_srcQuad, _dstQuad);

    // Small ring buffer to reduce transform->send concurrent overwrite on the same dst.
    // This is not a per-frame copy; it just reserves a few reusable buffers.
    constexpr std::size_t kRingSize = 3;
    _dstBuffers.reserve(kRingSize);
    for (std::size_t i = 0; i < kRingSize; i++)
        _dstBuffers.emplace_back(y, x, CV_8UC3);

    *widthOut = x;
    *heightOut = y;
}

uint8_t *OpencvTransform::transform(uint8_t *img)
{
    std::size_t idx = _dstIndex++ % _dstBuffers.size();
    cv::Mat src(_srcHeight, _srcWidth, CV_8UC3, img);
    cv::warpPerspective(src, _dstBuffers[idx], _warp, _dstBuffers[idx].size(), cv::INTER_NEAREST);
    return _dstBuffers[idx].data;
}

OpencvTransform::~OpencvTransform()
{
}