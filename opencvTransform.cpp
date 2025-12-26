#include "opencvTransform.h"

OpencvTransform::OpencvTransform(int width, int height, cv::Point2f *points, int *widthOut, int *heightOut, bool alignEvenOutput)
    : _srcQuad(points), _srcWidth(width), _srcHeight(height)
{
    int x = std::max(_srcQuad[1].x - _srcQuad[0].x, _srcQuad[3].x - _srcQuad[2].x);
    int y = std::max(_srcQuad[2].y - _srcQuad[0].y, _srcQuad[3].y - _srcQuad[1].y);

    if (alignEvenOutput)
    {
        x &= ~1;
        y &= ~1;
        if (x <= 0)
            x = 2;
        if (y <= 0)
            y = 2;
    }

    _dstQuad[0] = cv::Point2f(0.f, 0.f);
    _dstQuad[1] = cv::Point2f((float)x, 0.f);
    _dstQuad[2] = cv::Point2f(0.f, (float)y);
    _dstQuad[3] = cv::Point2f((float)x, (float)y);

    _warp = cv::getPerspectiveTransform(_srcQuad, _dstQuad);

    // Precompute remap tables for fast per-frame remap (inverse mapping).
    // This avoids recomputing the perspective mapping each frame.
    cv::Mat warpInv;
    cv::invert(_warp, warpInv);
    _mapX.create(y, x, CV_32FC1);
    _mapY.create(y, x, CV_32FC1);

    const double h00 = warpInv.at<double>(0, 0);
    const double h01 = warpInv.at<double>(0, 1);
    const double h02 = warpInv.at<double>(0, 2);
    const double h10 = warpInv.at<double>(1, 0);
    const double h11 = warpInv.at<double>(1, 1);
    const double h12 = warpInv.at<double>(1, 2);
    const double h20 = warpInv.at<double>(2, 0);
    const double h21 = warpInv.at<double>(2, 1);
    const double h22 = warpInv.at<double>(2, 2);

    for (int dy = 0; dy < y; dy++)
    {
        float *mx = _mapX.ptr<float>(dy);
        float *my = _mapY.ptr<float>(dy);
        for (int dx = 0; dx < x; dx++)
        {
            const double denom = h20 * dx + h21 * dy + h22;
            if (denom == 0.0)
            {
                mx[dx] = -1.f;
                my[dx] = -1.f;
                continue;
            }
            mx[dx] = static_cast<float>((h00 * dx + h01 * dy + h02) / denom);
            my[dx] = static_cast<float>((h10 * dx + h11 * dy + h12) / denom);
        }
    }

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
    cv::remap(src, _dstBuffers[idx], _mapX, _mapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
    return _dstBuffers[idx].data;
}

OpencvTransform::~OpencvTransform()
{
}