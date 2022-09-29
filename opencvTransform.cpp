#include "opencvTransform.h"

OpencvTransform::OpencvTransform(int width, int height, Point2f *points, int *widthOut, int *heightOut)
    : _srcQuad(points)
{
    _src = new Mat(height, width, CV_8UC3);

    int x = std::max(_srcQuad[1].x - _srcQuad[0].x, _srcQuad[3].x - _srcQuad[2].x);
    int y = std::max(_srcQuad[2].y - _srcQuad[0].y, _srcQuad[3].y - _srcQuad[1].y);

    _dstQuad[0] = Point2f(0.f, 0.f);
    _dstQuad[1] = Point2f((float)x, 0.f);
    _dstQuad[2] = Point2f(0.f, (float)y);
    _dstQuad[3] = Point2f((float)x, (float)y);

    _dst = new Mat(y, x, CV_8UC3);
    _warp = getPerspectiveTransform(_srcQuad, _dstQuad);

    *widthOut = x;
    *heightOut = y;
}

uint8_t *OpencvTransform::transform(uint8_t *img)
{
    _src->data = img;
    warpPerspective(*_src, *_dst, _warp, _dst->size(), INTER_NEAREST);
    return _dst->data;
}

OpencvTransform::~OpencvTransform()
{
}