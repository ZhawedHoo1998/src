#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#ifndef OBJECT_BOX_H
#define OBJECT_BOX_H

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
#endif  // OBJECT_BOX_H