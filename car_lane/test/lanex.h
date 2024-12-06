#ifndef   __LANEX_H
#define   __LANEX_H

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <vector>
#include <iostream>
#include <typeinfo>

#include "opencv2/opencv.hpp"


class LaneX {
    public:
        LaneX(int w, int h, float k1, float k2, float wh_r);
        ~LaneX();

        int is_inside(int cx, int cy, int org_w, int org_h);
        void set_mask(cv::Mat& seg_mask);
        void get_lanes(std::vector<cv::Vec6f>& lines);

    private:
        int w_;
        int h_;
        float k1_;
        float k2_;
        float wh_r_;
        std::vector<cv::Vec6f> lines_;
};

#endif