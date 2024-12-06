#include <vector>
#include  "log.h"
#include "lanex.h"


#define    TAG                            "lanex"

template <typename T>
T q_max(T x,T y,T z,T h){
    x = x > y ? x : y;
    x = x > z ? x : z;
    x=  x > h ? x : h;
    return x;
}

template <typename T>
T q_min(T x,T y,T z,T h){
    x = x < y ? x : y;
    x = x < z ? x : z;
    x=  x < h ? x : h;
    return x;
}

void fitLineRansac(const std::vector<cv::Point2f>& points,
                   cv::Vec4f &line,
                   int iterations = 1000,
                   double sigma = 1.,
                   double k_min = -7.,
                   double k_max = 7.){
    unsigned int n = points.size();

    if(n<2)
    {
        return;
    }

    cv::RNG rng;
    double bestScore = -1.;
    for(int k=0; k<iterations; k++)
    {
        int i1=0, i2=0;
        while(i1==i2)
        {
            i1 = rng(n);
            i2 = rng(n);
        }
        const cv::Point2f& p1 = points[i1];
        const cv::Point2f& p2 = points[i2];

        cv::Point2f dp = p2-p1;//直线的方向向量
        dp *= 1./norm(dp);
        double score = 0;

        if(dp.y/dp.x<=k_max && dp.y/dp.x>=k_min )
        {
            for(int i=0; i<n; i++)
            {
                cv::Point2f v = points[i]-p1;
                double d = v.y*dp.x - v.x*dp.y;//向量a与b叉乘/向量b的摸.||b||=1./norm(dp)
                //score += exp(-0.5*d*d/(sigma*sigma));//误差定义方式的一种
                if( fabs(d)<sigma )
                    score += 1;
            }
        }
        if(score > bestScore)
        {
            line = cv::Vec4f(dp.x, dp.y, p1.x, p1.y);
            bestScore = score;
        }
    }
}



LaneX::LaneX(int w, int h, float k1 , float k2 , float wh_r ) : w_(w), h_(h), k1_(k1),k2_(k2), wh_r_(wh_r) {}
LaneX::~LaneX() {}

int LaneX::is_inside(int cx, int cy, int org_w, int org_h) {
    float s = 1.0;
    std::vector<float> k_1;
    std::vector<float> b_1;
    std::vector<float> k_2;
    std::vector<float> b_2;
    if(lines_.size() == 0) return 2; // 若没有车道，则所有车在当前车道
    for(auto& line : lines_) {
        //x1 y1 x2 y2 k b
        float x1 = line[0] * org_w;
        float y1 = line[1] * org_h;
        float x2 = line[2] * org_w;
        float y2 = line[3] * org_h;
        float k = (y2 - y1) / (x2 - x1 + 1e-6);
        double b = y1 - k * x1;
            if(k > 0) {
                k_1.push_back(k);
                b_1.push_back(b);
            } else {
                k_2.push_back(k);
                b_2.push_back(b);
            }
    }
    if(k_1.size() > 0 && k_2.size() > 0) {
        float x = cx, y = cy;
        int max_k_i = std::max_element(k_1.begin(), k_1.end()) -k_1.begin();
        int min_k_i = std::min_element(k_2.begin(), k_2.end()) - k_2.begin();
        s *= y - k_1[max_k_i] * x- b_1[max_k_i];
        s *= y - k_2[min_k_i] * x- b_2[min_k_i];
        if(s > 0){
            return 2;
            }
        if(y > 1.0* (k_1[max_k_i] * x + b_1[max_k_i]) && y > 1.0* (k_2[min_k_i] * x + b_2[min_k_i])) {
            return 1;
            }
        else{
            return 3;
        }
    } else {
        return 1;
    }
}

void LaneX::set_mask(cv::Mat& seg_mask) {
    unsigned long long t0 = log_get_timestamp();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(seg_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    unsigned long long t1 = log_get_timestamp();
    contours.erase(
    std::remove_if(contours.begin(), contours.end(), [&](const std::vector<cv::Point>& c)
    {  
    cv::RotatedRect rect =cv::minAreaRect(c);
    float r0 = rect.size.width / (rect.size.height + 1e-6);
    float r1 = rect.size.height / (rect.size.width + 1e-6);
    return  (r0 < wh_r_ && r1 < wh_r_) || cv::contourArea(c) < 25;
    })
    , contours.end());
    unsigned long long t2 = log_get_timestamp();

    //std::vector<cv::Vec6f> lines;
    lines_.clear();
    int img_h = seg_mask.rows;
    int img_w = seg_mask.cols;
    for(auto& c : contours) {
        cv::Mat tmp(img_h, img_w, CV_8UC1, cv::Scalar(0));
        cv::fillPoly(tmp, c, cv::Scalar(255), 8, 0);
        std::vector<cv::Point2f> pts;
    
        for(int y = 0; y < img_h; y++) {
            for(int x = 0; x < img_w; x++) {
                if(tmp.at<uchar>(y, x) == 255) {
                    pts.push_back(cv::Point2f(x, y));
                }
            }
        }

        cv::Vec4f lineParam;
        fitLineRansac(pts,lineParam, 60, 10);
        float k = lineParam[1] / (lineParam[0] + 1e-6);
        float b = lineParam[3] - k * lineParam[2];

        if(k >= k1_ || k <= k2_) {
            std::vector<float> xa(pts.size());
            std::vector<float> ya(pts.size());

            for(int i = 0; i < pts.size(); i++) {
                xa[i] = pts[i].x;
                ya[i] = pts[i].y;
            }

            auto max_x = *std::max_element(xa.begin(), xa.end());
            auto max_y = *std::max_element(ya.begin(), ya.end());
            auto min_x = *std::min_element(xa.begin(), xa.end());
            auto min_y = *std::min_element(ya.begin(), ya.end());

            cv::Point2f p1,p2;
            p1.y = min_y;//min_y; //0
            p1.x = ( p1.y - b) / k;

            p2.y = max_y;//img_h;//max_y;//359;
            p2.x = (p2.y-b) / k;
            lines_.push_back(cv::Vec6f(p1.x / img_w, p1.y / img_h , p2.x / img_w, p2.y / img_h , k, b));                                 
        }
    }
    unsigned long long t3 = log_get_timestamp();
    // LOGD(TAG, "t1-t0=%lluus", t1 - t0);
    // LOGD(TAG, "t2-t1=%lluus", t2 - t1);
    // LOGD(TAG, "t3-t2=%lluus", t3 - t2);

    // cv::Mat mask_rgb;
    // std::vector<cv::Mat> bgr = {seg_mask, seg_mask, seg_mask};
    // cv::merge(bgr, mask_rgb);
    // for(auto& li : lines_) {
    //     int x1 = static_cast<int>(li[0] * img_w);
    //     int y1 = static_cast<int>(li[1] * img_h);
    //     int x2 = static_cast<int>(li[2] * img_w);
    //     int y2 = static_cast<int>(li[3] * img_h);
    //     cv::line(mask_rgb, cv::Point(x1,  y1),cv::Point(x2, y2), cv::Scalar(0,255,0),2);
    //     printf("k=%.3f b=%.3f\r\n", li[4], li[5]);  
    // }
    // cv::imwrite("mask_rgb.jpg", mask_rgb);
    // cv::imwrite("lane_seg_res.jpg", seg_mask);
}


void LaneX::get_lanes(std::vector<cv::Vec6f>& lanes) {
    lanes = lines_;
}