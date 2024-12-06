#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include "opencv2/opencv.hpp"


void laneseg_to_contours(cv::Mat& lane_mask, std::vector<std::vector<cv::Point>>& contours) {
    cv::Mat bin_gray;
    if(lane_mask.channels() == 3){
        cv::cvtColor(lane_mask, bin_gray, cv::COLOR_BGR2GRAY);
    } 
    else if(lane_mask.channels() == 1) {
        bin_gray = lane_mask.clone();
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat morph_img;
    cv::erode(bin_gray, morph_img, kernel);
    kernel= cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    cv::Mat dilate_img;
    morphologyEx(morph_img, dilate_img, cv::MORPH_RECT, kernel);

    cv::findContours(morph_img, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    contours.erase(
        std::remove_if(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c)
        {  
            cv::RotatedRect rect =cv::minAreaRect(c);
            float r0 = rect.size.width / (rect.size.height + 1e-6);
            float r1 = rect.size.height / (rect.size.width + 1e-6);
            return  (r0 < 8 && r1 < 8) || cv::contourArea(c) < 30;

        }) , contours.end());
}

int main(int argc, const char* argv[]) {

    cv::Mat lane_mask = cv::imread("../res_lane_seg_1.jpg");
    cv::Mat org_img = cv::imread("../0001.jpg");

    srand(static_cast<unsigned int>(time(0)));

    cv::Mat bg(lane_mask.rows, lane_mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<std::vector<cv::Point>> contours;
    laneseg_to_contours(lane_mask, contours);

    printf("lane size=%d\r\n", contours.size());

    const cv::Scalar color_lists[] = {
	cv::Scalar(255, 0, 0),
	cv::Scalar(255, 255, 0),
	cv::Scalar(0, 255, 0),
	cv::Scalar(0, 255, 255),
	cv::Scalar(222, 128, 255),
	cv::Scalar(125, 255, 0),
	cv::Scalar(255, 0, 255)};
	
    for(auto& c : contours) { 
        cv::fillPoly(org_img, c, color_lists[rand() % 5], 8, 0); 
    }
    cv::imshow("org_img", org_img);
    cv::waitKey(0);

    return 0;
}