#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <typeinfo>
#include <pthread.h>
#include "opencv2/opencv.hpp"
#include "camera_video.h"
#include "inference.h"
#include "lanex.h"
#include "log.h"

#define    TAG                            "test-lane-seg"

void seg_result_cb(void* arg, cv::Mat& seg_mask, int org_width, int org_height)  {
    LOGD(TAG, "org input image:%dx%d", org_width, org_height);
	LOGD(TAG, "mask %dx%dx%d", seg_mask.cols, seg_mask.rows, seg_mask.channels());

    LaneX lane_x(org_width, org_height);
    lane_x.set_mask(seg_mask);
}

int main() {

    algo_config_t cfg;
	
    cfg.algo_opt_mask = OPT_MASK(ALGO_LANE_SEG)| OPT_MASK(ALGO_CAR_DETECT);
	cfg.model_path[ALGO_LANE_SEG] = "";
    cfg.core_mask[ALGO_LANE_SEG] = NN_CORE_0; 
    cfg.model_path[ALGO_CAR_DETECT] = "";
    cfg.core_mask[ALGO_CAR_DETECT] = NN_CORE_0; 

    ALGO_ERR_CODE err;
    void* handle = algo_nn_init(&cfg, &err);

    cv::Mat img = cv::imread("../data/0012.jpg");
	if(img.empty()) {
		algo_nn_deinit(handle, &err);
		exit(-1);
	}

	LOGD(TAG, "input image:%dx%d", img.cols, img.rows);
	
	cv::Mat rgb_img;
    std::vector<object_box_t> boxes;
	cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    algo_nn_detect_car(handle, rgb_img, 0.1, boxes, &err);
    LOGD(TAG, "algo_nn_detect_car number:%d", boxes.size());;
	algo_nn_lane_seg_x(handle, rgb_img, 0.5,seg_result_cb, nullptr, &err);

    for(auto& box : boxes) {
		LOGD(TAG, "\t[%.3f,%.3f,%.3f,%.3f]", box.x1, box.y1, box.x2, box.y2);
		cv::rectangle(img, cv::Point(box.x1 * img.cols, box.y1 * img.rows), cv::Point(box.x2 * img.cols, box.y2 * img.rows),cv::Scalar(255, 0, 0), 2);
	}

    cv::imwrite("cat_det_res_0012.jpg", img);

    algo_nn_deinit(handle, &err);
    return 0;
}