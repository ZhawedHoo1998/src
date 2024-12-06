#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <typeinfo>
#include <pthread.h>
#include "opencv2/opencv.hpp"
#include "camera_video.h"
#include "inference.h"
#include "log.h"


#define    TAG                            "test-car-det"


int main() {

    algo_config_t cfg;
	
    cfg.algo_opt_mask = OPT_MASK(ALGO_CAR_DETECT);// | OPT_MASK(ALGO_LANE_SEG);
	cfg.model_path[ALGO_CAR_DETECT] = "";
    cfg.core_mask[ALGO_CAR_DETECT] = NN_CORE_0; 
 {
	algo_config_t cfg;
	
    cfg.algo_opt_mask = OPT_MASK(ALGO_CAR_DETECT);// | OPT_MASK(ALGO_LANE_SEG);
	cfg.model_path[ALGO_CAR_DETECT] = "";
    cfg.core_mask[ALGO_CAR_DETECT] = NN_CORE_0; 
 }

    ALGO_ERR_CODE err;
    void* car_handle = algo_nn_init(&cfg, &err);
	if(!car_handle) {
        exit(-1);
    }

    cv::Mat img = cv::imread("../data/car.jpg");
	if(img.empty()) {
		algo_nn_deinit(car_handle, &err);
		exit(-1);
	}

	LOGD(TAG, "input image:%dx%d", img.cols, img.rows);
	//LOGI(TAG, "read img:%dx%d", img.cols, img.rows);
	std::vector<object_box_t> boxes;
	
	int count = 50;
	unsigned long long start = log_get_timestamp();
	for(int i = 0; i < count; i++) {
		cv::Mat rgb_img;
		cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
		algo_nn_detect_car(car_handle, rgb_img, 0.1, boxes, &err);
		printf("boxes",boxes);
	}
	unsigned long long end = log_get_timestamp();
	LOGI(TAG, "thread_index[%d] %dx%d  detect object size:%d time:%lluus",  0, 1280, 720, boxes.size(), (end - start) / count);
	for(auto& box : boxes) {
		LOGD(TAG, "\t[%.3f,%.3f,%.3f,%.3f]", box.x1, box.y1, box.x2, box.y2);
		cv::rectangle(img, cv::Point(box.x1 * img.cols, box.y1 * img.rows), cv::Point(box.x2 * img.cols, box.y2 * img.rows),cv::Scalar(255, 0, 0), 2);
	}

	cv::imwrite("cat_det_res.jpg", img);

    algo_nn_deinit(car_handle, &err);
    return 0;
}