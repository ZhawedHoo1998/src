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
#include "lanex.h"


#define    TAG                            "test-car-lane-seg"

class ObjectMeasure {
    public:
        ObjectMeasure(float k = 28 * 33 * 1.4) : k_(k) {}
        void set(float d, float w) {
            // d/L=f/w
            // f = d * w / L_;
            // k_ = L_ * f;
            k_ = d * w;
        }
        float calc(float w) {
            return k_ / (w + 1e-6);
        }
    private:
        float k_;
        float L_ = 1.8;      
};

int main(int argc, const char* argv[]) {

    algo_config_t cfg;
	
    cfg.algo_opt_mask = OPT_MASK(ALGO_LANE_SEG)| OPT_MASK(ALGO_CAR_DETECT);
	cfg.model_path[ALGO_LANE_SEG] = "";
    cfg.core_mask[ALGO_LANE_SEG] = NN_CORE_0; 
    cfg.model_path[ALGO_CAR_DETECT] = "";
    cfg.core_mask[ALGO_CAR_DETECT] = NN_CORE_0; 

    ALGO_ERR_CODE err;
    void* handle = algo_nn_init(&cfg, &err);
    if(!handle) {
        exit(-1);
    }

    cv::Mat img = cv::imread("../data/0012.jpg");
	if(img.empty()) {
		algo_nn_deinit(handle, &err);
		exit(-1);
	}

    ObjectMeasure obj_meas;

	LOGD(TAG, "input image:%dx%d", img.cols, img.rows);
    int org_width = img.cols;
    int org_height = img.rows;
	
	cv::Mat rgb_img;
    std::vector<object_box_t> boxes;
	cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    algo_nn_detect_car(handle, rgb_img, 0.1, boxes, &err);
    LOGD(TAG, "algo_nn_detect_car number:%d", boxes.size());
    cv::Mat seg_mask;
	algo_nn_lane_seg(handle, rgb_img, 0.5, seg_mask, &err);

    LaneX lane_x(org_width, org_height, 0.3, -0.3, 8.0);
    lane_x.set_mask(seg_mask);

    for(auto& box : boxes) {
		LOGD(TAG, "[%.3f,%.3f,%.3f,%.3f]", box.x1, box.y1, box.x2, box.y2);   
        int x1 = std::round(box.x1 * org_width);
        int y1 = std::round(box.y1 * org_height);
        int x2 = std::round(box.x2 * org_width);
        int y2 = std::round(box.y2 * org_height);
        int cx = (x1 + x2) / 2;
        int cy =  y2;  
        bool flags = false;
        bool coord_x_flags = (cx > org_width / 3);
        cv::Scalar car_color(255, 0, 0);
        if(lane_x.is_inside(cx, cy, org_width, org_height)) {
             car_color = cv::Scalar(0, 0, 255);
        }
		cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), car_color, 2);
        if(coord_x_flags) {
            char text[32];
            sprintf(text, "%.2fm", obj_meas.calc(x2 - x1 + 1));
            putText(img, text, cv::Point(x1 , y1 -10), 0, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA); 
        }
	}

    //lane draw
    cv::Mat seg_mask2;
    cv::resize(seg_mask, seg_mask2, cv::Size(org_width, org_height), 0, 0, cv::INTER_NEAREST);
    for(int y = 0; y < org_height; y++) {
        for(int x = 0; x < org_width; x++) {
            if(seg_mask2.at<uint8_t>(y, x) == 255) {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(0,255, 0);  
            }
        }
    }

    cv::imwrite("cat_det_res_0012.jpg", img);

    algo_nn_deinit(handle, &err);
    return 0;
}