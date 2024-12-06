// TODO: 话题名称  发布的消息
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
#include <unordered_map>

#include <thread>
#include <chrono>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <compressed_image_transport/compressed_subscriber.h>
// ours msg
#include "fusion_interfaces/msg/boundary_box.hpp"
#include "fusion_interfaces/msg/boundary_box_array.hpp"
// SORT 
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "BYTETracker.h"

#define    TAG                            "test-car-det"

using namespace std;
using namespace cv;

using std::placeholders::_1;

// struct Detection_Velo {
//     int time_sec;
//     int count;
//     std::unordered_map<int, double, int> id_dist_map;
// };

struct DistanceCount {
    double dist;
    int count;
    int time_sec_record;
    float last_velo;
};

struct Detection_Velo {
    int time_sec;
    std::unordered_map<int, DistanceCount> id_dist_map;
};

// for car dist judge
class ObjectMeasure_6 {
    public:
        ObjectMeasure_6(float fx = 847.144214, float L = 1.8, float cx = 340.188407) : fx_(fx), L_(L), cx_(cx) {}
        //ObjectMeasure_6(float fx = 864.978802, float L = 1.8, float cx = 337.289950) : fx_(fx), L_(L), cx_(cx) {}
        void set(float fx, float L, float cx) {
            this->fx_ = fx;
            this->L_ = L;
            this->cx_ = cx;
        }

        std::pair<float, float> calc(int w, int u) {
            if (w <= 0) {
                return {INFINITY, INFINITY};
            }
            float dist = fx_ * L_ / w;
            float hori =(((u - cx_) * L_ ) / w);
            return {dist, hori};
        }
    
    private:
     
        float fx_;
        float L_;
        float cx_;
};

class ObjectMeasure_25 {
    public:
        ObjectMeasure_25(float fx = 2545.758161, float L = 1.8, float cx = 350.188407) : fx_(fx), L_(L), cx_(cx) {}
        // ObjectMeasure_25(float fx = 2625.084229, float L = 1.8, float cx = 325.855351) : fx_(fx), L_(L), cx_(cx) {}
        void set(float fx, float L, float cx) {
            this->fx_ = fx;
            this->L_ = L;
            this->cx_ = cx;
        }
        std::pair<float, float> calc(int w, int u) {
            if (w <= 0) {
                return {INFINITY, INFINITY};
            }
            float dist = fx_ * L_ / w;
            float hori = (((u - cx_) * L_ ) / w);
            return {dist, hori};
        }

    private:
     
        float fx_;
        float L_;
        float cx_;
};

class CAR_DETECT_NODE : public rclcpp::Node 
{
    public:
        CAR_DETECT_NODE () : Node("car_detect_node"), lane_x(org_width, org_height, 0.3, -0.3, 8.0)
        {
            six_box_array_message = fusion_interfaces::msg::BoundaryBoxArray();

            twenfive_box_array_message = fusion_interfaces::msg::BoundaryBoxArray();
            /* define publisher*/
            auto sensor_qos = rclcpp::QoS(rclcpp::SensorDataQoS());
            // six_img_publisher_zero_it_ = image_transport::create_publication(this, "/camera/six/image_raw");
            // twenfive_img_publisher_zero_it_ = it.advertise("/camera/twenfive/image_raw");
            six_detimg_publisher_ = image_transport::create_publisher(this, "/camera/six/det_image_raw", 
                                    rclcpp::QoS(1).reliability((rmw_qos_reliability_policy_t)1).get_rmw_qos_profile());

            twenfive_detimg_publisher_ = image_transport::create_publisher(this, "/camera/twenfive/det_image_raw", 
                                    rclcpp::QoS(1).reliability((rmw_qos_reliability_policy_t)1).get_rmw_qos_profile());

            // six_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/camera/six/image_rawcompressed", 10, 
            //                         std::bind(&CAR_DETECT_NODE::six_imageCallback, this, std::placeholders::_1));
            six_img_sub_ = image_transport::create_subscription(this, "/camera/six/image_raw", 
                                            std::bind(&CAR_DETECT_NODE::six_imageCallback, this, std::placeholders::_1), "raw", sensor_qos.get_rmw_qos_profile());

            twenfive_img_sub_ = image_transport::create_subscription(this, "/camera/twenfive/image_raw", 
                                            std::bind(&CAR_DETECT_NODE::twenfive_imageCallback, this, std::placeholders::_1), "raw", sensor_qos.get_rmw_qos_profile());

            six_bbox_pub_ = this->create_publisher<fusion_interfaces::msg::BoundaryBoxArray>("/car_detect/six_bbox", 10);
            twenfive_bbox_pub_ = this->create_publisher<fusion_interfaces::msg::BoundaryBoxArray>("/car_detect/twenfive_bbox", 10);

            // 实例化机构体 获取当前时间   每次收到消息后，判断和velo消息是不是差一秒，如果差一秒，就计算当前id和消息id相同的速度，如果没有差一秒，则保持速度
            // 图像检测一个数据没有，毫米波雷达有数据，不要发布
            
            // 获取当前时间，
            six_box_velo_msgs.time_sec = this->now().seconds();

            algo_config_t cfg_6 {0};
            cfg_6.algo_opt_mask = OPT_MASK(ALGO_CAR_DETECT);// | OPT_MASK(ALGO_LANE_SEG);
            cfg_6.model_path[ALGO_CAR_DETECT] = "";
            cfg_6.core_mask[ALGO_CAR_DETECT] = NN_CORE_0; 

            algo_config_t cfg_lane {0};
            cfg_lane.algo_opt_mask = OPT_MASK(ALGO_LANE_SEG);
            cfg_lane.model_path[ALGO_LANE_SEG] = "";
            cfg_lane.core_mask[ALGO_LANE_SEG] = NN_CORE_1; 

            algo_config_t cfg_25 {0};
            cfg_25.algo_opt_mask = OPT_MASK(ALGO_CAR_DETECT);
            cfg_25.model_path[ALGO_CAR_DETECT] = "";
            cfg_25.core_mask[ALGO_CAR_DETECT] = NN_CORE_2; 

            car_handle_6 = algo_nn_init(&cfg_6, &err);
            car_handle_25 = algo_nn_init(&cfg_25, &err);
            lane_handle = algo_nn_init(&cfg_lane, &err);
            
            // thread for lane detect
            startPeriodicTaskThread();

            if(!car_handle_6 || !car_handle_25 || !lane_handle) {
                exit(-1);
            }

        }

        ~CAR_DETECT_NODE()
        {
            //algo_nn_deinit(car_handle, &err);
            stopPeriodicTaskThread();
        }


        void six_imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
        {
            //printf("received_6\n");
            
            // 创建窗口并设置位置
            cv::namedWindow("six_det", cv::WINDOW_NORMAL);
            cv::moveWindow("six_det", 100, 550); // 将窗口移动到屏幕坐标 (100, 100)

            // 设置窗口大小
            cv::resizeWindow("six_det", 640, 340);


            /* save last id */
            auto last_bbox_msgs = six_box_array_message; 

            six_box_array_message.boundaryboxes.clear();
            six_box_array_message.header.stamp = this->now(); 

            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            // 获取图像数据的 cv::Mat
            cv::Mat img = cv_ptr->image;
            unsigned long long start = log_get_timestamp();
            cv::Mat rgb_img;
            // std::vector<object_box_t> boxes;
            std::vector<Object> objects;
            cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
            six_img_rgb = rgb_img;
            /* car detect */
            algo_nn_detect_car(car_handle_6, rgb_img, 0.5, objects, &err);

            /* transform box for tracking */
            for(int i = 0; i < objects.size(); i++) 
            {
                objects[i].rect.x = (objects[i].rect.x * img.cols);
                objects[i].rect.y = (objects[i].rect.y * img.rows);
                objects[i].rect.width = (objects[i].rect.width * img.cols) - objects[i].rect.x;
                objects[i].rect.height = (objects[i].rect.height * img.rows) - objects[i].rect.y;
            }

            ObjectMeasure_6 obj_meas;

            LOGD(TAG, "6 algo_nn_detect_car number:%d", objects.size());

            /* tracking */
            vector<STrack> output_stracks = tracker_6.update(objects);

            // 用于后续的速度估计
            std::unordered_map<int, double> box_map;

            // 跟踪数据导出
            for (int i = 0; i < output_stracks.size(); i++)
            {
                vector<float> tlwh = output_stracks[i].tlwh;
                /* publish ros bbox */
                auto box_message = fusion_interfaces::msg::BoundaryBox();
                box_message.x1 = tlwh[0];
                box_message.y1 = tlwh[1];
                box_message.x2 = tlwh[0]+tlwh[2];
                box_message.y2 = tlwh[1]+tlwh[3];
                box_message.in_current_lane = 0;
                box_message.probability = output_stracks[i].score;
                box_message.car_id = output_stracks[i].track_id;

                /* dist is_in_current_lane */
                int x1 = std::round(box_message.x1);
                int y1 = std::round(box_message.y1);
                int x2 = std::round(box_message.x2);
                int y2 = std::round(box_message.y2);
                int cx = (x1 + x2) / 2;
                int cy =  y2;  
                bool flags = false;
                bool coord_x_flags = (cx > org_width / 3);
                cv::Scalar car_color(255, 0, 0);
                int lane_num = lane_x.is_inside(cx, cy, org_width, org_height);
                if(lane_num ==2) {   // 1 左车道  2 当前车道 3 右车道
                    box_message.in_current_lane = 1;
                    box_message.lane_id = 2; 
                }
                else if (lane_num == 1)
                {
                    box_message.lane_id = 1; 
                }
                else if (lane_num == 3)
                {
                    box_message.lane_id = 3; 
                }

                /* dis */
                if(coord_x_flags) {
                    char text[32];
                    auto result = obj_meas.calc(x2 - x1, x1+(x2-x1)/2);
                    if(result.first > 80)
                    {
                        printf("this det dist over 80m");
                        continue;
                    }
                    else{
                        // 用于速度估计
                        box_map[box_message.car_id] = result.first;
                        box_message.dist = result.first;
                        box_message.hori = result.second;
                    }
                        
                    // 根据横向距离判断车道号
                    if ( box_message.hori < 1.35 && box_message.hori > -1.35 )
                        box_message.lane_id = 2; 
                    else if ( box_message.hori > 1.35 && box_message.hori < 4.05 )
                        box_message.lane_id = 3; 
                    else if ( box_message.hori < -1.35 && box_message.hori > -4.05 )
                        box_message.lane_id = 1; 
                    if(box_message.dist < 40){   // 距离小于40m，报警
                        car_color = cv::Scalar(0, 0, 255);
                    }
                    sprintf(text, "ID:%d, %.2fm", output_stracks[i].track_id, box_message.dist);
                    putText(img, text, cv::Point(x1 , y1 -10), 0, 0.4, car_color, 1, cv::LINE_AA); 
                }
                else{
                    char text[32];
                    box_message.dist = -1.0;
                    sprintf(text, "ID:%d", output_stracks[i].track_id);
                    putText(img, text, cv::Point(x1 , y1 -10), 0, 0.4, car_color, 1, cv::LINE_AA); 
                }
                cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), car_color, 2);
                six_box_array_message.boundaryboxes.push_back(box_message);  
            }
            // LOGD(TAG, "six_box_velo_msgs: %d", six_box_velo_msgs.time_sec);
            // LOGD(TAG, "six_box_array_message: %d", six_box_array_message.header.stamp.sec);

            // --------------
            // 每次数据进来都计算差值，count以及初始记录时间，先把这个数据存储下来，单独一个函数计算最终的速度
            // 最好是单独一个线程每一秒计算一下数据的速度；
            if (six_box_array_message.header.stamp.sec - six_box_velo_msgs.time_sec > 1.0) {
                if (!six_box_velo_msgs.id_dist_map.empty()) {
                    ////////
                    std::vector<int> ids_to_remove;
                    for (const auto& [id, dist_count] : six_box_velo_msgs.id_dist_map) {
                        auto it = box_map.find(id);
                        if (it != box_map.end()) {
                            // 计算速度
                            double delta_distance =std::round((box_map[id] - (dist_count.dist / dist_count.count)) * 100.0) / 100.0;
                            double delta_time = six_box_array_message.header.stamp.sec - dist_count.time_sec_record;
                            double velocity = std::round(delta_distance / delta_time* 100.0) / 100.0;
                            // LOGD("TAG", "delta dist: %f", delta_distance);
                            // LOGD("TAG", "delta_time: %f", delta_time);
                            // LOGD("TAG", "[LOG] Object with ID %d has velocity: %.2f m/s", id, velocity);
                            for (auto& box : six_box_array_message.boundaryboxes) {
                                if (box.car_id == id) {
                                    box.velocity = velocity;
                                    six_box_velo_msgs.id_dist_map[id].last_velo = velocity;
                                    break;
                                }
                            }
                        } 
                        else {
                            ids_to_remove.push_back(id);
                        }
                    }
                    for (int id : ids_to_remove) {
                        six_box_velo_msgs.id_dist_map.erase(id);
                        LOGD("TAG", "[LOG] Removed object with ID %d from id_dist_map", id);
                    }
                    ////////
                }
                else{
                    for (const auto& box : six_box_array_message.boundaryboxes) {
                        six_box_velo_msgs.id_dist_map[box.car_id] = {box.dist, 1, six_box_array_message.header.stamp.sec};
                        LOGD("TAG", "[LOG] New object with ID %d added to id_dist_map", box.car_id);
                    }
                }
                six_box_velo_msgs.time_sec = six_box_array_message.header.stamp.sec;
            }
            else{
                for (auto& box : six_box_array_message.boundaryboxes){
                    auto it = six_box_velo_msgs.id_dist_map.find(box.car_id);
                    if (it != six_box_velo_msgs.id_dist_map.end()) {
                        it->second.dist = box.dist + it->second.dist;  // 待修改
                        it->second.count++;
                        box.velocity = it->second.last_velo;
                        LOGD("TAG", "[LOG] Updated object with ID %d in id_dist_map", box.car_id);
                    }
                    else{
                        // 没有就添加到里面取
                        six_box_velo_msgs.id_dist_map[box.car_id] = {box.dist, 1, six_box_array_message.header.stamp.sec};
                    }
                }
                // 将数据内容对应存储起来 有相同id的就叠加，没有就新加
            }


            //!-------------

            /* lane draw */
            six_bbox_pub_->publish(six_box_array_message);
            // if(!six_seg_mask.empty()){
            //     cv::Mat seg_mask2;
            //     cv::resize(six_seg_mask, seg_mask2, cv::Size(org_width, org_height), 0, 0, cv::INTER_NEAREST);
            //     //------//
            //     std::vector<std::vector<cv::Point>> contours;
            //     laneseg_to_contours(seg_mask2, contours);

            //     //printf("lane size=%d\r\n", contours.size());

            //     const cv::Scalar color_lists[] = {
            //     cv::Scalar(255, 0, 0),
            //     cv::Scalar(255, 255, 0),
            //     cv::Scalar(0, 255, 0),
            //     cv::Scalar(0, 255, 255),
            //     cv::Scalar(222, 128, 255),
            //     cv::Scalar(125, 255, 0),
            //     cv::Scalar(255, 0, 255)};
                
            //     for(auto& c : contours) { 
            //         cv::fillPoly(img, c, color_lists[rand() % 5], 8, 0); 
            //     }
            //     //------//

            //     // for(int y = 0; y < org_height; y++) {
            //     //     for(int x = 0; x < org_width; x++) {
            //     //         if(seg_mask2.at<uint8_t>(y, x) == 255) {
            //     //             img.at<cv::Vec3b>(y, x) = cv::Vec3b(0,255, 0);  
            //     //         }
            //     //     }
            //     // }
            //}
            sensor_msgs::msg::Image::SharedPtr six_det_img = cv_bridge::CvImage(std_msgs::msg::Header(), 
                                            "bgr8", img).toImageMsg();
            unsigned long long end = log_get_timestamp();
            LOGI(TAG, "thread_index[%d] %dx%d  time:%lluus",  0, 1280, 720, (end - start) );
            six_detimg_publisher_.publish(six_det_img);
            cv::imshow("six_det", img);
            cv::waitKey(30);
            // cv::imwrite("cat_det_res_6.jpg", img);
        }

        void twenfive_imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
        {
            // 创建窗口并设置位置
            cv::namedWindow("twenfive_det", cv::WINDOW_NORMAL);
            cv::moveWindow("twenfive_det", 100, 100); // 将窗口移动到屏幕坐标 (100, 100)

            // 设置窗口大小
            cv::resizeWindow("twenfive_det", 640, 340);

            //printf("received_25\n");
            /* save last id */
            auto last_bbox_msgs = twenfive_box_array_message; 

            twenfive_box_array_message.boundaryboxes.clear();
            twenfive_box_array_message.header.stamp = this->now(); 

            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            // 获取图像数据的 cv::Mat
            cv::Mat img = cv_ptr->image;
            unsigned long long start = log_get_timestamp();
            cv::Mat rgb_img;
            // std::vector<object_box_t> boxes;
            std::vector<Object> objects;
            cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
            twenfive_img_rgb = rgb_img;
            /* car detect */
            algo_nn_detect_car(car_handle_25, rgb_img, 0.5, objects, &err);

            /* transform box for tracking */
            for(int i = 0; i < objects.size(); i++) 
            {
                objects[i].rect.x = (objects[i].rect.x * img.cols);
                objects[i].rect.y = (objects[i].rect.y * img.rows);
                objects[i].rect.width = (objects[i].rect.width * img.cols) - objects[i].rect.x;
                objects[i].rect.height = (objects[i].rect.height * img.rows) - objects[i].rect.y;
            }

            ObjectMeasure_25 obj_meas;

            LOGD(TAG, "algo_nn_detect_car number:%d", objects.size());

            /* tracking */
            vector<STrack> output_stracks = tracker_25.update(objects);

            std::unordered_map<int, double> box_map;

            for (int i = 0; i < output_stracks.size(); i++)
            {
                vector<float> tlwh = output_stracks[i].tlwh;
                /* publish ros bbox */
                auto box_message = fusion_interfaces::msg::BoundaryBox();
                box_message.x1 = tlwh[0];
                box_message.y1 = tlwh[1];
                box_message.x2 = tlwh[0]+tlwh[2];
                box_message.y2 = tlwh[1]+tlwh[3];
                box_message.in_current_lane = 0;
                box_message.probability = output_stracks[i].score;
                box_message.car_id = -(output_stracks[i].track_id);

                /* dist is_in_current_lane */
                int x1 = std::round(box_message.x1);
                int y1 = std::round(box_message.y1);
                int x2 = std::round(box_message.x2);
                int y2 = std::round(box_message.y2);
                int cx = (x1 + x2) / 2;
                int cy =  y2;  
                bool flags = false;
                bool coord_x_flags = (cx > org_width / 3);
                cv::Scalar car_color(255, 0, 0);
                int lane_num = lane_x.is_inside(cx, cy, org_width, org_height);
                if(lane_num ==2) {   // 1 左车道  2 当前车道 3 右车道
                    box_message.in_current_lane = 1;
                    box_message.lane_id = 2; 
                }
                else if (lane_num == 1)
                {
                    box_message.lane_id = 1; 
                }
                else if (lane_num == 3)
                {
                    box_message.lane_id = 3; 
                }

                /* dis */
                if(coord_x_flags) {
                    char text[32];
                    auto result = obj_meas.calc(x2 - x1, x1+(x2-x1)/2);
                    box_message.dist = result.first;
                    box_message.hori = result.second;
                    box_map[box_message.car_id] = result.first;

                    if(box_message.dist < 40){   // 距离小于40m，报警
                        car_color = cv::Scalar(0, 0, 255);
                    }
                    // 根据横向距离判断车道号
                    if ( box_message.hori < 1.35 && box_message.hori > -1.35 )
                        box_message.lane_id = 2; 
                    else if ( box_message.hori > 1.35 && box_message.hori < 4.05 )
                        box_message.lane_id = 3; 
                    else if ( box_message.hori < -1.35 && box_message.hori > -4.05 )
                        box_message.lane_id = 1; 
                    sprintf(text, "ID:%d, %.2fm", output_stracks[i].track_id, box_message.dist);
                    putText(img, text, cv::Point(x1 , y1 -10), 0, 0.4, car_color, 1, cv::LINE_AA); 
                }
                else{
                    char text[32];
                    box_message.dist = -1.0;
                    sprintf(text, "ID:%d", output_stracks[i].track_id);
                    putText(img, text, cv::Point(x1 , y1 -10), 0, 0.4, car_color, 1, cv::LINE_AA); 
                }

                cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), car_color, 2);
                twenfive_box_array_message.boundaryboxes.push_back(box_message);  
            }

            /*
                速度估计：只有当时间差值大于一定数值的时候，才开始计算速度，否则速度不变
                先获取时间
            */
            if (twenfive_box_array_message.header.stamp.sec - twenfive_box_velo_msgs.time_sec > 0.5) {
                if (!twenfive_box_velo_msgs.id_dist_map.empty()) {
                    ////////
                    std::vector<int> ids_to_remove;
                    for (const auto& [id, dist_count] : twenfive_box_velo_msgs.id_dist_map) {
                        auto it = box_map.find(id);
                        if (it != box_map.end()) {
                            // 计算速度
                            double delta_distance = std::round((box_map[id] - (dist_count.dist / dist_count.count)) * 100.0) / 100.0;
                            double delta_time = twenfive_box_array_message.header.stamp.sec - dist_count.time_sec_record;
                            double velocity = std::round(delta_distance / delta_time* 100.0) / 100.0;
                            // LOGD("TAG", "delta dist: %f", delta_distance);
                            // LOGD("TAG", "delta_time: %f", delta_time);
                            // LOGD("TAG", "[LOG] Object with ID %d has velocity: %.2f m/s", id, velocity);
                            for (auto& box : twenfive_box_array_message.boundaryboxes) {
                                if (box.car_id == id) {
                                    box.velocity = velocity;
                                    break;
                                }
                            }
                        } 
                        else {
                            ids_to_remove.push_back(id);
                        }
                    }
                    for (int id : ids_to_remove) {
                        twenfive_box_velo_msgs.id_dist_map.erase(id);
                        LOGD("TAG", "[LOG] Removed object with ID %d from id_dist_map", id);
                    }
                    ////////
                }
                else{
                    for (const auto& box : twenfive_box_array_message.boundaryboxes) {
                        twenfive_box_velo_msgs.id_dist_map[box.car_id] = {box.dist, 1, twenfive_box_array_message.header.stamp.sec};
                        LOGD("TAG", "[LOG] New object with ID %d added to id_dist_map", box.car_id);
                    }
                }
                twenfive_box_velo_msgs.time_sec = twenfive_box_array_message.header.stamp.sec;
            }
            else{
                for (const auto& box : twenfive_box_array_message.boundaryboxes){
                    auto it = twenfive_box_velo_msgs.id_dist_map.find(box.car_id);
                    if (it != twenfive_box_velo_msgs.id_dist_map.end()) {
                        it->second.dist = box.dist + it->second.dist;  // 待修改
                        it->second.count++;
                        LOGD("TAG", "[LOG] Updated object with ID %d in id_dist_map", box.car_id);
                    }
                    else{
                        // 没有就添加到里面取
                        twenfive_box_velo_msgs.id_dist_map[box.car_id] = {box.dist, 1, twenfive_box_array_message.header.stamp.sec};
                    }
                }
                // 将数据内容对应存储起来 有相同id的就叠加，没有就新加
            }

            /* lane draw */
            twenfive_bbox_pub_->publish(twenfive_box_array_message);
            // if(!twenfive_seg_mask.empty()){
            //     cv::Mat seg_mask2;
            //     cv::resize(twenfive_seg_mask, seg_mask2, cv::Size(org_width, org_height), 0, 0, cv::INTER_NEAREST);
            //     //-----//
            //     std::vector<std::vector<cv::Point>> contours;
            //     laneseg_to_contours(seg_mask2, contours);

            //     //printf("lane size=%d\r\n", contours.size());

            //     const cv::Scalar color_lists[] = {
            //     cv::Scalar(255, 0, 0),
            //     cv::Scalar(255, 255, 0),
            //     cv::Scalar(0, 255, 0),
            //     cv::Scalar(0, 255, 255),
            //     cv::Scalar(222, 128, 255),
            //     cv::Scalar(125, 255, 0),
            //     cv::Scalar(255, 0, 255)};
                
            //     for(auto& c : contours) { 
            //         cv::fillPoly(img, c, color_lists[rand() % 5], 8, 0); 
            //     }
            //     // //------//
            //     // for(int y = 0; y < org_height; y++) {
            //     //     for(int x = 0; x < org_width; x++) {
            //     //         if(seg_mask2.at<uint8_t>(y, x) == 255) {
            //     //             img.at<cv::Vec3b>(y, x) = cv::Vec3b(0,255, 0);  
            //     //         }
            //     //     }
            //     // }
            //}
            sensor_msgs::msg::Image::SharedPtr twenfive_det_img = cv_bridge::CvImage(std_msgs::msg::Header(), 
                                            "bgr8", img).toImageMsg();
            unsigned long long end = log_get_timestamp();
            LOGI(TAG, "thread_index[%d] %dx%d  time:%lluus",  0, 1280, 720, (end - start) );
            twenfive_detimg_publisher_.publish(twenfive_det_img);


            
            cv::imshow("twenfive_det", img);
            cv::waitKey(30);
            // cv::imwrite("cat_det_res_6.jpg", img);
        }

        void periodicTask()
        {
            while (!stop_thread_ )
            {
                if(six_img_rgb.empty() ){
                    LOGD(TAG, "six car lane img empty");
                }
                else{
                    algo_nn_lane_seg(lane_handle, six_img_rgb, 0.5, six_seg_mask, &err);
                    LOGD(TAG, "car lane img detecting");
                    lane_x.set_mask(six_seg_mask);
                }
                if( twenfive_img_rgb.empty()){
                    LOGD(TAG, "twenfive car lane img empty");
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                else{
                    algo_nn_lane_seg(lane_handle, twenfive_img_rgb, 0.5, twenfive_seg_mask, &err);
                    lane_x.set_mask(twenfive_seg_mask);
                    LOGD(TAG, "car lane img detecting");
                }
            }
        }

        void startPeriodicTaskThread()
        {
            stop_thread_ = false;
            std::thread thread(&CAR_DETECT_NODE::periodicTask, this);
            thread.detach();
        }

        void stopPeriodicTaskThread()
        {
            stop_thread_ = true;
        }
        
        void SORT_CAR(){
            printf("test");
        }

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
        
    private:
        image_transport::Subscriber twenfive_img_sub_;
        image_transport::Subscriber six_img_sub_;

        //rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr six_img_sub_;
        //rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr twenfive_img_sub_;

        rclcpp::Publisher<fusion_interfaces::msg::BoundaryBoxArray>::SharedPtr six_bbox_pub_;
        rclcpp::Publisher<fusion_interfaces::msg::BoundaryBoxArray>::SharedPtr twenfive_bbox_pub_;

        image_transport::Publisher six_detimg_publisher_;
        image_transport::Publisher twenfive_detimg_publisher_;

        ALGO_ERR_CODE err;
        void* car_handle_6;
        void* car_handle_25;
        void* lane_handle;

        // img for lane detect
        int org_width = 640;
        int org_height = 384;

        cv::Mat six_img_rgb;
        cv::Mat six_seg_mask;

        cv::Mat twenfive_img_rgb;
        cv::Mat twenfive_seg_mask;

        LaneX lane_x;
        BYTETracker tracker_6;
        BYTETracker tracker_25;

        // thread for lane detect
        bool stop_thread_; // sign for stop lane thread  

        /* msgs for detection */
        fusion_interfaces::msg::BoundaryBoxArray six_box_array_message;
        Detection_Velo six_box_velo_msgs;
        fusion_interfaces::msg::BoundaryBoxArray twenfive_box_array_message;
        Detection_Velo twenfive_box_velo_msgs;

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CAR_DETECT_NODE>());
    rclcpp::shutdown();
    return 0;
}
