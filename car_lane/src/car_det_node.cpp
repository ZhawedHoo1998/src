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

#define    TAG                            "test-car-det"

using namespace std;
using namespace cv;

using std::placeholders::_1;

// for car dist judge
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


class CAR_DETECT_NODE : public rclcpp::Node
{
    public:
        CAR_DETECT_NODE () : Node("car_detect_node")
        {
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

            //twenfive_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/camera/twenfive/image_raw/compressed", 10, 
              //                      std::bind(&CAR_DETECT_NODE::twenfive_imageCallback, this, std::placeholders::_1));

            six_bbox_pub_ = this->create_publisher<fusion_interfaces::msg::BoundaryBoxArray>("/car_detect/six_bbox", 10);
            twenfive_bbox_pub_ = this->create_publisher<fusion_interfaces::msg::BoundaryBoxArray>("/car_detect/twenfive_bbox", 10);

            algo_config_t cfg_6;
            cfg_6.algo_opt_mask = OPT_MASK(ALGO_CAR_DETECT);// | OPT_MASK(ALGO_LANE_SEG);
            cfg_6.model_path[ALGO_CAR_DETECT] = "";
            cfg_6.core_mask[ALGO_CAR_DETECT] = NN_CORE_0; 

            printf("******NN_CORE_0=%x\r\n", NN_CORE_0);

            algo_config_t cfg_25;
            cfg_25.algo_opt_mask = OPT_MASK(ALGO_CAR_DETECT);// | OPT_MASK(ALGO_LANE_SEG);
            cfg_25.model_path[ALGO_CAR_DETECT] = "";
            cfg_25.core_mask[ALGO_CAR_DETECT] = NN_CORE_1; 

            car_handle_6 = algo_nn_init(&cfg_6, &err);
            car_handle_25 = algo_nn_init(&cfg_25, &err);

            if(!car_handle_6 || !car_handle_25) {
                exit(-1);
            }

        }

        // ~CAR_DETECTION_NODE
        // {
        //     algo_nn_deinit(car_handle, &err);
        // }


        void six_imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
        {
            printf("received_6\n");
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            // 获取图像数据的 cv::Mat
            cv::Mat img = cv_ptr->image;
            int org_width = img.cols;
            int org_height = img.rows;
            unsigned long long start = log_get_timestamp();
            cv::Mat rgb_img;
            std::vector<object_box_t> boxes;
            cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
            algo_nn_detect_car(car_handle_6, rgb_img, 0.5, boxes, &err);
            unsigned long long end = log_get_timestamp();
            ObjectMeasure obj_meas;
            //LOGI(TAG, "thread_index[%d] %dx%d  detect object size:%d time:%lluus",  0, 1280, 720, boxes.size(), (end - start) / count);
            // box_array_msg
            auto box_array_message = fusion_interfaces::msg::BoundaryBoxArray();

            box_array_message.header.frame_id = "camera";
            box_array_message.header.stamp = this->now();

            for(auto& box : boxes) {
                LOGD(TAG, "\t[%.3f,%.3f,%.3f,%.3f]", box.x1, box.y1, box.x2, box.y2);
                cv::rectangle(img, cv::Point(box.x1 * img.cols, box.y1 * img.rows), cv::Point(box.x2 * img.cols, box.y2 * img.rows),cv::Scalar(255, 0, 0), 2);
                auto box_message = fusion_interfaces::msg::BoundaryBox();

                box_message.x1 = box.x1 * img.cols;
                box_message.y1 = box.y1 * img.rows;
                box_message.x2 = box.x2 * img.cols;
                box_message.y2 = box.y2 * img.rows;
                box_message.in_current_lane = 0;
                box_message.probability = box.prob;
                
                int x1 = std::round(box.x1 * org_width);
                int y1 = std::round(box.y1 * org_height);
                int x2 = std::round(box.x2 * org_width);
                int y2 = std::round(box.y2 * org_height);
                int cx = (x1 + x2) / 2;
                int cy =  y2;  
                bool flags = false;
                bool coord_x_flags = (cx > org_width / 3);
                /* dis */
                if(coord_x_flags) {
                    char text[32];
                    box_message.dist = obj_meas.calc(x2 - x1 + 1);
                    sprintf(text, "%.2fm", box_message.dist);
                    putText(img, text, cv::Point(x1 , y1 -10), 0, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA); 
                }

                box_array_message.boundaryboxes.push_back(box_message);


            }
            six_bbox_pub_->publish(box_array_message);

            sensor_msgs::msg::Image::SharedPtr six_det_img = cv_bridge::CvImage(std_msgs::msg::Header(), 
                                                            "bgr8", img).toImageMsg();

            six_detimg_publisher_.publish(six_det_img);
            //cv::imwrite("cat_det_res_6.jpg", img);
        }

        void twenfive_imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)  
        {
            printf("received_25\n");
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            // 获取图像数据的 cv::Mat
            cv::Mat img = cv_ptr->image;
            int org_width = img.cols;
            int org_height = img.rows;
            unsigned long long start = log_get_timestamp();
            cv::Mat rgb_img;
            std::vector<object_box_t> boxes;
            cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
            ObjectMeasure obj_meas;
            algo_nn_detect_car(car_handle_25, rgb_img, 0.5, boxes, &err);
            unsigned long long end = log_get_timestamp();
            auto box_array_message = fusion_interfaces::msg::BoundaryBoxArray();
            box_array_message.header.frame_id = "camera";
            box_array_message.header.stamp = this->now();
            //LOGI(TAG, "thread_index[%d] %dx%d  detect object size:%d time:%lluus",  0, 1280, 720, boxes.size(), (end - start) / count);
            for(auto& box : boxes) {
                LOGD(TAG, "\t[%.3f,%.3f,%.3f,%.3f]", box.x1, box.y1, box.x2, box.y2);
                cv::rectangle(img, cv::Point(box.x1 * img.cols, box.y1 * img.rows), cv::Point(box.x2 * img.cols, box.y2 * img.rows),cv::Scalar(255, 0, 0), 2);
                auto box_message = fusion_interfaces::msg::BoundaryBox();
                box_message.x1 = box.x1 * img.cols;
                box_message.y1 = box.y1 * img.rows;
                box_message.x2 = box.x2 * img.cols;
                box_message.y2 = box.y2 * img.rows;
                box_message.in_current_lane = 0;
                box_message.probability = box.prob;
            
                int x1 = std::round(box.x1 * org_width);
                int y1 = std::round(box.y1 * org_height);
                int x2 = std::round(box.x2 * org_width);
                int y2 = std::round(box.y2 * org_height);
                int cx = (x1 + x2) / 2;
                int cy =  y2;  
                bool flags = false;
                bool coord_x_flags = (cx > org_width / 3);
                /* dis */
                if(coord_x_flags) {
                    char text[32];
                    box_message.dist = obj_meas.calc(x2 - x1 + 1);
                    sprintf(text, "%.2fm", box_message.dist);
                    putText(img, text, cv::Point(x1 , y1 -10), 0, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA); 
                }
                box_array_message.boundaryboxes.push_back(box_message);

            }
            twenfive_bbox_pub_->publish(box_array_message);
            sensor_msgs::msg::Image::SharedPtr twenfive_det_img = cv_bridge::CvImage(std_msgs::msg::Header(), 
                                                            "bgr8", img).toImageMsg();

            twenfive_detimg_publisher_.publish(twenfive_det_img);
            //cv::imwrite("cat_det_res_25.jpg", img);
        }

    private:
        //rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr six_img_sub_;
        //rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr twenfive_img_sub_;
        image_transport::Subscriber twenfive_img_sub_;
        image_transport::Subscriber six_img_sub_;

        rclcpp::Publisher<fusion_interfaces::msg::BoundaryBoxArray>::SharedPtr six_bbox_pub_;
        rclcpp::Publisher<fusion_interfaces::msg::BoundaryBoxArray>::SharedPtr twenfive_bbox_pub_;

        image_transport::Publisher six_detimg_publisher_;
        image_transport::Publisher twenfive_detimg_publisher_;

        ALGO_ERR_CODE err;
        void* car_handle_6;
        void* car_handle_25;
    
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CAR_DETECT_NODE>());
    rclcpp::shutdown();
    return 0;
}
