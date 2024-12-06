#include <fcntl.h>
#include <dirent.h>
#include <linux/input.h>
#include <sys/stat.h>
#include <pthread.h>
#include <time.h>


#include <boost/bind.hpp>
#include <boost/thread.hpp>


#include "racgnss_decoder.h"



// TODO: 创建全局变量实时更新经纬度数据

CanControl::CanControl() : Node("racgnss_decoder_node")
{
	ros::NodeHandle private_node("~");
	RCLCPP_INFO(this->get_logger(), "Initializing GNSS Driver Node");
	GNSS_0x360_pub_ = this->create_publisher<racgnss_can_msgs::msg::rac_0x360>("gnss_0x360", 10);
	GNSS_0x361_pub_ = this->create_publisher<racgnss_can_msgs::msg::rac_0x361>("gnss_0x361", 10);
	GNSS_0x362_pub_ = this->create_publisher<racgnss_can_msgs::msg::rac_0x362>("gnss_0x362", 10);
	GNSS_0x363_pub_ = this->create_publisher<racgnss_can_msgs::msg::rac_0x360>("gnss_0x363", 10);
    GNSS_0x364_pub_ = this->create_publisher<racgnss_can_msgs::msg::rac_0x361>("gnss_0x364", 10);
    GNSS_0x365_pub_ = this->create_publisher<racgnss_can_msgs::msg::rac_0x362>("gnss_0x365", 10);
    GNSS_0x366_pub_ = this->create_publisher<racgnss_can_msgs::msg::rac_0x360>("gnss_0x366", 10);
	current_pos_pub_ = this->create_publisher<sensor_msgs::msg::NavSatFix>("gps", 10);

	
	打开设备
	dev_handler_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
	if (dev_handler_ < 0) 
	{
		RCLCPP_ERROR(">>open can deivce error!");
		return;
	}
    else
	{
		RCLCPP_INFO(this->get_logger(), ">>open can deivce success!");
	}


	struct ifreq ifr;
	
	std::string can_name("can0");

	strcpy(ifr.ifr_name,can_name.c_str());

	ioctl(dev_handler_,SIOCGIFINDEX, &ifr);


    // bind socket to network interface
	struct sockaddr_can addr;
	memset(&addr, 0, sizeof(addr));
	addr.can_family = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;
	int ret = ::bind(dev_handler_, reinterpret_cast<struct sockaddr *>(&addr),sizeof(addr));
	if (ret < 0) 
	{
		ROS_ERROR(">>bind dev_handler error!\r\n");
		return;
	}

	//创建接收发送数据线程
	boost::thread recvdata_thread(boost::bind(&CanControl::recvData, this));

    // 创建发布当前位置信息的线程
    boost::thread publish_thread(boost::bind(&CanControl::publishCurrentPos, this));

	ros::spin(); // ??
	
	close(dev_handler_);
}


CanControl::~CanControl()
{

}


//数据接收解析线程
void CanControl::recvData()
{

	while(ros::ok())
	{

		if(read(dev_handler_, &recv_frames_[0], sizeof(recv_frames_[0])) >= 0)
		{
			for(int j=0;j<1;j++)
			{
				
				switch (recv_frames_[0].can_id)
				{
					//速度控制反馈
					ROS_INFO("Received_can_msgs");

					case 0x360:
					{
						racgnss_can_msgs::rac_0x360 msg;
						msg.Week = (int)((0xff & recv_frames_[0].data[1] << 8) | (0xff & recv_frames_[0].data[0]));
						msg.Millisecond = (int)((0xff & recv_frames_[0].data[3] << 16) | (0xff & recv_frames_[0].data[2] << 8) | (0xff & recv_frames_[0].data[1]));
						msg.Position_type =	(int)((0xff & recv_frames_[0].data[4] << 8) | (0xff & recv_frames_[0].data[5]));
						msg.Satellites = (int)((0xff & recv_frames_[0].data[6] << 8) | (0xff & recv_frames_[0].data[7]));
						
						
						ROS_INFO("msg.Week :%d", msg.Week);
						ROS_INFO("msg.Milliseconds :%d", msg.Millisecond);
						ROS_INFO("msg.Position_type :%d", msg.Position_type);
						ROS_INFO("msg.Satellites :%d", msg.Satellites);

						GNSS_0x360_pub_.publish(msg);

						break;
					}

					case 0x361:
					{
						racgnss_can_msgs::rac_0x361 msg;
						msg.latitude = ((int)(recv_frames_[0].data[3] << 24 | recv_frames_[0].data[2] << 16 | recv_frames_[0].data[1] << 8 | recv_frames_[0].data[0]));
						msg.longitude = ((int)(recv_frames_[0].data[7] << 24 | recv_frames_[0].data[6] << 16 | recv_frames_[0].data[5] << 8 | recv_frames_[4].data[0]));
						
						current_pos.latitude = msg.latitude;
						current_pos.longitude = msg.longitude; //TODO : 待确认

						ROS_INFO("msg.latitude :%d", msg.latitude);
						ROS_INFO("msg.longitude :%d", msg.longitude);

						GNSS_0x361_pub_.publish(msg);
						break;
					}

					case 0x362:
					{
					    racgnss_can_msgs::rac_0x362 msg;
						msg.altitude = ((int)(recv_frames_[0].data[3] << 24 | recv_frames_[0].data[2] << 16 | recv_frames_[0].data[1] << 8 | recv_frames_[0].data[0]));
						msg.heading = ((int)(recv_frames_[0].data[7] << 24 | recv_frames_[0].data[6] << 16 | recv_frames_[0].data[5] << 8 | recv_frames_[4].data[0]));
						current_pos.altitude = msg.altitude;
						ROS_INFO("msg.altitude :%d", msg.altitude);
						ROS_INFO("msg.heading :%d", msg.heading);

						GNSS_0x362_pub_.publish(msg);
						break;
					}
					


				}

				
				}

			}

					
		}
}

void CanControl::publishCurrentPos()
{
    ros::Rate rate(100); // 10 Hz
    while (ros::ok())
    {
        // 发布 current_pos
        current_pos.header.stamp = ros::Time::now();
		current_pos.header.frame_id = "gps";
		//
		current_pos.latitude = 31.0643068288;
		current_pos.longitude = 111.687814458;
		current_pos.altitude = 150.1476;
		//
        // 假设有一个 publisher 为 current_pos_pub_
        current_pos_pub_.publish(current_pos);
        rate.sleep();
    }
}




int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CanControl>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}