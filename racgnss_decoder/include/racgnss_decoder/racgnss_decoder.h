#ifndef __PIDCONTROL_NODE_H__
#define __PIDCONTROL_NODE_H__



#include "ros/ros.h"
#include "std_msgs/Int32.h"
#include "racgnss_can_msgs/rac_0x360.h"
#include "racgnss_can_msgs/rac_0x361.h"
#include "racgnss_can_msgs/rac_0x362.h"
#include "racgnss_can_msgs/rac_0x363.h"
#include "racgnss_can_msgs/rac_0x364.h"
#include "racgnss_can_msgs/rac_0x365.h"
#include "racgnss_can_msgs/rac_0x366.h"


#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sensor_msgs/NavSatFix.h>

#include <string>

#include <linux/can.h>
#include <linux/can/raw.h>



class CanControl
{
public:
	CanControl();
	~CanControl();
	
	void run();
private:
	ros::NodeHandle nh_;

	ros::Publisher GNSS_0x360_pub_;
	ros::Publisher GNSS_0x361_pub_;
	ros::Publisher GNSS_0x362_pub_;
	ros::Publisher GNSS_0x363_pub_;
	ros::Publisher GNSS_0x364_pub_;
	ros::Publisher GNSS_0x365_pub_;
	ros::Publisher GNSS_0x366_pub_;

	ros::Publisher current_pos_pub_;


	boost::mutex cmd_mutex_;

	int dev_handler_;
	can_frame send_frames_[2];
	can_frame recv_frames_[1];

	sensor_msgs::NavSatFix current_pos;

	void recvData();
	void publishCurrentPos();

};




#endif

