#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include "dirent.h"
#include <unistd.h>


#include <dlfcn.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include <sys/mman.h>
#include <sys/ioctl.h>
#include <asm/types.h>
#include <linux/videodev2.h>
#include <libv4l2.h>
#include <signal.h>
#include <stdint.h>
#include <inttypes.h>

#include "camera_video.h"
#include "inference.h"
#include "MyDetect.h"
#include "log.h"

//#include "log.h"

#define TAG  "test"

using namespace std;
using namespace std::chrono;

static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
#define CLEAR(x) memset(&(x), 0, sizeof(x))

int xioctl(int fd, int request, void *arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

void errno_exit(const char *s) {
    fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
    exit(EXIT_FAILURE);
}
 
#define TEST_HREAD_NUM                                 1
NN_CORE_MASK core_mask_tab[] = {NN_CORE_0, NN_CORE_1};
VIDEO_ENUM_INDEX video_enum[] = {HD_USB0, HD_USB1};
void* thread_func(void* arg) {

	int index = (int)arg;
	
	algo_config_t cfg;

    void* cam_inst = camera_open(video_enum[index], 1280, 720, 15, FRAME_TYPE_BGR888,  FRAME_NOT_ROT_FILP, NULL); 
	
	cfg.algo_opt_mask = OPT_MASK(ALGO_CAR_DETECT);// | OPT_MASK(ALGO_LANE_SEG);
	cfg.model_path[ALGO_CAR_DETECT] = "";//"model/model_car_det.rknn";
	//cfg.model_path[ALGO_LANE_SEG] = "model/model_seg.rknn";
	cfg.core_mask[ALGO_CAR_DETECT] = core_mask_tab[index]; 
	//cfg.core_mask[ALGO_LANE_SEG] = NN_CORE_0; 
	
	ALGO_ERR_CODE err = ALGO_ERR_NONE;
	void* inst = algo_nn_init(&cfg,  &err);

	cv::Mat img = cv::imread("/home/orangepi/work/rknn_det_seg/car-lane-imgs/0001.jpg");
	if(img.empty()) {
		algo_nn_deinit(inst, &err);
		exit(-1);
	}

	//LOGI(TAG, "read img:%dx%d", img.cols, img.rows);
	std::vector<object_box_t> boxes;
	
	int count = 50;
	unsigned long long start = log_get_timestamp();
	for(int i = 0; i < count; i++) {
		 frame_data_t frame_data; 
         int ret = camera_get_frame(cam_inst, &frame_data, NULL) ;
		if(ret == 0) {
			algo_nn_detect_car(inst, img, 0.3, boxes, &err);
            release_frame(&frame_data, NULL);
         }
	}
	unsigned long long end = log_get_timestamp();
	LOGI(TAG, "thread_index[%d] %dx%d get frame && detect object  time:%lluus",  index, 1280, 720, (end - start) / count);

	//printf("algo_nn_init err:%x\n", err);

	//finish_handle:
	algo_nn_deinit(inst, &err);


	camera_close(cam_inst, NULL);



	return (void*)0;
}


#include <pthread.h>
int main(int argc, char** argv)
{
	pthread_t tid[TEST_HREAD_NUM];
	for(int i = 0; i < TEST_HREAD_NUM; i++) {
		pthread_create(&tid[i], NULL, thread_func, (void*)i);
	}

	for(int i = 0; i < TEST_HREAD_NUM; i++) {
		pthread_join(tid[i], NULL);
	}
	
	
	return 0;
}


