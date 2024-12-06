#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <mutex>
#include <thread>

#include "vpu_decode.h"
#include "log.h"
#include <rga/RgaApi.h>

int rga_control_buffer_init(bo_t *bo, int *buf_fd, int width, int height, int bpp);
void rga_control_buffer_deinit(bo_t *bo, int buf_fd);

static void delay_ms(uint32_t n_ms) {
    struct timeval tv;
    tv.tv_sec = n_ms / 1000;
    tv.tv_usec = n_ms % 1000;
    select(NULL, NULL, NULL, NULL, &tv);
}

int main() {
	struct vpu_decode decode = {0};
	bo_t dec_bo = {0};
	int dec_fd = -1;
	int width = 1280;
	int height = 720;

	rga_control_buffer_init(&dec_bo, &dec_fd, width, height, 16);

	
	vpu_decode_jpeg_init(&decode, width, height);
	
	#if 1

	FILE* fp_in = fopen("720.jpg", "rb");
	fseek(fp_in, 0, SEEK_END);
	uint32_t size = ftell(fp_in);
	fseek(fp_in, 0, SEEK_SET);
	uint8_t* buf = new uint8_t[size];
	fread(buf, size, 1, fp_in);
	fclose(fp_in);

	uint8_t* out_buf = new uint8_t[width * height * 3];

	unsigned long long total[2] = {0};
	int count = 100;
	for(int i = 0; i < count; i++) {
		unsigned long long t0 = log_get_timestamp();
		vpu_decode_jpeg_doing(&decode, buf, size, dec_fd, dec_bo.ptr);
	    unsigned long long t1 = log_get_timestamp();

		RgaSURF_FORMAT dec_fmt = (decode.fmt == MPP_FMT_YUV422SP ? RK_FORMAT_YCbCr_422_SP : RK_FORMAT_YCbCr_420_SP);
		
	
		rga_info_t src, dst;
		int fmt = RK_FORMAT_RGB_888;

		memset(&src, 0, sizeof(rga_info_t));
		src.fd = -1;
		src.virAddr = dec_bo.ptr;
		src.mmuFlag = 1;
		src.rotation = 0;//HAL_TRANSFORM_ROT_90;
		rga_set_rect(&src.rect, 0, 0, width, height, width, height, dec_fmt);

		memset(&dst, 0, sizeof(rga_info_t));
		dst.fd = -1;
		dst.virAddr = out_buf;
		dst.mmuFlag = 1;
		rga_set_rect(&dst.rect, 0, 0, width, height, width, height, fmt);
		if (c_RkRgaBlit(&src, &dst, NULL)) {
			printf("%s: rga fail\n", __func__);
			//goto err_handle;
		}
		unsigned long long t2 = log_get_timestamp();
		total[0] += t1 - t0;
		total[1] += t2 - t1;

	    delay_ms(10);
	}
	
	LOGI("test", "t1 - t0 cost:%lluus\r\n", (total[0]) / count);
	LOGI("test", "t2 - t1 cost:%lluus\r\n", (total[1]) / count);


	FILE* out_fp = fopen("out.rgb", "wb");
	fwrite(out_buf, width * height * 3, 1, out_fp);
	fclose(out_fp);

	err_handle:
	if(buf) delete  buf;
	if(out_buf)delete  out_buf;
	#endif
	vpu_decode_jpeg_done(&decode);
	rga_control_buffer_deinit(&dec_bo, dec_fd);
	
}