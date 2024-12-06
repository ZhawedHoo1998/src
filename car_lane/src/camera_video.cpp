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

#include "camera_video.h"
#include "log.h"

//#include "clist.h"
#include "vpu_decode.h"
//#include "rga_control.h"
//#include "rkfacial.h"
#include <rga/RgaApi.h>

#define TAG                 "camera_video"

#define obj_clr(v)   memset(&v, 0, sizeof(v))

static void str_split(const char* str, const char split, std::vector<std::string>& results);
static inline size_t least_common_multiple(size_t a, size_t b);
static int strlenx(const char* str, int max_length);
static int running_command(const char *command,char *result, int max_length);
static std::vector<video_dev_t> get_video_lists_init(void);
static void get_video_lists(std::vector<video_dev_t>& video_dev_lists, int* err);
static int get_dev_path(VIDEO_ENUM_INDEX video_index, char  dev_path[64], int* dev_index, VIDEO_TYPE* video_type, int* err);

int rga_control_buffer_init(bo_t *bo, int *buf_fd, int width, int height, int bpp);
void rga_control_buffer_deinit(bo_t *bo, int buf_fd);

static std::mutex mtx;
static std::vector<video_dev_t> g_video_dev_lists = get_video_lists_init(); 

#define BUF_COUNT                   3
#define FMT_NUM_PLANES              1

typedef struct mipi_map_t_ {
    uint8_t *mptr[BUF_COUNT];
    uint32_t size[BUF_COUNT];
} mipi_map_t;

typedef struct mipi_camera_t_{
    // ============= v4l2 =============
    int fd;
    char dev_path[64];
    enum v4l2_buf_type buff_type;
    VIDEO_TYPE video_type;
    mipi_map_t mipi_map;
    // ============= output =============  
    int out_width;
    int out_height;
    FRAME_ROTATION_DIR rotation;
    FRANE_TYPE out_format;
    //============= jpeg ============= 
    struct vpu_decode decode;
	bo_t dec_bo;
	int dec_fd;
} mipi_camera_t;



void* camera_open(VIDEO_ENUM_INDEX video_index, int width, int height, int fps, FRANE_TYPE frame_type, FRAME_ROTATION_DIR rotation, int* err) {
    char dev_path[64] = {0};
    int dev_index = -1;
    VIDEO_TYPE video_type;
    int fd = -1;
    mipi_map_t mipi_map {0}; //obj_clr(mipi_map);
    enum v4l2_buf_type buff_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;


    do {
        if(get_dev_path(video_index, dev_path, &dev_index, &video_type, nullptr) < 0){
           break;
        }

        LOGI(TAG,"dev_path:%s dev_index:%d video_type:%d open...", dev_path, dev_index, video_type);
        // 打开摄像头
        int fd = open(dev_path, O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (fd == -1) {
            LOGE(TAG, "Error opening camera device[%s]", dev_path);
            break;
        }

        // 查询设备属性
        struct v4l2_capability cap;
        int ret = ioctl(fd, VIDIOC_QUERYCAP, &cap);
        if(ret < 0) {
            LOGE(TAG, "1. ioctl: VIDIOC_QUERYCAP fail");
            break;
        }

         LOGI(TAG, "Driver Name:%s\n\tCard Name:%s\n\tBus info:%s\n\tDriver Version:%u.%u.%u",
         cap.driver, cap.card, cap.bus_info, (cap.version>>16)&0XFF, (cap.version>>8)&0XFF, cap.version&0XFF);

        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) &&
            !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
            LOGE(TAG, "The Current device is not a video capture device, capabilities: %x", cap.capabilities);
            break;
        }
        //judge whether or not to supply the form of video stream
        if(!(cap.capabilities & V4L2_CAP_STREAMING)){
            LOGE(TAG, "The Current device does not support streaming i/o");
            break;
        }
        
        //printf("Capabilities:\n");
        if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE){
            buff_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            LOGI(TAG, "\tV4L2_BUF_TYPE_VIDEO_CAPTURE");
        }else if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE){
            buff_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            LOGI(TAG, "\tV4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE");
        } else {
            break;
        }


        //printf("Support format:\n");
        struct v4l2_fmtdesc fmtdesc; 
        fmtdesc.type  = buff_type;
        fmtdesc.index = 0;
        while(ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) != -1) {
        // fmt
            //printf("\t%d.%s[", fmtdesc.index+1, fmtdesc.description);
            // resolution
            struct v4l2_frmsizeenum frmsize;
            memset(&frmsize, 0, sizeof(frmsize));
            frmsize.pixel_format = fmtdesc.pixelformat;
            frmsize.index = 0;
            while(ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) != -1) {
                //printf("%d-(%ux%u);", frmsize.index+1, frmsize.stepwise.max_width, frmsize.stepwise.max_height);
                frmsize.index++;
            }
            //printf("]\n");
            fmtdesc.index++;
        }

        //设置摄像头参数
        struct v4l2_format vfmt;
        vfmt.type = buff_type;   //设置类型摄像头采集
        vfmt.fmt.pix.width = width;
        vfmt.fmt.pix.height = height;
        vfmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
        //vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_H264;   //根据摄像头设置格式
        //vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;   //根据摄像头设置格式
        //if(video_type == CSI_MIPI_ISP) vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
        switch(video_type) {
            case CSI_MIPI_ISP:vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;break;
            case USB_CAMERA:vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;break;
            default:goto err_handle;

        }
        //vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12; //根据摄像头设置格式V4L2_PIX_FMT_MJPEG
        vfmt.fmt.pix_mp.quantization = V4L2_QUANTIZATION_FULL_RANGE;
        ret = ioctl(fd, VIDIOC_S_FMT, &vfmt);
        if(ret < 0) {
            LOGE(TAG, "2.ioctl: VIDIOC_S_FMT fail");
            break;
        }

        //memset(&vfmt, 0, sizeof(vfmt));

        if(ioctl(fd, VIDIOC_G_FMT, &vfmt) < 0){
            LOGE(TAG,"2.ioctl: VIDIOC_G_FMT fail");
            break;
        }

       
        struct v4l2_streamparm parm;
        memset(&parm, 0, sizeof(struct v4l2_streamparm));   
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.denominator = fps;
        parm.parm.capture.timeperframe.numerator = 1;

        ret = ioctl(fd, VIDIOC_S_PARM, &parm);
        if (ret < 0) {
            LOGE(TAG, "3.ioctl: VIDIOC_S_PARM fail");
            //break;
        }

        memset(&parm, 0, sizeof(struct v4l2_streamparm));      
        if(ret == 0) {       
            parm.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ret = ioctl(fd, VIDIOC_G_PARM, &parm);
            if (ret < 0) {
                LOGE(TAG, "3.ioctl: VIDIOC_G_PARM fail");
                //break;
            }
        }
             
        float result_fps = -1;
        if(parm.parm.capture.timeperframe.denominator != 0  &&
           parm.parm.capture.timeperframe.numerator != 0) {
            result_fps = (parm.parm.capture.timeperframe.denominator) * 1.0 / parm.parm.capture.timeperframe.numerator;
        }

        const char* pp = (const char* )&vfmt.fmt.pix.pixelformat;
        LOGI(TAG, "Current data format information:\n\twidth:%d\n\theight:%d\n\tformat:%c%c%c%c\n\tfps:%.1f",
        vfmt.fmt.pix.width, vfmt.fmt.pix.height, pp[0], pp[1], pp[2], pp[3],result_fps);


        // 申请一个拥有四个缓冲帧的缓冲区
        struct v4l2_requestbuffers req;
        req.count  = BUF_COUNT; //RK的SDK至少需要申请3个缓冲区
        req.type   = buff_type; // 缓冲帧数据格式
        req.memory = V4L2_MEMORY_MMAP; //内存映射
        if(ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
            LOGE(TAG, "3.ioctl: VIDIOC_REQBUFS fail");
            break;
        }

        
        for (unsigned int n_buffers = 0; n_buffers < req.count; ++n_buffers) {
            struct v4l2_buffer mapbuffer;
            struct v4l2_plane planes[FMT_NUM_PLANES] = {0};
            memset(&mapbuffer, 0, sizeof(mapbuffer));
            mapbuffer.index = n_buffers;                //buffer 序号
            mapbuffer.type = buff_type;                   //设置类型摄像头采集
            mapbuffer.memory = V4L2_MEMORY_MMAP;        //内存映射  IO 方式，被应用程序设置

            //mapbuffer.memory = V4L2_MEMORY_USERPTR;
            if(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == mapbuffer.type){
                mapbuffer.m.planes = planes;
                mapbuffer.length = FMT_NUM_PLANES;
            }

            // 查询序号为n_buffers 的缓冲区，得到其起始物理地址和大小
            if (-1 == ioctl (fd, VIDIOC_QUERYBUF, &mapbuffer)){
                LOGE(TAG, "4.ioctl: VIDIOC_QUERYBUF(mmap %d buff from kernal) fail\n", n_buffers);
                goto err_handle;
            }

            // 把通过ioctrl申请的内存地址映射到用户空间
            if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE ==  mapbuffer.type) {
                mipi_map.size[n_buffers] = mapbuffer.m.planes[0].length;
                mipi_map.mptr[n_buffers] = (uint8_t *)mmap(NULL, mapbuffer.m.planes[0].length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mapbuffer.m.planes[0].m.mem_offset);
            } else {
                mipi_map.size[n_buffers] = mapbuffer.length;
                mipi_map.mptr[n_buffers] = (uint8_t *)mmap(NULL, mapbuffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mapbuffer.m.offset);
            }

            if(mipi_map.mptr[n_buffers] == MAP_FAILED) {
                LOGE(TAG, "Fail to mmap[%d]", n_buffers);
                goto err_handle;
            }

             // 将缓冲帧放入队列
            if(ioctl(fd, VIDIOC_QBUF, &mapbuffer) < 0) {   //【VIDIOC_QBUF】把帧放入队列；【VIDIOC_DQBUF】从队列中取出帧。
                LOGE(TAG,"5. ioctl: VIDIOC_QBUF(put %d buff to Queue) faild", n_buffers);   
                goto err_handle;
            }

            LOGI(TAG, "usr_buf[%d] start= 0x%p size=%d", n_buffers, mipi_map.mptr[n_buffers], mipi_map.size[n_buffers]); 
        }


        int type = buff_type;
        if(ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
            LOGE(TAG, "6. ioctl: VIDIOC_STREAMON fail");
            goto err_handle;
        }

        struct vpu_decode decode = {0};
        bo_t dec_bo = {0};
        int dec_fd = -1;
        ret = rga_control_buffer_init(&dec_bo, &dec_fd, width, height, 16);
        if (ret != 0) {
            LOGE(TAG, "rga_control_buffer_init error ret:%d", ret);
            goto err_handle;
        }
        ret = vpu_decode_jpeg_init(&decode, width, height);
        if (ret != 0) {
            LOGE(TAG, "rga_control_buffer_init error ret:%d", ret);
            goto err_handle;
        }

        mipi_camera_t* cam = new mipi_camera_t();
        cam->fd = fd;
        strcpy(cam->dev_path, dev_path);
        cam->mipi_map = mipi_map;
        cam->buff_type = buff_type;
        cam->video_type = video_type;
        cam->out_format = frame_type;
        cam->rotation = rotation;
        cam->out_width = width;
        cam->out_height = height;
        cam->dec_bo = dec_bo;
        cam->dec_fd = dec_fd;
        cam->decode = decode;

        return cam;
    } while(0);

    
    err_handle:
        if(fd > 0) {
            close(fd);
        }

        for(int i = 0; i < BUF_COUNT; i++) {
            LOGI(TAG, "need to unmap [%d] buffer\n", i);
            if(mipi_map.mptr[i] != nullptr) {
                munmap(mipi_map.mptr[i], mipi_map.size[i]); // 断开映射
            }        
        }

    return nullptr;   
}


void release_frame(frame_data_t* p_frame, int* err) {
    if(p_frame != nullptr && p_frame->data != nullptr) {
        free(p_frame->data);
    }
}

int camera_get_frame(void* inst, frame_data_t* p_frame, int* err) {
    int ret = 0;

    if(inst == nullptr || p_frame == nullptr) {
        ret = -1;
        return ret;
    }

    memset(p_frame, 0, sizeof(frame_data_t));

    mipi_camera_t* cam = (mipi_camera_t*)inst;
    int fd = cam->fd;
    const char* dev_path = cam->dev_path;
    if(fd < 0) {
        ret = -2;
        return ret;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;

    int r = select(fd + 1, &fds, NULL, NULL, &tv);

    if (r == -1)
    {
        LOGE(TAG, "select %s failed\n", dev_path);
        ret = -3;
        return ret;
    }

    if (r == 0)
    {
        LOGE(TAG, "select %s timeout\n", dev_path);
        ret = -4;
        return ret;
    }

     //从队列中提取一帧数据
    struct v4l2_buffer buf;
    struct v4l2_plane planes[FMT_NUM_PLANES];
    memset(&buf, 0, sizeof(buf));
    memset(&planes[0], 0, sizeof(planes));

    buf.type = cam->buff_type;
    buf.memory = V4L2_MEMORY_MMAP;
    if(cam->buff_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
         buf.m.planes = planes;
		 buf.length = FMT_NUM_PLANES;
    }

    // 从队列中取出一个缓冲帧
    if(ioctl(fd, VIDIOC_DQBUF, &buf) < 0){
		LOGE(TAG, "ioctl VIDIOC_DQBUF error, get buffer form queue fail !");
        ret = -5;
        return ret;
    }

    uint8_t* mptr = ( uint8_t*)cam->mipi_map.mptr[buf.index];
    uint32_t size = cam->mipi_map.size[buf.index];
    int w = cam->out_width;
    int h = cam->out_height;

    LOGI(TAG, "get frame successful inst:%x :0x%p size:%d(%d)", inst, mptr, size, buf.bytesused);

    if(cam->video_type == CSI_MIPI_ISP) {
        uint32_t out_size = 0;
        p_frame->frame_type = cam->out_format;
        p_frame->width = w;
        p_frame->height = h;

        int out_fmt = 0;//RK_FORMAT_BGR_888;
        switch(cam->out_format) {
            case FRAME_TYPE_NV12:
                out_size = w * h * 3 / 2;
                out_fmt = RK_FORMAT_YCbCr_420_SP;
                break;
            case FRAME_TYPE_NV21:
                out_size = w * h * 3 / 2;
                out_fmt = RK_FORMAT_YCrCb_420_SP;
                break;
            case FRAME_TYPE_RGB888:
                out_size = w * h * 3;
                out_fmt = RK_FORMAT_RGB_888;
                break;
            case FRAME_TYPE_BGR888:
                out_size = w * h * 3;
                out_fmt = RK_FORMAT_BGR_888;
                break;
            default: goto end_handle;
        }

        p_frame->data = malloc(out_size);
        if(!p_frame->data) {
            ret = -6;
            goto end_handle;
        }

        if(out_fmt == RK_FORMAT_YCbCr_420_SP) {
            memcpy(p_frame->data, mptr, size);
            goto end_handle;
        }

        rga_info_t src;
        memset(&src, 0, sizeof(rga_info_t));
        src.fd = -1;
        src.virAddr = mptr;
        src.mmuFlag = 1;
        src.rotation = cam->rotation;
        rga_set_rect(&src.rect, 0, 0, w, h, w, h, RK_FORMAT_YCbCr_420_SP);//NV12

        rga_info_t dst;
        memset(&dst, 0, sizeof(rga_info_t));
        dst.fd = -1;
        dst.virAddr = p_frame->data;
        dst.mmuFlag = 1;
        rga_set_rect(&dst.rect, 0, 0, w, h, w, h, out_fmt);

        if(c_RkRgaBlit(&src, &dst, NULL)) {
             ret = -7;
	    }

    } else if(cam->video_type == USB_CAMERA) {       
        int ret = vpu_decode_jpeg_doing(&cam->decode, mptr, buf.bytesused, cam->dec_fd, cam->dec_bo.ptr);
        if(ret != 0) {
            ret = -8;
            goto end_handle;;
        }
        RgaSURF_FORMAT dec_fmt = (cam->decode.fmt == MPP_FMT_YUV422SP ? RK_FORMAT_YCbCr_422_SP : RK_FORMAT_YCbCr_420_SP);
        //LOGI(TAG, "vpu_decode_jpeg_doing ret:%d dec_fmt=%d(%x)", ret, dec_fmt, dec_fmt); 

        uint32_t out_size = 0;
        p_frame->frame_type = cam->out_format;
        p_frame->width = w;
        p_frame->height = h;

        int out_fmt = 0;//RK_FORMAT_BGR_888;
        switch(cam->out_format) {
            case FRAME_TYPE_NV12:
                out_size = w * h * 3 / 2;
                out_fmt = RK_FORMAT_YCbCr_420_SP;
                break;
            case FRAME_TYPE_NV21:
                out_size = w * h * 3 / 2;
                out_fmt = RK_FORMAT_YCrCb_420_SP;
                break;
            case FRAME_TYPE_RGB888:
                out_size = w * h * 3;
                out_fmt = RK_FORMAT_RGB_888;
                break;
            case FRAME_TYPE_BGR888:
                out_size = w * h * 3;
                out_fmt = RK_FORMAT_BGR_888;
                break;
            default: goto end_handle;
        }

        p_frame->data = malloc(out_size);
        if(!p_frame->data) {
            ret = -6;
            goto end_handle;
        }

        if(out_fmt == RK_FORMAT_YCbCr_420_SP) {
            memcpy(p_frame->data, cam->dec_bo.ptr, out_size);
            goto end_handle;
        }

        rga_info_t src, dst;
        memset(&src, 0, sizeof(rga_info_t));
        src.fd = -1;
        src.virAddr = cam->dec_bo.ptr;
        src.mmuFlag = 1;
        src.rotation = cam->rotation;
        rga_set_rect(&src.rect, 0, 0, w, h, w, h, dec_fmt);

        memset(&dst, 0, sizeof(rga_info_t));
        dst.fd = -1;
        dst.virAddr = p_frame->data;
        dst.mmuFlag = 1;
        rga_set_rect(&dst.rect, 0, 0, w, h, w, h, out_fmt);
        if(c_RkRgaBlit(&src, &dst, NULL)) {
             ret = -7;
	    }
    }

    end_handle:
        ret = ioctl(fd, VIDIOC_QBUF, &buf); // 用完以后把缓存帧放回队列中
        if(ret < 0) {
            LOGE(TAG, "ioctl VIDIOC_QBUF error, put buffer form queue fail !");
            ret = -10;
        }

    return ret;
}


void camera_close(void* inst, int* err) {
    if(inst == nullptr) {
        return ;
    }

    mipi_camera_t* cam = (mipi_camera_t*)inst;

    if(cam->fd < 0) {
        return;
    }
   
    int type = (int) cam->buff_type;
    if(ioctl(cam->fd, VIDIOC_STREAMOFF, &type) < 0) {
        LOGE(TAG, "ioctl: VIDIOC_STREAMOFF fail");
    }
    
    // unmap buff
    for(int i=0; i<BUF_COUNT; i++) {
        LOGI(TAG, "need to unmap [%d] buffer", i);
        if(cam->mipi_map.mptr[i] != nullptr) {
            munmap(cam->mipi_map.mptr[i], cam->mipi_map.size[i]); // 断开映射
        }        
    }
  
    close(cam->fd);
    vpu_decode_jpeg_done(&cam->decode);
    rga_control_buffer_deinit(&cam->dec_bo, cam->dec_fd);
}

static void delay_ms(uint32_t n_ms) {
    struct timeval tv;
    tv.tv_sec = n_ms / 1000;
    tv.tv_usec = n_ms % 1000;
    select(NULL, NULL, NULL, NULL, &tv);
}

#if 0
static char date_buff[32];
static char time_buff[32];
#include <pthread.h>
void* read_thread(void* arg) {
    void* inst = arg;

    int i = 0;
    while(i < 100) {
         frame_data_t frame_data;
         int err;
         int ret = camera_get_frame(inst, &frame_data, &err) ;
         if(ret == 0) {
            release_frame(&frame_data, &err);
         }
         //LOGD(TAG, "camera_get_frame ret:%d", ret);
         delay_ms(10);
         i++;
    }

    return (void*)0;
}


int mainxx() {

    strcpy(date_buff, __DATE__);
    strcpy(time_buff, __TIME__);

    {
        std::vector<std::string> res;
        str_split(date_buff, ' ', res);

        str_split(time_buff, ' ', res);
    }


    LOGI(TAG, "build date:%s %s", date_buff, time_buff);
    #if 0
    {
        struct vpu_decode decode = {0};
        bo_t dec_bo = {0};
        int dec_fd = -1;
        int width = 1280;
        int height = 720;

        rga_control_buffer_init(&dec_bo, &dec_fd, width, height, 16);

        vpu_decode_jpeg_init(&decode, width, height);

        FILE* fp_in = fopen("720.jpg", "rb");
        fseek(fp_in, 0, SEEK_END);
        uint32_t size = ftell(fp_in);
        fseek(fp_in, 0, SEEK_SET);
        uint8_t* buf = new uint8_t[size];
        fread(buf, size, 1, fp_in);
        fclose(fp_in);

        uint8_t* out_buf = new uint8_t[width * height * 3];

        unsigned long long total = 0;
        for(int i = 0; i < 1; i++) {
            unsigned long long start = log_get_timestamp();
            vpu_decode_jpeg_doing(&decode, buf, size, dec_fd, dec_bo.ptr);
            unsigned long long end = log_get_timestamp();

            RgaSURF_FORMAT dec_fmt = (decode.fmt == MPP_FMT_YUV422SP ? RK_FORMAT_YCbCr_422_SP : RK_FORMAT_YCbCr_420_SP);
            
            //printf("dec_fmt=%x RK_FORMAT_YCbCr_422_SP(%x) RK_FORMAT_YCbCr_420_SP(%x)\r\n", dec_fmt, RK_FORMAT_YCbCr_422_SP, RK_FORMAT_YCbCr_420_SP);       
            #if 1
            rga_info_t src, dst;
            int fmt = RK_FORMAT_BGR_888;

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
            #endif
        
            total += end -start;

            delay_ms(5);
        }
       

        printf("cost:%lluus\r\n", (total) / 1);

        FILE* out_fp = fopen("out.bgr", "wb");
        fwrite(out_buf, width * height * 3, 1, out_fp);
        fclose(out_fp);

        err_handle:
        if(buf) delete  buf;
        if(out_buf)delete  out_buf;
        vpu_decode_jpeg_done(&decode);
        rga_control_buffer_deinit(&dec_bo, dec_fd);

       exit(-1);
    }
    #endif


    {
      int err;      
      void* cam_inst = camera_open(HD_USB0, 1280, 720, 15, FRAME_TYPE_BGR888,  FRAME_NOT_ROT_FILP, &err);   

      void* cam_inst1 = camera_open(HD_USB1, 1280, 720, 15, FRAME_TYPE_BGR888,  FRAME_NOT_ROT_FILP, &err);   

      pthread_t th, th1;
      pthread_create(&th, NULL, read_thread, cam_inst);
      pthread_create(&th1, NULL, read_thread, cam_inst1);

      pthread_join(th, NULL);
      pthread_join(th1, NULL);

      camera_close(cam_inst, &err);
      camera_close(cam_inst1, &err);
    }


    return 0;
}
#endif

int strlenx(const char* str, int max_length)
{
  int i = 0;
  while(str[i] && i < 	max_length) i++;
  return i;
}

void str_split(const char* str, const char split, std::vector<std::string>& results)
{
	std::istringstream iss(str);	// 输入流
	std::string token;			// 接收缓冲区

    results.clear();
	while (getline(iss, token, split))	// 以split为分隔符
	{
		results.push_back(token);
	}
}

size_t least_common_multiple(size_t a, size_t b)
{
    if (a == b)
        return a;

    if (a > b)
        return least_common_multiple(b, a);

    size_t lcm = b;
    while (lcm % a != 0)
    {
        lcm += b;
    }

    return lcm;
}

int running_command(const char *command,char *result, int max_length)
{
    FILE *fp = NULL;
    if (!command || !(fp = popen(command,"r"))){
        fprintf(stderr,"%s %s %d was error\n",__FILE__,__func__,__LINE__);
        return -1; 
    }   
    memset(result, 0, max_length);
    fread(result, max_length, 1, fp);
    fclose(fp);
    return strlenx(result, max_length);
}


int get_dev_path(VIDEO_ENUM_INDEX video_index, char  dev_path[64], int* dev_index, VIDEO_TYPE* video_type, int* err) {    
     if(g_video_dev_lists.size() == 0) {
        return -1;
    }

    for(int i = 0; i < g_video_dev_lists.size(); i++) {
        video_dev_t& video_dev = g_video_dev_lists[i];     
        switch(video_dev.video_type) {
            case CSI_MIPI_ISP: 
                if(video_dev.dev_index != video_index)  continue;
                break;        
            case USB_CAMERA:
                if(video_dev.dev_index != (video_index - HD_USB0))  continue;
                break;    
            default:continue;        
        }
        char* p= strcpy(dev_path, video_dev.dev_path);  
        *dev_index = video_dev.dev_index;
        *video_type = video_dev.video_type;
        return strlen(p);
    }

    return -2;
}

void get_video_lists(std::vector<video_dev_t>& video_dev_lists, int* err) {  
    std::lock_guard<std::mutex> lock(mtx);
    int mipi_index = 0;
    int usb_index = 0;
    video_dev_lists.clear();
    for(int index = 0; index < 128; index++) {
        char dev_path[32];
        sprintf(dev_path, "/dev/video%d", index);
    
        struct stat s_buf = {0};
        int ret = stat(dev_path, &s_buf);   
        if(ret < 0 || !S_ISCHR(s_buf.st_mode)) {
            continue;
        }   

        char name_path[256];
        sprintf(name_path, "/sys/class/video4linux/video%d/name", index);
        FILE* fp = fopen(name_path, "r");
        if(fp == nullptr) {
            continue;
        }

        char line[32] = {0};
        fgets(line, 32, fp);
        fclose(fp);

        video_dev_t dev = {0};
        //printf("dev_path:%s name:%s\r\n", dev_path, line);
        if(strncmp(line, "rkisp_mainpath", 14) == 0) {
            strncpy(dev.dev_path, dev_path, sizeof(dev_path));
            dev.video_type = CSI_MIPI_ISP;
            //dev.dev_index = mipi_index++;

        }
        else if(strncmp(line, "USB", 3) == 0) {
            strncpy(dev.dev_path, dev_path, sizeof(dev_path));
            dev.video_type = USB_CAMERA;
            //dev.dev_index = usb_index++;
        } else {
           continue;
        }

        int fd = ::open(dev_path, O_RDWR | O_NONBLOCK, 0);
        if (fd < 0) {
            fprintf(stderr, "open %s failed %d %s\n", dev_path, errno, strerror(errno));
            continue;
        }

        #define obj_clr(v)   memset(&v, 0, sizeof(v))
        struct v4l2_capability caps;
        v4l2_buf_type buf_type;
        __u32 cap_pixelformat;
        __u32 cap_width;
        __u32 cap_height;
        __u32 cap_numerator;
        __u32 cap_denominator;
        int width = 640;
        int height = 480;
        obj_clr(caps);
        obj_clr(buf_type);
        obj_clr(cap_pixelformat);
        obj_clr(cap_width);
        obj_clr(cap_height);
        obj_clr(cap_numerator); 
        obj_clr(cap_denominator);
        {
           
            memset(&caps, 0, sizeof(caps));

            if (ioctl(fd, VIDIOC_QUERYCAP, &caps))
            {
                fprintf(stderr, "%s ioctl VIDIOC_QUERYCAP failed %d %s\n", dev_path, errno, strerror(errno));
                goto LL;
            }

            fprintf(stderr, "devpath = %s\n", dev_path);
            fprintf(stderr, "\tdriver = %s\n", caps.driver);
            fprintf(stderr, "\tcard = %s\n", caps.card);
            fprintf(stderr, "\tbus_info = %s\n", caps.bus_info);
            fprintf(stderr, "\tversion = %x\n", caps.version);
            fprintf(stderr, "\tcapabilities = %x\n", caps.capabilities);
            fprintf(stderr, "\tdevice_caps = %x\n", caps.device_caps);

            if (caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)
            {
                buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            }
            else if (caps.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
            {
                buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            }
            else
            {
                fprintf(stderr, "%s is not V4L2_CAP_VIDEO_CAPTURE or V4L2_CAP_VIDEO_CAPTURE_MPLANE\n", dev_path);
                goto LL;
            }

            if (!(caps.capabilities & V4L2_CAP_STREAMING))
            {
                fprintf(stderr, "%s is not V4L2_CAP_STREAMING\n", dev_path);
                goto LL;
            }
        }

        // enumerate format
        for (int i = 0; ; i++)
        {
            struct v4l2_fmtdesc fmtdesc;
            memset(&fmtdesc, 0, sizeof(fmtdesc));
            fmtdesc.index = i;
            fmtdesc.type = buf_type;
            if (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc))
            {
                if (errno == EINVAL)
                    break;

                fprintf(stderr, "%s ioctl VIDIOC_ENUM_FMT failed %d %s\n", dev_path, errno, strerror(errno));
                goto LL;
            }

            const char* pp = (const char*)&fmtdesc.pixelformat;
            fprintf(stderr, "\tfmt = %s  %x (%c%c%c%c)\n", fmtdesc.description, fmtdesc.pixelformat, pp[0], pp[1], pp[2], pp[3]);
            if (cap_pixelformat == 0)
            {
                cap_pixelformat = fmtdesc.pixelformat;
            }

            // enumerate size
            for (int j = 0; ; j++)
            {
                struct v4l2_frmsizeenum frmsizeenum;
                memset(&frmsizeenum, 0, sizeof(frmsizeenum));
                frmsizeenum.index = j;
                frmsizeenum.pixel_format = fmtdesc.pixelformat;
                if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsizeenum))
                {
                    if (errno == EINVAL)
                        break;

                    fprintf(stderr, "%s ioctl VIDIOC_ENUM_FRAMESIZES failed %d %s\n", dev_path, errno, strerror(errno));
                    goto LL;
                }

                // NOTE
                // cap_width must be a multiple of 16
                // cap_height must be a multiple of 2
                if (frmsizeenum.type == V4L2_FRMSIZE_TYPE_DISCRETE)
                {
                    __u32 w = frmsizeenum.discrete.width;
                    __u32 h = frmsizeenum.discrete.height;
                    fprintf(stderr, "\t\tsize = %d x %d\n", w, h);

                    if (cap_width == 0 || cap_height == 0)
                    {
                        cap_width = w;
                        cap_height = h;
                    }
                }
                if (frmsizeenum.type == V4L2_FRMSIZE_TYPE_CONTINUOUS)
                {
                    __u32 minw = frmsizeenum.stepwise.min_width;
                    __u32 maxw = frmsizeenum.stepwise.max_width;
                    __u32 minh = frmsizeenum.stepwise.min_height;
                    __u32 maxh = frmsizeenum.stepwise.max_height;
                    fprintf(stderr, "\t\tsize = %d x %d  ~  %d x %d\n", minw, minh, maxw, maxh);

                    if (cap_width == 0 || cap_height == 0)
                    {
                        if (width / (float)height > maxw / (float)maxh)
                        {
                            // fatter
                            cap_height = (width * maxh / maxw + 1) / 2 * 2;
                            cap_width = (width + 15) / 16 * 16;
                        }
                        else
                        {
                            // thinner
                            cap_width = (height * maxw / maxh + 15) / 16 * 16;
                            cap_height = (height + 1) / 2 * 2;
                        }

                        if (cap_width < minw || cap_height < minh)
                        {
                            cap_width = minw;
                            cap_height = minh;
                        }
                    }
                }
                if (frmsizeenum.type == V4L2_FRMSIZE_TYPE_STEPWISE)
                {
                    __u32 minw = frmsizeenum.stepwise.min_width;
                    __u32 maxw = frmsizeenum.stepwise.max_width;
                    __u32 sw = frmsizeenum.stepwise.step_width;
                    __u32 minh = frmsizeenum.stepwise.min_height;
                    __u32 maxh = frmsizeenum.stepwise.max_height;
                    __u32 sh = frmsizeenum.stepwise.step_height;
                    fprintf(stderr, "\t\tsize = %d x %d  ~  %d x %d  (+%d +%d)\n", minw, minh, maxw, maxh, sw, sh);

                    sw = least_common_multiple(sw, 16);
                    sh = least_common_multiple(sh, 2);

                    if (cap_width == 0 || cap_height == 0)
                    {
                        if (width / (float)height > maxw / (float)maxh)
                        {
                            // fatter
                            cap_height = (width * maxh / maxw + sh - 1) / sh * sh;
                            cap_width = (width + sw - 1) / sw * sw;
                        }
                        else
                        {
                            // thinner
                            cap_width = (height * maxw / maxh + sw - 1) / sw * sw;
                            cap_height = (height + sh - 1) / sh * sh;
                        }

                        if (cap_width < minw || cap_height < minh)
                        {
                            cap_width = minw;
                            cap_height = minh;
                        }
                    }
                }

                // enumerate fps
                for (int k = 0; ; k++)
                {
                    struct v4l2_frmivalenum frmivalenum;
                    memset(&frmivalenum, 0, sizeof(frmivalenum));
                    frmivalenum.index = k;
                    frmivalenum.pixel_format = fmtdesc.pixelformat;
                    frmivalenum.width = cap_width;
                    frmivalenum.height = cap_height;

                    if (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frmivalenum))
                    {
                        if (errno == EINVAL)
                            break;

                        fprintf(stderr, "%s ioctl VIDIOC_ENUM_FRAMEINTERVALS failed %d %s\n", dev_path, errno, strerror(errno));
                        goto LL;
                    }

                    if (frmivalenum.type == V4L2_FRMIVAL_TYPE_DISCRETE)
                    {
                        __u32 numer = frmivalenum.discrete.numerator;
                        __u32 denom = frmivalenum.discrete.denominator;
                        fprintf(stderr, "\t\t\tfps = %d / %d\n", numer, denom);

                        if (cap_numerator == 0 || cap_denominator == 0)
                        {
                            cap_numerator = numer;
                            cap_denominator = denom;
                        }
                    }
                    if (frmivalenum.type == V4L2_FRMIVAL_TYPE_CONTINUOUS)
                    {
                        __u32 min_numer = frmivalenum.stepwise.min.numerator;
                        __u32 max_numer = frmivalenum.stepwise.max.numerator;
                        __u32 min_denom = frmivalenum.stepwise.min.denominator;
                        __u32 max_denom = frmivalenum.stepwise.max.denominator;
                        fprintf(stderr, "\t\t\tfps = %d / %d  ~   %d / %d\n", min_numer, min_denom, max_numer, max_denom);

                        if (cap_numerator == 0 || cap_denominator == 0)
                        {
                            cap_numerator = std::max(min_numer, max_numer / 2);
                            cap_denominator = std::max(min_denom, max_denom / 2);
                        }
                    }
                    if (frmivalenum.type == V4L2_FRMIVAL_TYPE_STEPWISE)
                    {
                        __u32 min_numer = frmivalenum.stepwise.min.numerator;
                        __u32 max_numer = frmivalenum.stepwise.max.numerator;
                        __u32 snumer = frmivalenum.stepwise.step.numerator;
                        __u32 min_denom = frmivalenum.stepwise.min.denominator;
                        __u32 max_denom = frmivalenum.stepwise.max.denominator;
                        __u32 sdenom = frmivalenum.stepwise.step.denominator;
                        fprintf(stderr, "\t\t\tfps = %d / %d  ~   %d / %d  (+%d +%d)\n", min_numer, min_denom, max_numer, max_denom, snumer, sdenom);

                        if (cap_numerator == 0 || cap_denominator == 0)
                        {
                            cap_numerator = std::max(min_numer, max_numer - max_numer / 2 / snumer * snumer);
                            cap_denominator = std::max(min_denom, max_denom - max_denom / 2 / sdenom * sdenom);
                        }
                    }

                    if (frmivalenum.type != V4L2_FRMIVAL_TYPE_DISCRETE)
                        break;
                }

                if (frmsizeenum.type != V4L2_FRMSIZE_TYPE_DISCRETE)
                    break;
            }
        }

        if (cap_pixelformat == 0 || cap_width == 0 || cap_height == 0)
        {
            fprintf(stderr, "%s no supported pixel format or size\n", dev_path);
            goto LL;
        }
        //dev.caps = caps;   
        if(dev.video_type == CSI_MIPI_ISP)  dev.dev_index = mipi_index++;  
        if(dev.video_type == USB_CAMERA)  dev.dev_index = usb_index++;
        video_dev_lists.push_back(dev);
        LL:
        close(fd);
    }
}

static std::vector<video_dev_t> get_video_lists_init(void) {
    static std::vector<video_dev_t> v_lists;  
    get_video_lists(v_lists, nullptr);
    return v_lists;
}
