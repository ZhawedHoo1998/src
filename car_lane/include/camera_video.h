#ifndef     __CAMERA_VIDEO_H
#define     __CAMERA_VIDEO_H



#define CAMERA_API                                  __attribute__((visibility("default"))) 

enum VIDEO_TYPE{
    OTHER_IF = 0,
    CSI_MIPI_ISP = 1,
    USB_CAMERA = 2,    
};

enum VIDEO_ENUM_INDEX {
    MIPI_CSI0 = 0,
    MIPI_CSI1 = 1,
    MIPI_CSI2 = 2,
    MIPI_CSI3 = 3,
    MIPI_CSI4 = 4,

    HD_USB0 = 32,
    HD_USB1=  33,
    HD_USB2 = 34,
    HD_USB3=  35,
    HD_USB4=  35,
};

enum FRANE_TYPE {
    FRAME_TYPE_NV12 = 0, //YYYYY   UVUV
    FRAME_TYPE_NV21 = 1, //YYYY    VUVU
    FRAME_TYPE_I420 = 2, //YYYY    UUVV
    FRAME_TYPE_YUYV = 4, //YUYV YUYV
    FRAME_TYPE_RGB888  = 5, //RGB RGB
    FRAME_TYPE_BGR888  = 6, //BGR BGR
};


enum FRAME_ROTATION_DIR {
    FRAME_NOT_ROT_FILP = 0x00,
    FRAME_TRANSFORM_FLIP_H = 0x01,
    FRAME_TRANSFORM_FLIP_V = 0x02,
    FRAME_TRANSFORM_ROT_90 = 0x04,
    FRAME_TRANSFORM_ROT_180 = 0x03,
    FRAME_TRANSFORM_ROT_270 = 0x07,
    FRAME_TRANSFORM_FLIP_H_V= 0x08
};

struct video_dev_t {
    char dev_path[32];
    int dev_index;
    VIDEO_TYPE video_type;
};

struct frame_data_t {
    int width;
    int height;
    FRANE_TYPE frame_type;
    void* data;
};

#ifdef __cplusplus
extern "C" {
#endif

CAMERA_API void* camera_open(VIDEO_ENUM_INDEX video_index, 
                            int width, 
                            int height, 
                            int fps, 
                            FRANE_TYPE frame_type, 
                            FRAME_ROTATION_DIR rotation, 
                            int* err);


CAMERA_API int camera_get_frame(void* inst, 
                                frame_data_t* frame_data, 
                                int* err);


CAMERA_API void camera_close(void* inst, int* err) ;

CAMERA_API void release_frame(frame_data_t* p_frame, int* err) ;

#ifdef __cplusplus
}
#endif

#endif