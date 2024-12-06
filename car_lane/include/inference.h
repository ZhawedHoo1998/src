#ifndef     __INFERENCE_H
#define     __INFERENCE_H


#include "opencv2/opencv.hpp"
#include <vector>
#include "object.h"

#define ALGO_API                      __attribute__((visibility("default")))     


typedef enum ALGO_ERR_CODE_ {
    ALGO_ERR_NONE = 0x00000000,
    ALGO_ERR_CFG_IS_NONE = 0xFFFFFFFF,
    ALGO_ERR_CFG_OPT_NONE = 0xFFFFFFFE,
    ALGO_ERR_CFG_MODEL_PATH_ERR = 0xFFFFFFFD,


    ALGO_ERR_INST_NONE = 0xFFFFFF80,
    ALGO_ERR_MODEL_INIT_ERR = 0xFFFFFF7f,
    ALGO_ERR_IMG_IS_NONE = 0xFFFFFF7e,
    ALGO_ERR_MODEL_INFERENCE_ERR = 0xFFFFFF7d,

    ALGO_ERR_AUTHORIZATION_FAILED = 0xFFFFFF20,
} ALGO_ERR_CODE;

 typedef enum ALGO_TYPE
{
  ALGO_CAR_DETECT = 0x00,
  ALGO_FACE_DETECT = 0x01,
  ALGO_LANE_SEG = 0x02,
  ALGO_NB
} ALGO_ENUM;



typedef enum _NN_CORE_MASK {
    NN_CORE_AUTO = 0,                                       /* default, run on NPU core randomly. */
    NN_CORE_0 = 1,                                          /* run on NPU core 0. */
    NN_CORE_1 = 2,                                          /* run on NPU core 1. */
    NN_CORE_2 = 4,                                          /* run on NPU core 2. */
    NN_CORE_0_1 = NN_CORE_0 | NN_CORE_1,                   /* run on NPU core 1 and core 2. */
    NN_CORE_0_1_2 = NN_CORE_0_1 | NN_CORE_2,               /* run on NPU core 1 and core 2 and core 3. */
    NN_CORE_UNDEFINED,
} NN_CORE_MASK;

#define      OPT_MASK(x)                                (1 << (x))
 typedef struct _algo_config_t
{
   int algo_opt_mask;
   NN_CORE_MASK core_mask[ALGO_NB];
   char* model_path[ALGO_NB];
} algo_config_t;

typedef struct _object_box_t
{
  float x1;
  float y1;
  float x2;
  float y2;
  int cls_index;
  float prob;
}object_box_t;


typedef struct _point_t 
{
 float x;
 float y;
} point_t;

typedef struct _lane_seg_point_t
{
     std::vector<point_t>  pt;
} lane_seg_point_t;


typedef void FUNC_SEG_RESULTS_CB(void* arg, cv::Mat& seg_mask, int org_width, int org_height);

#ifdef __cplusplus
extern "C" {
#endif

ALGO_API void* algo_nn_init(algo_config_t* cfg, ALGO_ERR_CODE* err);

//RGB
ALGO_API void algo_nn_detect_car(void* inst, cv::Mat& img, float conf_threshold, std::vector<Object>& boxes,ALGO_ERR_CODE* err);
//RGB
ALGO_API void  algo_nn_lane_seg(void* inst, cv::Mat& img,  float conf_threshold, cv::Mat& out_mask, ALGO_ERR_CODE* err);
//RGB
ALGO_API void  algo_nn_lane_seg_x(void* inst, cv::Mat& img,  float conf_threshold, FUNC_SEG_RESULTS_CB* pfunc_seg_results_cb, void* arg, ALGO_ERR_CODE* err);

ALGO_API void algo_nn_deinit(void* inst, ALGO_ERR_CODE* err);


#ifdef __cplusplus
}
#endif

#endif