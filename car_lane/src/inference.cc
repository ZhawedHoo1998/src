#include  <unistd.h>
#include  <stdlib.h>
#include  <stdio.h>
#include  <time.h>
#include <sys/time.h>
#include "inference.h"
#include "rknn_api.h"
#include  "models.h"
#include "MyDetect.h"
#include "log.h"


#define  OBJ_CLR(obj, size)      memset(obj, 0, size)
#define  TAG                     "ALGO"


struct algo_inst_t {
  int model_status;
  MyDetect* car_det;
  RKNNSeg* lane_seg;
};





static void __attribute__ ((constructor)) image_open_init(void) {
    set_log_level(LOG_VERBOSE_LEVEL);
    LOGI(TAG, "**build Date:%s %s", __DATE__, __TIME__);
}

static void __attribute__ ((destructor)) image_open_deinit(void) {
	
}


static void delay_ms(uint32_t n_ms) {
    struct timeval tv;
    tv.tv_sec = n_ms / 1000;
    tv.tv_usec = n_ms % 1000;
    select(NULL, NULL, NULL, NULL, &tv);
}

void* algo_nn_init(algo_config_t* cfg, ALGO_ERR_CODE* err) {
  algo_inst_t* algo_inst  = nullptr;
  MyDetect* car_det = nullptr; 
  RKNNSeg* lane_seg = nullptr;
  do {

        if(nullptr == cfg) {
          *err = ALGO_ERR_CFG_IS_NONE;
          LOGE(TAG, "config is Null!");
          break;
        }

        if(0 == cfg->algo_opt_mask){
            *err = ALGO_ERR_CFG_OPT_NONE;
            LOGE(TAG, "is not config algo option!(eg:car detect|lane seg)");
            break;
        }

       #define      OPT_ALL_ALGO_ASK                           OPT_MASK(ALGO_CAR_DETECT) | OPT_MASK(ALGO_LANE_SEG)
       int model_status = 0;
       int ret = -1;
       printf("cfg->core_mask[ALGO_CAR_DETECT]=%x NN_CORE_0=%x\r\n", cfg->core_mask[ALGO_CAR_DETECT], NN_CORE_0);
       if(cfg->algo_opt_mask & OPT_MASK(ALGO_CAR_DETECT))  {
            car_det = new MyDetect();       
           if(access(cfg->model_path[ALGO_CAR_DETECT], F_OK | R_OK) != 0) {          
            ret = car_det->InitModel(model_car_Det.models, model_car_Det.size, (rknn_core_mask)cfg->core_mask[ALGO_CAR_DETECT]);        
          }
          else  {
            ret = car_det->InitModel(cfg->model_path[ALGO_CAR_DETECT], (rknn_core_mask)cfg->core_mask[ALGO_CAR_DETECT]);          
          }

          if(ret == 0) {
            model_status |= OPT_MASK(ALGO_CAR_DETECT);
          }
          else {
            LOGE(TAG, "car model init error! ret:%x", ret);
            *err = ALGO_ERR_MODEL_INIT_ERR;
            //if(car_det)  delete car_det;
           // car_det = nullptr;
          }   
      }

      if(cfg->algo_opt_mask & OPT_MASK(ALGO_LANE_SEG))  {
            lane_seg = new RKNNSeg();
            if(access(cfg->model_path[ALGO_LANE_SEG], F_OK | R_OK) != 0) {
                ret = lane_seg->InitModel(model_lane_seg.models, model_lane_seg.size, (rknn_core_mask)cfg->core_mask[ALGO_LANE_SEG]);
            } else {
                ret = lane_seg->InitModel(cfg->model_path[ALGO_LANE_SEG], (rknn_core_mask)cfg->core_mask[ALGO_LANE_SEG]);
            }

            if(ret == 0) {
              model_status |= OPT_MASK(ALGO_LANE_SEG);
            }
            else {
              LOGE(TAG, "lane seg init error! ret:%x", ret);
              *err = ALGO_ERR_MODEL_INIT_ERR;
             // if(lane_seg) lane_seg;
             // lane_seg = nullptr;
            }         
      }

      algo_inst  = new algo_inst_t(); 
      OBJ_CLR(algo_inst, sizeof(algo_inst_t));

      algo_inst->car_det = car_det;
      algo_inst->lane_seg = lane_seg;
      algo_inst->model_status = model_status;
      if(model_status == cfg->algo_opt_mask)  *err = ALGO_ERR_NONE;

    } while(0);

    return algo_inst;
}

void algo_nn_detect_car(void* inst, cv::Mat& img, float conf_threshold, std::vector<Object>& boxes, ALGO_ERR_CODE* err) {


  if(inst == nullptr) {
    LOGE(TAG, "handle is NULL!");
    *err = ALGO_ERR_INST_NONE;
    return;
  }

  if(img.empty()) {
     LOGE(TAG, "img is None!");
     *err = ALGO_ERR_IMG_IS_NONE;
     return;
  }

  algo_inst_t* algo_inst = (algo_inst_t*) inst;
  MyDetect* car_det = algo_inst->car_det;
  if(!(algo_inst->model_status & OPT_MASK(ALGO_CAR_DETECT)) || !car_det) {
     LOGE(TAG, "car model init status err!");
     *err = ALGO_ERR_MODEL_INIT_ERR;
     return;  
  }

  // std::vector<Object> objects;
  car_det->Detect(img, boxes);
  car_det->decode_outputs(boxes, img.cols, img.rows, conf_threshold);

  // boxes.resize(objects.size());
  for(int i = 0; i < boxes.size(); i++) {
    //  注意： 此处的width和height为x2、y2
    cv::Rect_<float>& rect = boxes[i].rect;
    boxes[i].rect.x = rect.x * 1.0 / img.cols;
    boxes[i].rect.y = rect.y * 1.0 / img.cols;
    boxes[i].rect.width = (rect.x + rect.width - 1) * 1.0 / img.cols;
    boxes[i].rect.height = (rect.y + rect.height - 1) * 1.0 / img.rows;
  }

  *err = ALGO_ERR_NONE;
}

ALGO_API void  algo_nn_lane_seg(void* inst, cv::Mat& img,  float conf_threshold, cv::Mat& out_mask, ALGO_ERR_CODE* err) {
  if(inst == nullptr) {
    LOGE(TAG, "handle is NULL!");
    *err = ALGO_ERR_INST_NONE;
    return;
  }

  if(img.empty()) {
     LOGE(TAG, "img is None!");
     *err = ALGO_ERR_IMG_IS_NONE;
     return;
  }

  algo_inst_t* algo_inst = (algo_inst_t*) inst;
  RKNNSeg* lane_seg = algo_inst->lane_seg;
  if(!(algo_inst->model_status & OPT_MASK(ALGO_LANE_SEG)) || !lane_seg) {
     LOGE(TAG, "lane seg model init status err!");
     *err = ALGO_ERR_MODEL_INIT_ERR;
     return;  
  }

  cv::Mat re_img;
  //cv::cvtColor(img, re_img, cv::COLOR_BGR2RGB);
  cv::resize(img,re_img,  cv::Size(640,360));
  int ret = lane_seg->Detect(re_img);
  if(ret != 0) {
    LOGE(TAG, "lane_seg detect error! ret:%d", ret);
    *err = ALGO_ERR_MODEL_INFERENCE_ERR;
    return ;
  }   
  out_mask = lane_seg->postProcessMulCls();

  *err = ALGO_ERR_NONE;
}

void  algo_nn_lane_seg_x(void* inst, cv::Mat& img,  float conf_threshold, FUNC_SEG_RESULTS_CB* pfunc_seg_results_cb, void* arg, ALGO_ERR_CODE* err)
{
  if(inst == nullptr) {
    LOGE(TAG, "handle is NULL!");
    *err = ALGO_ERR_INST_NONE;
    return;
  }

  if(img.empty()) {
     LOGE(TAG, "img is None!");
     *err = ALGO_ERR_IMG_IS_NONE;
     return;
  }

  algo_inst_t* algo_inst = (algo_inst_t*) inst;
  RKNNSeg* lane_seg = algo_inst->lane_seg;
  if(!(algo_inst->model_status & OPT_MASK(ALGO_LANE_SEG)) || !lane_seg) {
     LOGE(TAG, "lane seg model init status err!");
     *err = ALGO_ERR_MODEL_INIT_ERR;
     return;  
  }

  cv::Mat re_img;
  //cv::cvtColor(img, re_img, cv::COLOR_BGR2RGB);
  cv::resize(img,re_img,  cv::Size(640,360));
  int ret = lane_seg->Detect(re_img);
  if(ret != 0) {
    LOGE(TAG, "lane_seg detect error! ret:%d", ret);
    *err = ALGO_ERR_MODEL_INFERENCE_ERR;
    return ;
  }   
	cv::Mat binImg = lane_seg->postProcessMulCls();

  if(pfunc_seg_results_cb) {
    pfunc_seg_results_cb(arg, binImg, img.cols, img.rows);
  }
 
  *err = ALGO_ERR_NONE;
}

void algo_nn_deinit(void* inst, ALGO_ERR_CODE* err)
{
    if(inst == nullptr) {
      *err = ALGO_ERR_INST_NONE;
      return;
   }

   algo_inst_t* algo_inst = (algo_inst_t*) inst;
   MyDetect* car_det = algo_inst->car_det;
   RKNNSeg* lane_seg = algo_inst->lane_seg;
   if(car_det) delete car_det;
   if(lane_seg) delete lane_seg;
   if(algo_inst) delete algo_inst;

   *err = ALGO_ERR_INST_NONE;
}

