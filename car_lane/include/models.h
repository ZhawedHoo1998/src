#ifndef     __MODELS_H
#define     __MODELS_H


typedef struct rknn_model_t_
{
  unsigned char* models;
  unsigned int size;
} rknn_model_t;


extern  rknn_model_t model_car_Det;
extern  rknn_model_t model_lane_seg;
 

#endif
