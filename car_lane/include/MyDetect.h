#pragma once  //避免头文件重复包含

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "object.h"
#define NMS_THRESH 0.45                       // NMS阈值
#define NUM_ANCHORS 3                        // anchor的数量
#define BBOX_CONF_THRESH 0.01                // 置信度阈值
#define OBJ_NUMB_MAX_SIZE 64                 // 最多检测的目标数
using namespace cv;
using namespace std;
struct GridAndStride
{
    int grid0; // grid_w：模型输入width/stride   width方向的网格数量
    int grid1; // grid_h：模型输入height/stride    height方向的网格数量
    int stride; // 下采样倍数，取值有8、16、32
};


class MyDetect
{
public:
	MyDetect();
	~MyDetect();
public:
	int  InitModel(const char* model_name, rknn_core_mask core_mask_ = RKNN_NPU_CORE_0_1_2);
	int  InitModel(const unsigned char* model_buff, unsigned int size, rknn_core_mask core_mask_ = RKNN_NPU_CORE_0_1_2);
	void Detect(cv::Mat img,vector<Object>& objects);
	void decode_outputs(vector<Object>& objects, const int img_w, const int img_h, float bbox_conf_thresh = BBOX_CONF_THRESH);
private:
	cv::Mat static_resize(cv::Mat& img);
	void check_ret(int ret, string ret_name);
	void dump_tensor_attr(rknn_tensor_attr *attr);
	float sigmoid(float x);
	float unsigmoid(float y);
	void generate_grids_and_stride(const int target_w, const int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides);
	void generate_grids(const int target_w, const int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides);
	void generate_yolo_proposals(GridAndStride grid_stride, int8_t *output, int *anchor, float prob_threshold, vector<Object>& objects, int32_t zp, float scale);

private:
	int classNum = 80;
	int channel = 3;
	int width   = 640;
	int height  = 480;//!图像高度
	rknn_output outputs[3];//!输出层数
	rknn_input inputs[1];
	rknn_input_output_num io_num;
	int anchor[3][6] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
	float resize_scale = 1.0;
	float out_scales[3] = {0, 0, 0}; // 存储scales 和 zp
	int32_t out_zps[3] = {0, 0, 0};
	rknn_context ctx; // 创建rknn_context对象
};

#define OUTNUM 1
class RKNNSeg
{
	public:
		RKNNSeg();
		~RKNNSeg();
	
	public:
		int InitModel(const char* model_name, rknn_core_mask core_mask_ = RKNN_NPU_CORE_0_1_2);
		int InitModel(const unsigned char* model_buff, unsigned int size, rknn_core_mask core_mask_ = RKNN_NPU_CORE_0_1_2);
		int Detect(cv::Mat &img);
		cv::Mat postProcessMulCls();

	public:
		float sigmoid_x(float x);
		void dump_tensor_attr(rknn_tensor_attr *attr);
		unsigned char* model;
		rknn_context   ctx;
		int            ret;

		//post process
		float scale_w = 1.0;
		float scale_h = 1.0;

		std::vector<float>    out_scales;
		std::vector<int32_t>  out_zps;
		rknn_output outputs[OUTNUM];
		rknn_input inputs[1];
		rknn_input_output_num io_num;

		int channel = 3;
		int width   = 640;
		int height  = 360;
		int classNum = 2;

};