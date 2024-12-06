#include"MyDetect.h"
#include "log.h"

#define   TAG                         __FUNCTION__
#define INVALID_CTX_VALUE                      -1
MyDetect::MyDetect(){
    ctx = INVALID_CTX_VALUE;
}


MyDetect::~MyDetect(){	
    //LOGW(TAG, "ctx:%x", ctx);
	LOGD(TAG, "detect ctx:%x rknn_destroy", ctx);
	if(ctx != INVALID_CTX_VALUE) {
		rknn_destroy(ctx);
	}
	return ;
}

cv::Mat MyDetect::static_resize(cv::Mat& img) {
    float r = min(width / (img.cols*1.0), height / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

void MyDetect::check_ret(int ret, string ret_name)
{
    // 检查ret是否正确并输出，ret_name表示哪一步
    if (ret < 0)
    {
       // cout << ret_name << " error ret=" << ret << endl;
       LOGE(TAG, "ret_name:%s error ret=%d", ret_name.c_str(), ret);
    }
}

void MyDetect::dump_tensor_attr(rknn_tensor_attr *attr)
{
    // 打印模型输入和输出的信息
    LOGI(TAG, "index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

float MyDetect::sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

 float MyDetect::unsigmoid(float y)
{
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

inline int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)  // 量化
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

inline float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)  // 反量化
{
    return ((float)qnt - (float)zp) * scale;
}

void MyDetect::generate_grids_and_stride(const int target_w, const int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides)
{
    /*
        生成网格(直接精确到具体哪个网格)和步幅[没用到]
        target_w：模型输入width
        target_h：模型输入height
        strides：下采样倍数  vector<int> strides = {8, 16, 32};
        grid_stride：vector<GridAndStride> grid_strides;上面定义了GridAndStride结构体
    */
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                // printf("generate_grids_and_stride:  %d\t,%d\t,%d\n", g0, g1, stride);
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}
void MyDetect::generate_grids(const int target_w, const int target_h, vector<int>& strides, vector<GridAndStride>& grid_strides)
{
    /*
        生成网格和步幅
        target_w：模型输入width
        target_h：模型输入height
        strides：下采样倍数  vector<int> strides = {8, 16, 32};
        grid_stride：vector<GridAndStride> grid_strides;上面定义了GridAndStride结构体
    */
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        grid_strides.push_back((GridAndStride){num_grid_w, num_grid_h, stride});
        
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static inline void qsort_descent_inplace(vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static inline void qsort_descent_inplace(vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline void nms_sorted_bboxes(const vector<Object>& faceobjects, vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void MyDetect::generate_yolo_proposals(GridAndStride grid_stride, int8_t *output, int *anchor, float prob_threshold, vector<Object>& objects, int32_t zp, float scale)
{
    const int num_class = classNum;
    const int num_anchors = NUM_ANCHORS;
    const int grid0 = grid_stride.grid0;
    const int grid1 = grid_stride.grid1;
    const int stride = grid_stride.stride;
    int grid_len = grid0 * grid1;
    float thres = unsigmoid(prob_threshold);
    int8_t thres_i8 = qnt_f32_to_affine(thres, zp, scale);

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        // 三个anchor
        for (int i = 0; i < grid1; i++)
        {
            for (int j = 0; j < grid0; j++)
            {
                int8_t box_objectness = output[(anchor_idx * (num_class + 5) + 4) * grid_len + i * grid0 + j];
                if (box_objectness >= thres_i8)
                {
                    const int basic_pos = anchor_idx * (num_class + 5) * grid_len + i * grid0 + j;; // 相当于detection.py中的(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = output + basic_pos;
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    float x_center = (box_x + j) * (float)stride;
                    float y_center = (box_y + i) * (float)stride;
                    float w = box_w * box_w * (float)anchor[anchor_idx * 2];
                    float h = box_h * box_h * (float)anchor[anchor_idx * 2 + 1];
                    float x0 = x_center - w * 0.5f;
                    float y0 = y_center - h * 0.5f;
					
					int8_t maxClassProbs = in_ptr[5 * grid_len];
					int    maxClassId    = 0;
					for (int k = 1; k < classNum; ++k) {
					int8_t prob = in_ptr[(5 + k) * grid_len];
					if (prob > maxClassProbs) {
					  maxClassId    = k;
					  maxClassProbs = prob;
						}
					}
					if (maxClassProbs>thres_i8){
						float prob = sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale))* sigmoid(deqnt_affine_to_f32(box_objectness, zp, scale));
						int label = maxClassId;
					    Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = w;
                        obj.rect.height = h;
                        obj.label = label;
                        obj.prob = prob;
						//std::cout<<"nms before :  x0 "<<x0<<" y0 "<<y0<<"check prob : "<<obj.prob<<" label "<<obj.label<<std::endl;
                        objects.push_back(obj);
					}
					
                    // for (int class_idx = 0; class_idx < num_class; class_idx++)
                    // {
                        // int8_t box_cls_score = output[(anchor_idx * (num_class + 5) + 5) * grid_len + i * grid0 + j + class_idx];
						// float prob = sigmoid(deqnt_affine_to_f32(box_cls_score, zp, scale));
						// if(prob < prob_threshold) continue;
                        ;
                    // }
                }
            }
        }
    }
}
// inline float* blobFromImage(cv::Mat& img){
    // // 归一化操作 将原始图像拉成一维数组了
    // cvtColor(img, img, COLOR_BGR2RGB);

    // float* blob = new float[img.total()*3]; //Mat.total()返回Mat的单个通道中的元素总数，该元素等于Mat.size中值的乘积。对于RGB图像，total() = rows*cols。
    // int channels = 3;
    // int img_h = img.rows;
    // int img_w = img.cols;
    // //vector<float> mean = {0.485, 0.456, 0.406};
    // //vector<float> std = {0.229, 0.224, 0.225};
	// vector<float> mean = {0, 0, 0};
    // vector<float> std = {1,1,1};
    // for (size_t c = 0; c < channels; c++) 
    // {
        // for (size_t  h = 0; h < img_h; h++) 
        // {
            // for (size_t w = 0; w < img_w; w++) 
            // {
                // blob[c * img_w * img_h + h * img_w + w] =
                    // (((float)img.at<Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
            // }
        // }
    // }
    // return blob;
// }


static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        LOGE(TAG, "blob seek failure");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        LOGE(TAG, "buffer malloc failure");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    /* 
    加载rknn模型
    filename : rknn模型文件路径
    model_size : 模型的大小
    */
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        LOGE(TAG, "Open rknn model file %s failed.", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

void MyDetect::decode_outputs(vector<Object>& objects, const int img_w, const int img_h, float bbox_conf_thresh) {
	vector<Object> proposals;
	vector<int> strides = {8, 16, 32};
	vector<GridAndStride> grid_strides;
	vector<int> picked;
	vector<Object> proposal;
	generate_grids(width, height, strides, grid_strides);
	
	//generate_yolo_proposals(grid_strides[0], (int8_t *)outputs[0].buf, (int *)anchor[0], 0.35, proposals, out_zps[0], out_scales[0]);
	
	
	for(int i = 0; i < 3; i++){
		// 三个尺度
		generate_yolo_proposals(grid_strides[i], (int8_t *)outputs[i].buf, (int *)anchor[i], bbox_conf_thresh, proposals, out_zps[i], out_scales[i]);
	}
   // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

	//按照类别进行nms
	for(int cls=0;cls<classNum;cls++){
		proposal.clear();
		for(int i=0;i<proposals.size();i++){
			if(proposals[i].label == cls){
				proposal.push_back(proposals[i]);
			}
		}
		qsort_descent_inplace(proposal);
		picked.clear();
		nms_sorted_bboxes(proposal, picked, NMS_THRESH);
		int count = picked.size();
		//std::cout << "num of boxes: " << count << std::endl;
		for (int i = 0; i < count; i++)
		{
			Object obj;
			obj = proposal[picked[i]];

			// adjust offset to original unpadded
			float x0 = (obj.rect.x) / resize_scale;
			float y0 = (obj.rect.y) / resize_scale;
			float x1 = (obj.rect.x + obj.rect.width) / resize_scale;
			float y1 = (obj.rect.y + obj.rect.height) / resize_scale;

			obj.rect.x = x0;
			obj.rect.y = y0;
			obj.rect.width = x1 - x0;
			obj.rect.height = y1 - y0;
			obj.label = obj.label;
			obj.prob = obj.prob;
			objects.push_back(obj);
		//	std::cout<< "nms after : x0 : "<<x0<<" y0 "<<y0<<" label "<<obj.label<<" prob "<<obj.prob<<std::endl;
		}
	}
	rknn_outputs_release(ctx, 3, outputs);//释放
}


int MyDetect::InitModel(const char* model_name, rknn_core_mask core_mask_){
    // 开始计时
    //gettimeofday(&begin_time, NULL);
    /********************rknn init*********************/
    string ret_name;
    ret_name = "rknn_init"; // 表示rknn的步骤名称
    int model_data_size = 0; // 模型的大小
    unsigned char *model_data = load_model(model_name, &model_data_size); // 加载RKNN模型
    /* 初始化参数flag
    RKNN_FLAG_COLLECT_PERF_MASK：用于运行时查询网络各层时间。
    RKNN_FLAG_MEM_ALLOC_OUTSIDE：用于表示模型输入、输出、权重、中间 tensor 内存全部由用户分配。
    */
    int ret = rknn_init(&ctx, model_data, model_data_size, RKNN_FLAG_COLLECT_PERF_MASK, NULL); // 初始化RKNN
    check_ret(ret, ret_name);
    if(ret != 0) {
         return -1;
    }
    // 设置NPU核心为自动调度
   // rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;core_mask
    rknn_core_mask core_mask = core_mask_;
	ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
         LOGE(TAG, "rknn_init core error ret=%d", ret);
         return -1;
    }
	
	// rknn_core_mask core_mask;
    // if (CORETYPE == 0)
        // core_mask = RKNN_NPU_CORE_0;
    // else if(CORETYPE == 1)
        // core_mask = RKNN_NPU_CORE_1;
    // else if(CORETYPE == 2)
        // core_mask = RKNN_NPU_CORE_2;
	// else if(CORETYPE == 3)
        // core_mask = RKNN_NPU_CORE_0_1_2;
	// else if(CORETYPE == 4)
        // core_mask = RKNN_NPU_CORE_0_1;
	// else
        // core_mask = RKNN_NPU_CORE_AUTO;
    // ret = rknn_set_core_mask(ctx, core_mask);
    // if (ret < 0)
    // {
        // printf("rknn_init core error ret=%d", ret);
        // exit(-1);
    // }

    /********************rknn query*********************/
    // rknn_query 函数能够查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、
    // SDK 版本、内存占用信息、用户自定义字符串等信息。
    // 版本信息
    ret_name = "rknn_query";
    rknn_sdk_version version; // SDK版本信息结构体
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    check_ret(ret, ret_name);
    LOGI(TAG, "sdk api version: %s", version.api_version);
    LOGI(TAG, "driver version: %s", version.drv_version);

    // 输入输出信息
    
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    check_ret(ret, ret_name);
    LOGD(TAG, "model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    // 输入输出Tensor属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs)); // 初始化内存
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i; // 输入的索引位置
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        check_ret(ret, ret_name);
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        check_ret(ret, ret_name);
        dump_tensor_attr(&(output_attrs[i]));
    }

    // 模型输入信息
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        LOGD(TAG,"model is NCHW input fmt");
        channel = input_attrs[0].dims[1];
		height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
       
    }
    else
    {
        LOGD(TAG, "model is NHWC input fmt");
		height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
	
	//std::cout<<"init model : width : "<<width<<" height : "<<height<<std::endl;
    LOGD(TAG, "init model : width : %d height :%d", width, height);
	ret_name = "rknn_inputs_set";
	memset(inputs, 0, sizeof(inputs));

	inputs[0].index = 0;                       // 输入的索引位置
	inputs[0].type = RKNN_TENSOR_UINT8;        // 输入数据类型 采用INT8
	inputs[0].size = width * height * channel; // 这里用的是模型的
	inputs[0].fmt = input_attrs[0].fmt;        // 输入格式，NHWC
	inputs[0].pass_through = 0;                // 为0代表需要进行预处理
	
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) 
	{ 
		outputs[i].index = i; // 输出索引
		outputs[i].is_prealloc = 0; // 由rknn来分配输出的buf，指向输出数据
		outputs[i].want_float = 0;
		out_scales[i] = output_attrs[i].scale;
		out_zps[i] = output_attrs[i].zp; 
		classNum = output_attrs->dims[2]-5;
		//std::cout<<"classNum : "<<classNum<<std::endl;
        LOGD(TAG, "classNum : %d", classNum);
	}

    return 0;
}


int  MyDetect::InitModel(const unsigned char* model_buff, unsigned int size, rknn_core_mask core_mask_) {

     // 开始计时
    //gettimeofday(&begin_time, NULL);
    /********************rknn init*********************/
    string ret_name;
    ret_name = "rknn_init"; // 表示rknn的步骤名称
    //int model_data_size = 0; // 模型的大小
    //unsigned char *model_data = load_model(model_name, &model_data_size); // 加载RKNN模型
    /* 初始化参数flag
    RKNN_FLAG_COLLECT_PERF_MASK：用于运行时查询网络各层时间。
    RKNN_FLAG_MEM_ALLOC_OUTSIDE：用于表示模型输入、输出、权重、中间 tensor 内存全部由用户分配。
    */
    int ret = rknn_init(&ctx, (void* )model_buff, size, RKNN_FLAG_COLLECT_PERF_MASK, NULL); // 初始化RKNN
    check_ret(ret, ret_name);
    if(ret != 0) {
         return -1;
    }
    // 设置NPU核心为自动调度
   // rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;core_mask
    rknn_core_mask core_mask = core_mask_;
	ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
         LOGE(TAG, "rknn_init core error ret=%d", ret);
         return -2;
    }

    /********************rknn query*********************/
    // rknn_query 函数能够查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、
    // SDK 版本、内存占用信息、用户自定义字符串等信息。
    // 版本信息
    ret_name = "rknn_query";
    rknn_sdk_version version; // SDK版本信息结构体
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    check_ret(ret, ret_name);
    LOGD(TAG, "sdk api version: %s", version.api_version);
    LOGD(TAG, "driver version: %s", version.drv_version);

    // 输入输出信息
    
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    check_ret(ret, ret_name);
    LOGD(TAG, "model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    // 输入输出Tensor属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs)); // 初始化内存
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i; // 输入的索引位置
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        check_ret(ret, ret_name);
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        check_ret(ret, ret_name);
        dump_tensor_attr(&(output_attrs[i]));
    }

    // 模型输入信息
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        LOGD(TAG,"model is NCHW input fmt");
        channel = input_attrs[0].dims[1];
		height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
       
    }
    else
    {
        LOGD(TAG, "model is NHWC input fmt");
		height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
	
	//std::cout<<"init model : width : "<<width<<" height : "<<height<<std::endl;
    LOGD(TAG, "init model : width : %d height :%d", width, height);
	ret_name = "rknn_inputs_set";
	memset(inputs, 0, sizeof(inputs));

	inputs[0].index = 0;                       // 输入的索引位置
	inputs[0].type = RKNN_TENSOR_UINT8;        // 输入数据类型 采用INT8
	inputs[0].size = width * height * channel; // 这里用的是模型的
	inputs[0].fmt = input_attrs[0].fmt;        // 输入格式，NHWC
	inputs[0].pass_through = 0;                // 为0代表需要进行预处理
	
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) 
	{ 
		outputs[i].index = i; // 输出索引
		outputs[i].is_prealloc = 0; // 由rknn来分配输出的buf，指向输出数据
		outputs[i].want_float = 0;
		out_scales[i] = output_attrs[i].scale;
		out_zps[i] = output_attrs[i].zp; 
		classNum = output_attrs->dims[2]-5;
		//std::cout<<"classNum : "<<classNum<<std::endl;
        LOGD(TAG, "classNum : %d", classNum);
	}

    return 0;
}


void MyDetect::Detect(cv::Mat img, vector<Object>& objects){
	//cv::Mat bgrImg;
	//cvtColor(img, bgrImg, COLOR_BGR2RGB);
    cv::Mat pr_img = static_resize(img); // resize图像
	resize_scale = min(width / (img.cols*1.0), height / (img.rows*1.0)); // 图像缩放尺度
	inputs[0].buf = (void*)pr_img.data;// 如果进行resize需要改为resize的data

	int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);

	ret = rknn_run(ctx, NULL); // 推理

	ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
	
	 if(ret < 0) {
		 LOGE(TAG, "rknn_outputs_get fail! ret=%d", ret);
		 return;
   }
	return;
}

void RKNNSeg::dump_tensor_attr(rknn_tensor_attr *attr)
{
    // 打印模型输入和输出的信息
    LOGI(TAG, "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


RKNNSeg::RKNNSeg(){
    ctx = INVALID_CTX_VALUE;
    model = nullptr;
}


RKNNSeg::~RKNNSeg(){	
	// Release
    //LOGE(TAG, "ctx:%x", ctx);
	LOGD(TAG, "seg ctx:%x rknn_destroy", ctx);
	if(ctx != INVALID_CTX_VALUE) {
		rknn_destroy(ctx);
	}
	if(model) {
		free(model);
	}
	return ;
}


//初始化模型
int RKNNSeg::InitModel(const char* model_name, rknn_core_mask core_mask_)
{
	/* Create the neural network */
  LOGD(TAG, "Loading mode...");
  int model_data_size = 0;
  model = load_model(model_name, &model_data_size);
  ret = rknn_init(&ctx, model, model_data_size, 0, NULL);
  if (ret < 0) {
    LOGE(TAG,"rknn_init error ret=%d", ret);
    return -1;
  }
  
  rknn_core_mask core_mask = core_mask_;
  ret = rknn_set_core_mask(ctx, core_mask);
  
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    LOGE(TAG,"rknn_init error ret=%d", ret);
    return -1;
  }
  LOGD(TAG, "model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      LOGE(TAG,"rknn_query error ret=%d", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
	LOGD(TAG, "model is NHWC input fmt");
	height = input_attrs[0].dims[1];
	width  = input_attrs[0].dims[2];
	LOGD(TAG, "model input height=%d, width=%d, channel=%d", height, width, channel);
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
	if (ret < 0) {
      LOGE(TAG, "rknn_query error ret=%d", ret);
      return -1;
    }
	//rknn_tensor_attr *out_attr = &(output_attrs[i]);
	//detect_category = out_attr->dims[0]-5;
  }
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index        = 0;
  inputs[0].type         = RKNN_TENSOR_UINT8;
  inputs[0].size         = width * height * channel;
  inputs[0].fmt          = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;// 1 :no preprocess
	
// post process
  for (int i = 0; i < io_num.n_output; ++i) {
    out_scales.push_back(output_attrs[i].scale);
    out_zps.push_back(output_attrs[i].zp);
  }
  
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++) {
    outputs[i].want_float = true;
	outputs[i].is_prealloc = false;
  }
 
  return 0;
}


int RKNNSeg::InitModel(const unsigned char* model_buff, unsigned int size, rknn_core_mask core_mask_)
{
    	/* Create the neural network */
  //LOGD(TAG, "Loading mode...");
  ret = rknn_init(&ctx, (void*)model_buff, size, 0, NULL);
  if (ret < 0) {
    LOGE(TAG,"rknn_init error ret=%d", ret);
    return -1;
  }
  
  rknn_core_mask core_mask = core_mask_;
  ret = rknn_set_core_mask(ctx, core_mask);
  
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    LOGE(TAG,"rknn_init error ret=%d", ret);
    return -1;
  }
  LOGD(TAG, "model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      LOGE(TAG,"rknn_query error ret=%d", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
	LOGD(TAG, "model is NHWC input fmt");
	height = input_attrs[0].dims[1];
	width  = input_attrs[0].dims[2];
	LOGD(TAG, "model input height=%d, width=%d, channel=%d", height, width, channel);
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
	if (ret < 0) {
      LOGE(TAG, "rknn_query error ret=%d", ret);
      return -1;
    }
	//rknn_tensor_attr *out_attr = &(output_attrs[i]);
	//detect_category = out_attr->dims[0]-5;
  }
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index        = 0;
  inputs[0].type         = RKNN_TENSOR_UINT8;
  inputs[0].size         = width * height * channel;
  inputs[0].fmt          = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;// 1 :no preprocess
	
// post process
  for (int i = 0; i < io_num.n_output; ++i) {
    out_scales.push_back(output_attrs[i].scale);
    out_zps.push_back(output_attrs[i].zp);
  }
  
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++) {
    outputs[i].want_float = true;
	outputs[i].is_prealloc = false;
  }
 
  return 0;
}


int RKNNSeg::Detect(cv::Mat &img)
{
  //  cv::imwrite("t1.png",img);
  inputs[0].buf = (void*)img.data;
  //inputs[0].buf = img.data;
 
  rknn_inputs_set(ctx, io_num.n_input, inputs);
  
  ret = rknn_run(ctx, NULL);  
  
  if(ret < 0) {
		LOGE(TAG, "rknn_run fail! ret=%d", ret);
		return -1;
  }
  
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

  if(ret < 0) {
		LOGE(TAG,"rknn_outputs_get fail! ret=%d", ret);
		return -1;
  }
  return ret;
  
}


Mat RKNNSeg::postProcessMulCls()
{
    // std::cout<<"postProcessMulCls  start!"<<std::endl;
        float* ptr_out = (float*)(outputs[0].buf);
	cv::Mat pred = cv::Mat::zeros(height,width,CV_8UC1);
	uchar* uTmpPtr = NULL;
	 // Postprocess
	int INPUT_H = 360;
	int INPUT_W = 640;

	cv::Mat result = cv::Mat::zeros(INPUT_H, INPUT_W, CV_8UC1);
	for (int i = 0; i < INPUT_H * INPUT_W; i++) {
		float fmax = (ptr_out[i]);
		int index = 0;
		for (int j = 1; j < classNum; j++) {
			float f1 = (ptr_out[i + j * INPUT_H * INPUT_W]);
		  if (f1 > fmax) {
			index = j;
			//fmax = prob[i + j * INPUT_H * INPUT_W];
			fmax = f1;
		  }
		}

		if (index  == 1) {
			result.at<uchar>(i) = 255;
		}
	}
	
	rknn_outputs_release(ctx, 1, outputs);//释放
	
    return result;
}

