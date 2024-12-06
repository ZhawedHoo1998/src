#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "camera_video.h"
#include "log.h"

#define TAG                     "test-camera"

int main() {

    std::string save_dir = "imgs";
    system(("mkdir -p " + save_dir).c_str());
    void* cam_inst = camera_open(MIPI_CSI0, 1280, 720, 60, FRAME_TYPE_RGB888,  FRAME_NOT_ROT_FILP, NULL); 
    if(!cam_inst) exit(-1);
    {
        for(int i = 0; i < 20 ; i++) {
            frame_data_t frame_data; 
            int ret = camera_get_frame(cam_inst, &frame_data, NULL) ;
            if(ret == 0) {
                LOGI(TAG, "camera_get_frame ret=%d", ret);
                {
                    char buf[64];
                    sprintf(buf, "%s/%04d.rgb", save_dir.c_str(), i);
                    FILE* fp = fopen(buf, "wb");
                    if(fp) {
                        fwrite(frame_data.data, frame_data.height * frame_data.width * 3,  1, fp);
                        fclose(fp);
                    }          
                }
                release_frame(&frame_data, NULL);
            }
        }    
    }
    camera_close(cam_inst, NULL); 
    return 0;
}