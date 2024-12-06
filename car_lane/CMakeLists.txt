##############################
cmake_minimum_required(VERSION 3.5)
project(car_lane)

# Find dependencies ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(compressed_image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(fusion_interfaces REQUIRED)

#################################
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fpermissive -fvisibility=hidden -fvisibility-inlines-hidden -ffast-math")

# install target and libraries
#set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
set(RKNN_API_PATH  /home/orangepi/work/rknpu2-1.4.0/runtime/RK3588/Linux/librknn_api)

set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
set(RKNN_DET_LIB ${CMAKE_SOURCE_DIR}/lib/librkdet.so)

include_directories(
     ${RKNN_API_PATH}/include
     /opt/ros/humble/include/

)

#set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/lib")
include_directories( ${CMAKE_SOURCE_DIR})
include_directories( ${CMAKE_SOURCE_DIR}/include)
include_directories( ${CMAKE_SOURCE_DIR}/test)
#include_directories( ${CMAKE_SOURCE_DIR}/camera)

link_directories(${CMAKE_SOURCE_DIR}/model)

#create liblogx.so
add_library(logx SHARED src/log.cc)

#create librkdet.so
add_library(rkdet SHARED src/MyDetect.cc 
                        src/inference.cc)
target_link_libraries(rkdet
     ${RKNN_RT_LIB}
     ${OpenCV_LIBS}
     logx
     nn_models)

#create libcmaera.so
add_library(rkcamera SHARED src/camera_video.cpp
                             src/rga_control.cpp 
                             src/vpu_decode.cpp)
set(link_libs pthread)

#create tracker.so
add_library(tracker src/BYTETracker.cpp
                    src/kalmanFilter.cpp
                    src/lapjv.cpp
                    src/STrack.cpp
                    src/utils.cpp
                    )
target_link_libraries(tracker 
     ${OpenCV_LIBS}
     ${link_libs}
)

add_executable(car_lane_detect_node src/car_lane_seg_node.cpp test/lanex.cpp)

target_link_libraries(car_lane_detect_node 
     logx
     rkcamera 
     rkdet 
     tracker
     ${OpenCV_LIBS}
     ${cv_bridge_LIBRARIES}
)

ament_target_dependencies(car_lane_detect_node 
        rclcpp 
        sensor_msgs 
        cv_bridge
        image_transport
        compressed_image_transport
        cv_bridge
        fusion_interfaces
        )
  
# Install executable ROS2
install(TARGETS car_lane_detect_node #car_detect_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files ROS2
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

# Install configuration files ROS2
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

# Install package.xml ROS2
ament_package()

### make install
install(TARGETS car_lane_detect_node RUNTIME DESTINATION bin)

