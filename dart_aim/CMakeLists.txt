# 飞镖视觉算法库：绿色圆形检测 + 运动中心估计
add_library(dart_aim STATIC
    dart_detector/dart_detector.cpp
    dart_track/dart_track.cpp
)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# 设置库的包含目录
target_include_directories(dart_aim PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/dart_detector
    ${CMAKE_CURRENT_SOURCE_DIR}/dart_track
)

# 链接库
target_link_libraries(dart_aim PUBLIC
    rm_utils
    hardware_common
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
)
