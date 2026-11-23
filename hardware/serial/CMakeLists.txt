# 串口底层库（wjwwood/serial）+ 飞镖视觉串口封装
add_library(serial STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/src/serial.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/src/impl/list_ports/list_ports_linux.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/src/impl/unix.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include/serial/serial.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include/serial/v8stdint.h
    ${CMAKE_CURRENT_SOURCE_DIR}/user_serial.cpp
)

set(rt_LIBRARIES rt)
set(pthread_LIBRARIES pthread)

# 设置库的包含目录
target_include_directories(serial PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include/serial
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include/serial/impl
)

# 链接所需库（不再依赖 Eigen：已删除 IMU 四元数相关代码）
target_link_libraries(serial PUBLIC
    rt
    pthread
    rm_utils
    hardware_common
)
