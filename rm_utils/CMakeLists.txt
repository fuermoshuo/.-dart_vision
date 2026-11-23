# 复用工具库：日志 / SDL2 显示（已移除 plotter、EKF、弹道等自瞄专用部分）
add_library(rm_utils STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/mas_log/mas_log.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/display/display.cpp
)

find_package(spdlog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(SDL2_ttf REQUIRED)

# 设置 rm_utils 库的包含目录
target_include_directories(rm_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/mas_log
    ${CMAKE_CURRENT_SOURCE_DIR}/thread_pool
    ${CMAKE_CURRENT_SOURCE_DIR}/SPSCQueue
    ${CMAKE_CURRENT_SOURCE_DIR}/display
    ${OpenCV_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
)

# 链接 rm_utils 库
target_link_libraries(rm_utils PUBLIC
    spdlog::spdlog
    ${OpenCV_LIBS}
    SDL2::SDL2
    SDL2_image::SDL2_image
    SDL2_ttf::SDL2_ttf
)
