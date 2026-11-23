cmake_minimum_required(VERSION 3.10)
project(dart_vision LANGUAGES CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源文件是 UTF-8（含中文注释）。MSVC 默认按本地代码页解析，
# 会把中文注释后面的换行吞掉，导致莫名其妙的 "不是成员" 报错。
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 离线模式：只编译视频测试工具，不编译海康相机 / 串口 / 主程序
# 适用于没有海康 SDK 的机器（如 Windows 上先用视频验证算法）
option(DART_VISION_OFFLINE "Only build the offline video test tool" OFF)

# 子模块（注意顺序：先定义被依赖的库）
add_subdirectory(rm_utils)            # 日志 / 显示
add_subdirectory(hardware)            # hardware_common（通用定义）
add_subdirectory(dart_aim)            # 绿色圆形检测 + 运动中心估计

# 离线视频测试工具（只需要 OpenCV / yaml-cpp / spdlog / SDL2）
add_executable(dart_video_test
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/dart_video_test.cpp
)
target_link_libraries(dart_video_test PRIVATE dart_aim)
# 相对输出路径统一以项目根目录为基准（例如 --save out/xxx.mp4），
# 这样无论从项目根目录还是 build/ 里运行，都写到同一个 out/
target_compile_definitions(dart_video_test PRIVATE
    DART_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# 棋盘格内参标定工具：输入图片目录 / 图片列表 / 视频，输出 fx/cx 与 YAML 片段。
# 只依赖 OpenCV，在线/离线模式都会编译，WSL / miniPC 上都能直接用。
find_package(OpenCV REQUIRED)
add_executable(dart_calibrate
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/dart_calibrate.cpp
)
target_link_libraries(dart_calibrate PRIVATE ${OpenCV_LIBS})
target_include_directories(dart_calibrate PRIVATE ${OpenCV_INCLUDE_DIRS})

if(NOT DART_VISION_OFFLINE)
    add_subdirectory(hardware/hikcamera)  # 海康相机驱动
    add_subdirectory(hardware/serial)     # 串口
    add_subdirectory(threads)             # 线程封装

    # 完整模式下，标定工具可以 --camera 直接连海康相机实时采样（不落盘）
    target_link_libraries(dart_calibrate PRIVATE hikcamera)
    target_compile_definitions(dart_calibrate PRIVATE DART_CALIBRATE_WITH_CAMERA=1)

    # 主程序
    add_executable(${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/apps/main.cpp
    )

    # 链接子模块库
    target_link_libraries(${PROJECT_NAME} PRIVATE
        rm_utils
        hikcamera
        serial
        hardware_common
        threads
        dart_aim
    )

    # 截图等输出路径以项目根目录为基准，不受启动目录影响
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        DART_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

# 创建软链接到源代码中的 config 文件夹
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${CMAKE_CURRENT_BINARY_DIR}/config
)
