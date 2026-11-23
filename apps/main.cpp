#include <atomic>
#include <csignal>
#include <cstddef>
#include <spdlog/common.h>
#include <thread>

#include <SDL2/SDL.h>

#include "BS_thread_pool.hpp"
#include "SPSCQueue.h"
#include "display.hpp"
#include "mas_log.hpp"

#include "dart_thread.hpp"
#include "hikcamera_thread.hpp"
#include "serial_thread.hpp"

// 全局原子变量用于控制所有线程的生命周期
std::atomic<bool> g_shutdown{false};
// 按 s 置位，由飞镖处理线程保存一帧彩色图到 out/photos
std::atomic<bool> g_save_photo{false};
// 调试用：键盘直接下发目标选择，-1 表示没有新命令（正式比赛由电控串口下发）
std::atomic<int> g_manual_command{-1};

// 信号处理函数
void signal_handler(int signum) { g_shutdown = true; }

int main(int argc, char *argv[])
{
    // 注册信号处理器
    std::signal(SIGINT, signal_handler);  // Ctrl+C
    std::signal(SIGTERM, signal_handler); // 终止信号

    // 初始化日志
    rm_utils::MasLog::init("logs/dart_vision.log", spdlog::level::debug, spdlog::level::info);

    MAS_LOG_INFO("Dart Vision System Start");

    try
    {
        // 创建线程池（相机 / 串口 / 飞镖处理 / 显示）
        BS::thread_pool pool(4);
        MAS_LOG_INFO("Thread pool created with {} threads", pool.get_thread_count());

        // 退出方式：
        //   - ESC：SDL 键盘事件，只有窗口本身获得键盘焦点时才会送进来
        //     （这是系统/GUI 的输入路由决定的，不是程序能绕过的）；
        //   - 点窗口的关闭按钮：直接退出。
        rm_utils::Display::getInstance().set_event_callback(
            [](const SDL_Event &event, void *) -> bool {
                if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
                {
                    g_shutdown = true;
                    return true;
                }
                if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_s)
                {
                    g_save_photo = true;
                    return true;
                }
                // 调试：0=待机, 1=前哨站, 2=基地, 3/r=重新对准
                if (event.type == SDL_KEYDOWN)
                {
                    const SDL_Keycode k = event.key.keysym.sym;
                    if (k >= SDLK_0 && k <= SDLK_3)
                    {
                        g_manual_command = static_cast<int>(k - SDLK_0);
                        return true;
                    }
                    if (k == SDLK_r)
                    {
                        g_manual_command = 3;
                        return true;
                    }
                }
                if (event.type == SDL_QUIT ||
                    (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE))
                {
                    g_shutdown = true;
                    return true;
                }
                return false;
            },
            nullptr);

        // 启动显示线程
        rm_utils::Display::getInstance().start(pool);

        // 创建 SPSCQueue，用于相机图像
        size_t camera_buffer_size = 2;
        auto   camera_buffer      = std::make_shared<rigtorp::SPSCQueue<CameraFrame>>(camera_buffer_size);

        // 启动海康相机线程
        auto camera_future = pool.submit_task([camera_buffer_size, camera_buffer]() {
            threads::hikcamera_thread_func("config/hikcamera.yaml", camera_buffer_size, camera_buffer);
        });

        // 启动串口线程
        auto serial_future = pool.submit_task([]() { threads::serial_thread_func(); });

        // 启动飞镖识别/瞄准处理线程
        auto dart_future = pool.submit_task([camera_buffer]() { threads::dart_thread_func(camera_buffer); });

        // 主线程等待关闭信号
        while (!g_shutdown)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        rm_utils::Display::getInstance().shutdown();

        // 等待所有线程退出
        if (camera_future.valid())
        {
            camera_future.get();
        }
        if (serial_future.valid())
        {
            serial_future.get();
        }
        if (dart_future.valid())
        {
            dart_future.get();
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        g_shutdown = true;
        return 1;
    }
}
