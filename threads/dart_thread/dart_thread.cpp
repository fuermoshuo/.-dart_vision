#include "dart_thread.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "dart_detector.hpp"
#include "dart_track.hpp"
#include "display.hpp"
#include "mas_log.hpp"
#include "user_serial.hpp"

namespace
{
#ifndef DART_PROJECT_ROOT
#define DART_PROJECT_ROOT "."
#endif

// 保存一帧彩色图（原始 BGR 图，不带调试叠加）到 <项目根>/out/photos
void saveColorPhoto(const cv::Mat &img)
{
    if (img.empty()) return;

    const std::filesystem::path dir = std::filesystem::path(DART_PROJECT_ROOT) / "out" / "photos";
    std::error_code             ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        MAS_LOG_WARN("Screenshot: create {} failed: {}", dir.string(), ec.message());
        return;
    }

    const auto        now  = std::chrono::system_clock::now();
    const std::time_t t    = std::chrono::system_clock::to_time_t(now);
    std::tm          *tm_p = std::localtime(&t);
    if (tm_p == nullptr) return;

    const int ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);

    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", tm_p);

    const std::filesystem::path file =
        dir / (std::string(stamp) + "_" + std::to_string(ms) + ".png");

    if (cv::imwrite(file.string(), img))
    {
        MAS_LOG_INFO("Screenshot saved: {}", file.string());
    }
    else
    {
        MAS_LOG_WARN("Screenshot save failed: {}", file.string());
    }
}
// 独立的瞄准调试窗口：显示状态、误差，HOLD 时画品红色锁定框。
// 手拿相机测试时：把灯转到画面中心附近停住 → 状态变成 HOLD；
// 然后转动相机，绿色检测框跟着灯走，品红锁定框不动，就说明锁定生效了。
void showAimHud(const cv::Mat &img, const dart_aim::DartAimResult &r)
{
    if (img.empty()) return;

    std::vector<rm_utils::DisplayText>  texts;
    std::vector<rm_utils::DisplayPoint> points;
    std::vector<rm_utils::DisplayLine>  lines;

    const char *state_name  = (r.state == dart_aim::AIM_HOLD) ? "HOLD"
                              : (r.state == dart_aim::AIM_ACQUIRE ? "ACQUIRE" : "IDLE");
    const char *target_name = (r.target == dart_aim::TARGET_OUTPOST) ? "OUTPOST"
                              : (r.target == dart_aim::TARGET_BASE ? "BASE" : "NONE");

    rm_utils::DisplayText t1;
    t1.content = std::string("AIM: ") + state_name + "   target: " + target_name;
    t1.x       = 20;
    t1.y       = 20;
    t1.size    = 24;
    t1.color   = (r.state == dart_aim::AIM_HOLD) ? SDL_Color{255, 0, 255, 255}
                                                 : SDL_Color{0, 255, 0, 255};
    texts.push_back(t1);

    char buf[160];
    std::snprintf(buf, sizeof(buf), "found=%d  yaw=%+.2f deg  center=(%.0f,%.0f)",
                  r.found ? 1 : 0, r.target_yaw * 57.29578, r.center_x, r.center_y);

    rm_utils::DisplayText t2;
    t2.content = buf;
    t2.x       = 20;
    t2.y       = 52;
    t2.size    = 20;
    t2.color   = {220, 220, 220, 255};
    texts.push_back(t2);

    if (r.found)
    {
        const int cx = static_cast<int>(r.center_x);
        const int cy = static_cast<int>(r.center_y);

        rm_utils::DisplayPoint p;
        p.x     = cx;
        p.y     = cy;
        p.size  = 6;
        p.color = (r.state == dart_aim::AIM_HOLD) ? SDL_Color{255, 0, 255, 255}
                                                 : SDL_Color{0, 160, 255, 255};
        points.push_back(p);

        const int half = (r.state == dart_aim::AIM_HOLD) ? 16 : 8;

        rm_utils::DisplayLine h;
        h.x1 = cx - half; h.y1 = cy; h.x2 = cx + half; h.y2 = cy;
        h.thickness = 2; h.color = p.color;
        lines.push_back(h);

        rm_utils::DisplayLine v;
        v.x1 = cx; v.y1 = cy - half; v.x2 = cx; v.y2 = cy + half;
        v.thickness = 2; v.color = p.color;
        lines.push_back(v);

        if (r.state == dart_aim::AIM_HOLD)
        {
            const int box = 22;
            const int edges[4][4] = {{cx - box, cy - box, cx + box, cy - box},
                                     {cx + box, cy - box, cx + box, cy + box},
                                     {cx + box, cy + box, cx - box, cy + box},
                                     {cx - box, cy + box, cx - box, cy - box}};
            for (const auto &e : edges)
            {
                rm_utils::DisplayLine l;
                l.x1 = e[0]; l.y1 = e[1]; l.x2 = e[2]; l.y2 = e[3];
                l.thickness = 2;
                l.color     = {255, 0, 255, 255};
                lines.push_back(l);
            }
        }
    }

    rm_utils::Display::getInstance().display_add("DartAim", img, texts, points, lines);
}
} // namespace

namespace threads
{
void dart_thread_func(std::shared_ptr<rigtorp::SPSCQueue<CameraFrame>> camera_buffer)
{
    MAS_LOG_INFO("Dart processing thread started");

    dart_aim::DartDetector detector("config/dart_vision.yaml");
    dart_aim::DartTrack    tracker("config/dart_vision.yaml", "config/hikcamera.yaml");

    UserSerial &serial = UserSerial::getInstance();

    // 只在状态变化时打日志，避免刷屏
    uint8_t  last_aim_state  = 255;
    uint8_t  last_aim_target = 255;
    uint32_t last_cmd_seq    = 0;   // 只处理真正新收到的命令字节，避免重放旧的 3

    while (!g_shutdown)
    {
        bool processed = false;

        if (camera_buffer)
        {
            if (auto *f = camera_buffer->front())
            {
                CameraFrame frame = std::move(*f);
                camera_buffer->pop();

                if (!frame.frame.empty())
                {
                    // 1. 检测所有引导灯（按 x 从小到大排序）
                    auto targets = detector.detectAll(frame.frame, "DartCamera");

                    // 2. 串口下发的目标选择（1=前哨站, 2=基地, 3=保持目标重新粗对准）
                    //    commandSeq 变化才说明是"新命令"；否则 lastCommand 一直是旧值，
                    //    每帧重放会让一次性的 3 反复清掉 HOLD / 对齐计数。
                    const uint32_t cmd_seq = serial.commandSeq();
                    if (cmd_seq != last_cmd_seq)
                    {
                        last_cmd_seq = cmd_seq;
                        tracker.setCommand(serial.lastCommand());
                    }

                    // 2.5 调试键盘命令（1/2/0/3 或 r），方便没接电控时测试
                    const int manual = g_manual_command.exchange(-1, std::memory_order_relaxed);
                    if (manual >= 0)
                    {
                        tracker.setCommand(static_cast<uint8_t>(manual));
                    }

                    // 3. 目标选择 + 粗瞄准/锁定（锁定后不再跟随目标移动）
                    auto result = tracker.update(targets);

                    // 4. 发送给电控：只发角度（度）
                    serial.sendDart(result.found ? 1 : 0, result.target, result.state,
                                    result.target_yaw * 57.29578f);

                    // 状态变化打日志 + 调试窗口
                    if (result.state != last_aim_state || result.target != last_aim_target)
                    {
                        MAS_LOG_INFO("Aim state: target={} state={} (0=idle,1=acquire,2=hold)",
                                     static_cast<int>(result.target), static_cast<int>(result.state));
                        last_aim_state  = result.state;
                        last_aim_target = result.target;
                    }
                    if (tracker.debug())
                    {
                        showAimHud(frame.frame, result);
                    }

                    // 5. 按 s 保存一帧彩色原图到 out/photos
                    //    编码 PNG + 写 /mnt/c 可能几十 ms，放到独立线程，避免阻塞检测循环造成卡顿
                    if (g_save_photo.exchange(false, std::memory_order_relaxed))
                    {
                        cv::Mat shot = frame.frame;   // 浅拷贝，引用计数保证数据不被相机池覆盖
                        try
                        {
                            std::thread([shot]() { saveColorPhoto(shot); }).detach();
                        }
                        catch (const std::exception &e)
                        {
                            MAS_LOG_WARN("Screenshot: spawn saver thread failed ({}), saving synchronously", e.what());
                            saveColorPhoto(shot);
                        }
                    }

                    processed = true;
                }
            }
        }

        if (!processed)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    MAS_LOG_INFO("Dart processing thread exited");
}
} // namespace threads
