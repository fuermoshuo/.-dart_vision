// ============================================================================
// 离线视频测试工具
//
// 用视频文件代替海康相机，直接调用真实的 DartDetector / DartTrack，
// 逐帧显示：检测框、瞬时圆心、粗瞄准点、图像中心、误差与角度。
// 同时可保存带标注的视频和逐帧 CSV，便于回看和定量评估稳定性。
//
// 注意：本工具不涉及串口、相机驱动和线程池，可在没有海康 SDK 的机器上编译运行。
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "dart_detector.hpp"
#include "dart_track.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct Options
{
    std::string video_path;
    std::string aim_config    = "config/dart_vision.yaml";
    std::string camera_config = "config/hikcamera.yaml";
    std::string save_path;
    std::string csv_path;

    bool   show      = true;
    bool   pace      = true;   // 按视频帧率实时处理（丢失判定依赖真实时间）
    bool   write_csv = true;

    int    start_frame   = 0;
    int    max_frames    = -1;
    int    history_len   = 300;
    double display_scale = 0.0;   // 0 = 自动（最大显示宽度 1600）
    double fps_override  = 0.0;
    double cx_override   = 0.0;   // >0 时覆盖主点
    double fx_override   = 0.0;   // >0 时覆盖焦距
    double gamma         = 1.0;   // 亮度 gamma，>1 变亮
};

void print_usage(const char *exe)
{
    std::cout
        << "用法: " << exe << " <video> [选项]\n"
        << "\n"
        << "用视频文件测试绿色圆检测 + 目标选择/粗瞄准锁定。\n"
        << "\n"
        << "选项:\n"
        << "  --video <path>          视频文件（也可直接作为第一个位置参数）\n"
        << "  --config <path>         检测/追踪参数（默认 config/dart_vision.yaml）\n"
        << "  --camera-config <path>  相机内参（默认 config/hikcamera.yaml）\n"
        << "  --save <path>           保存带标注的视频，如 out.mp4 / out.avi；\n"
        << "                          同时生成 out_mask.mp4（黑白掩码）和\n"
        << "                          out_compare.mp4（左标注/右掩码对比）\n"
        << "  --csv <path>            逐帧数据 CSV（默认 <视频名>_track.csv）\n"
        << "  --no-csv                不写 CSV\n"
        << "  --no-show               不弹窗，只处理与保存（无显示器时用）\n"
        << "  --fast                  不按视频帧率等待，尽可能快\n"
        << "                          注意：丢失判定用真实时间，fast 模式下\n"
        << "                          found / LOST 的时序不可信\n"
        << "  --start <n>             从第 n 帧开始处理\n"
        << "  --frames <n>            最多处理 n 帧\n"
        << "  --fps <v>               覆盖视频帧率（影响节奏和输出视频）\n"
        << "  --cx <px>               覆盖主点 x（默认自动判断是否与视频宽度匹配）\n"
        << "  --fx <px>               覆盖焦距（只影响 yaw 显示，不影响检测/追踪）\n"
        << "  --gamma <v>             亮度 gamma，>1 变亮（默认 1.0，用于对齐相机亮度）\n"
        << "  --scale <v>             窗口显示缩放，默认自动（最大显示宽度 1600）\n"
        << "  --history <n>           底部曲线保留的采样数，默认 300\n"
        << "  --help                  显示本帮助\n"
        << "\n"
        << "快捷键: 空格=暂停/继续  n=单步  s=截图  q/ESC=退出\n";
}

bool parse_options(int argc, char **argv, Options &opt)
{
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        auto value = [&](const char *name) -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "参数 " << name << " 缺少取值\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--video" || arg == "-v") opt.video_path      = value("--video");
        else if (arg == "--config")               opt.aim_config      = value("--config");
        else if (arg == "--camera-config")        opt.camera_config   = value("--camera-config");
        else if (arg == "--save")                 opt.save_path       = value("--save");
        else if (arg == "--csv")                  { opt.csv_path = value("--csv"); opt.write_csv = true; }
        else if (arg == "--no-csv")               opt.write_csv       = false;
        else if (arg == "--no-show")              opt.show            = false;
        else if (arg == "--fast")                 opt.pace            = false;
        else if (arg == "--start")                opt.start_frame     = std::stoi(value("--start"));
        else if (arg == "--frames")               opt.max_frames      = std::stoi(value("--frames"));
        else if (arg == "--fps")                  opt.fps_override    = std::stod(value("--fps"));
        else if (arg == "--cx")                   opt.cx_override     = std::stod(value("--cx"));
        else if (arg == "--fx")                   opt.fx_override     = std::stod(value("--fx"));
        else if (arg == "--gamma")                opt.gamma           = std::stod(value("--gamma"));
        else if (arg == "--scale")                opt.display_scale   = std::stod(value("--scale"));
        else if (arg == "--history")              opt.history_len     = std::max(10, std::stoi(value("--history")));
        else if (!arg.empty() && arg[0] == '-')
        {
            std::cerr << "未知参数: " << arg << "\n";
            return false;
        }
        else
        {
            positional.push_back(arg);
        }
    }

    if (opt.video_path.empty() && !positional.empty())
    {
        opt.video_path = positional.front();
    }
    if (opt.video_path.empty())
    {
        std::cerr << "缺少视频路径，用 --help 查看用法\n";
        return false;
    }
    return true;
}

std::string default_csv_path(const std::string &video_path)
{
    const auto slash = video_path.find_last_of("/\\");
    const auto dot   = video_path.find_last_of('.');

    std::string stem = video_path;
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
    {
        stem = video_path.substr(0, dot);
    }
    return stem + "_track.csv";
}

#ifndef DART_PROJECT_ROOT
#define DART_PROJECT_ROOT "."
#endif

// 相对输出路径统一以项目根目录为基准：
// 从项目根目录运行 --save out/x.mp4 和从 build/ 里运行，都写到 <项目根>/out/x.mp4
std::string resolve_output_path(const std::string &path)
{
    std::filesystem::path p(path);
    if (p.is_relative())
    {
        p = std::filesystem::path(DART_PROJECT_ROOT) / p;
    }
    return p.string();
}

// 输出路径的父目录不存在时自动创建（例如 out/xxx.mp4）
void ensure_parent_dir(const std::string &path)
{
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
    {
        std::filesystem::create_directories(parent, ec);
    }
}

// 由标注视频路径推导附属视频路径：out.mp4 + "_mask" -> out_mask.mp4
std::string with_suffix(const std::string &save_path, const std::string &suffix)
{
    const auto slash = save_path.find_last_of("/\\");
    const auto dot   = save_path.find_last_of('.');

    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
    {
        std::string out = save_path;
        out.insert(dot, suffix);
        return out;
    }
    return save_path + suffix + ".mp4";
}

bool has_suffix(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// ---------------------------------------------------------------------------
// 画面绘制
// ---------------------------------------------------------------------------

struct Sample
{
    bool  valid = false;
    float inst_x = 0.0f;   // 本帧瞬时检测圆心 x
    float ema_x  = 0.0f;   // 本帧瞄准点 x
};

struct Stats
{
    long   frames       = 0;
    long   det_frames   = 0;
    long   found_frames = 0;
    long   cur_gap      = 0;
    long   max_gap      = 0;

    double sum_detect_ms = 0.0;
    double max_detect_ms = 0.0;

    float det_min_x = std::numeric_limits<float>::max();
    float det_max_x = std::numeric_limits<float>::lowest();
    float ema_min_x = std::numeric_limits<float>::max();
    float ema_max_x = std::numeric_limits<float>::lowest();

    double ema_step_sum   = 0.0;
    long   ema_step_count = 0;
    float  prev_ema_x     = 0.0f;
    bool   has_prev_ema   = false;
};

void draw_history_panel(cv::Mat &img, const std::vector<Sample> &history, float cx)
{
    if (img.empty() || history.size() < 2) return;

    const double ui     = std::max(1.0, img.cols / 1600.0);
    const int    margin = static_cast<int>(12 * ui);
    const int    width  = std::min(img.cols - 2 * margin, static_cast<int>(900 * ui));
    const int    height = static_cast<int>(110 * ui);
    const int    x0     = margin;
    const int    y0     = img.rows - margin - height;

    if (width <= 20 || y0 <= 0) return;

    // 半透明背景
    cv::Mat roi = img(cv::Rect(x0, y0, width, height));
    cv::Mat dark(height, width, img.type(), cv::Scalar(20, 20, 20));
    cv::addWeighted(roi, 0.35, dark, 0.65, 0.0, roi);

    auto to_y = [&](float px) {
        const double norm = static_cast<double>(px) / std::max(1, img.cols);
        return y0 + height - 1 - static_cast<int>(std::clamp(norm, 0.0, 1.0) * (height - 1));
    };
    auto to_x = [&](size_t i) {
        const double t = static_cast<double>(i) / static_cast<double>(history.size() - 1);
        return x0 + static_cast<int>(t * (width - 1));
    };

    // 图像中心参考线
    cv::line(img, cv::Point(to_x(0), to_y(cx)), cv::Point(to_x(history.size() - 1), to_y(cx)),
             cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

    // 瞬时检测值（灰）
    for (size_t i = 1; i < history.size(); ++i)
    {
        if (!history[i - 1].valid || !history[i].valid) continue;
        cv::line(img, cv::Point(to_x(i - 1), to_y(history[i - 1].inst_x)),
                 cv::Point(to_x(i), to_y(history[i].inst_x)), cv::Scalar(170, 170, 170), 1, cv::LINE_AA);
    }

    // 瞄准点（绿）
    for (size_t i = 1; i < history.size(); ++i)
    {
        cv::line(img, cv::Point(to_x(i - 1), to_y(history[i - 1].ema_x)),
                 cv::Point(to_x(i), to_y(history[i].ema_x)), cv::Scalar(80, 255, 80), 2, cv::LINE_AA);
    }

    cv::putText(img, "x history: inst(gray) / aim(green) / cx(cyan)",
                cv::Point(x0 + 8, y0 + static_cast<int>(16 * ui)), cv::FONT_HERSHEY_SIMPLEX, 0.45 * ui,
                cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
}

void draw_overlay(cv::Mat &img, const dart_aim::DartTarget &target, const dart_aim::DartAimResult &aim,
                  const dart_aim::DartTrack &track, long frame_index, double video_time, double detect_ms,
                  const std::vector<Sample> &history)
{
    const int    cx            = static_cast<int>(track.cx());
    const float  fx            = track.fx();
    const bool   yaw_available = fx > 10.0f;   // fx=1 是未标定时的默认值
    const double ui            = std::max(1.0, img.cols / 1600.0);   // HUD 随分辨率放大

    // ---- 图像中心 / 死区 ----
    cv::line(img, cv::Point(cx, 0), cv::Point(cx, img.rows - 1), cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    cv::putText(img, "cx", cv::Point(cx + 4, static_cast<int>(18 * ui)), cv::FONT_HERSHEY_SIMPLEX, 0.5 * ui,
                cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

    // ---- 瞬时检测 ----
    if (target.valid)
    {
        const cv::Point c(static_cast<int>(target.center.x), static_cast<int>(target.center.y));
        const int       r = std::max(1, static_cast<int>(target.radius));

        // 外接方框
        cv::rectangle(img, cv::Rect(c.x - r, c.y - r, 2 * r, 2 * r), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        // 外接圆
        cv::circle(img, c, r, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        // 圆心十字
        cv::line(img, cv::Point(c.x - 6, c.y), cv::Point(c.x + 6, c.y), cv::Scalar(0, 0, 255), 2);
        cv::line(img, cv::Point(c.x, c.y - 6), cv::Point(c.x, c.y + 6), cv::Scalar(0, 0, 255), 2);

        char label[128];
        std::snprintf(label, sizeof(label), "(%d,%d) r=%.1f", c.x, c.y, target.radius);
        cv::putText(img, label, cv::Point(c.x + r + 6, c.y - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5 * ui,
                    cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    // ---- 粗瞄准点 ----
    if (aim.found)
    {
        const int ex = static_cast<int>(aim.center_x);
        const int ey = static_cast<int>(aim.center_y);
        cv::line(img, cv::Point(ex, 0), cv::Point(ex, img.rows - 1), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        cv::circle(img, cv::Point(ex, ey), 8, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        cv::putText(img, "aim point", cv::Point(ex + 6, std::max(static_cast<int>(20 * ui), ey - static_cast<int>(12 * ui))),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5 * ui, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
    }

    // ---- HUD ----
    const char *track_state = aim.found ? "TRACK" : (target.valid ? "WAIT" : "LOST");
    const cv::Scalar state_color = aim.found ? cv::Scalar(0, 255, 0)
                                   : target.valid ? cv::Scalar(0, 255, 255)
                                                  : cv::Scalar(0, 0, 255);

    char line1[256];
    std::snprintf(line1, sizeof(line1), "frame %ld   t=%.2fs   detect=%s   %s   %.2fms",
                  frame_index, video_time, target.valid ? "FOUND" : "NONE", track_state, detect_ms);
    cv::putText(img, line1, cv::Point(14, static_cast<int>(26 * ui)), cv::FONT_HERSHEY_SIMPLEX, 0.6 * ui,
                target.valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    char line2[256];
    if (target.valid)
    {
        std::snprintf(line2, sizeof(line2), "target=(%d,%d) r=%.1f area=%.0f circ=%.2f",
                      static_cast<int>(target.center.x), static_cast<int>(target.center.y),
                      target.radius, target.area, target.circularity);
    }
    else
    {
        std::snprintf(line2, sizeof(line2), "target=none");
    }
    cv::putText(img, line2, cv::Point(14, static_cast<int>(52 * ui)), cv::FONT_HERSHEY_SIMPLEX, 0.55 * ui,
                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

    char line3[256];
    if (yaw_available)
    {
        std::snprintf(line3, sizeof(line3), "center=(%.1f,%.1f)  error_x=%+.4f  yaw=%+.2fdeg",
                      aim.center_x, aim.center_y, aim.error_x, aim.target_yaw * 180.0 / kPi);
    }
    else
    {
        std::snprintf(line3, sizeof(line3), "center=(%.1f,%.1f)  error_x=%+.4f  yaw=n/a (fx not calibrated)",
                      aim.center_x, aim.center_y, aim.error_x);
    }
    cv::putText(img, line3, cv::Point(14, static_cast<int>(78 * ui)), cv::FONT_HERSHEY_SIMPLEX, 0.55 * ui,
                state_color, 1, cv::LINE_AA);

    draw_history_panel(img, history, static_cast<float>(cx));
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_options(argc, argv, opt)) return 2;

    // 显式给的相对输出路径统一挂到项目根目录下
    if (!opt.save_path.empty()) opt.save_path = resolve_output_path(opt.save_path);
    if (!opt.csv_path.empty())  opt.csv_path  = resolve_output_path(opt.csv_path);

    if (opt.write_csv && opt.csv_path.empty())
    {
        opt.csv_path = default_csv_path(opt.video_path);
    }

    cv::VideoCapture cap(opt.video_path);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频: " << opt.video_path << "\n";
        return 1;
    }

    const double src_fps = opt.fps_override > 0.0 ? opt.fps_override
                                                  : cap.get(cv::CAP_PROP_FPS);
    const double fps = (src_fps > 1.0 && src_fps < 1000.0) ? src_fps : 30.0;
    const double total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    cv::Mat first;
    if (!cap.read(first) || first.empty())
    {
        std::cerr << "视频为空或第一帧读取失败\n";
        return 1;
    }
    const cv::Size frame_size = first.size();

    // 窗口显示缩放：默认自动限制到 1600 宽，避免 WSLg 把超大窗口整体缩小导致发虚
    double disp_scale = opt.display_scale;
    if (disp_scale <= 0.0)
    {
        disp_scale = (frame_size.width > 1600)
                         ? 1600.0 / static_cast<double>(frame_size.width)
                         : 1.0;
    }
    if (disp_scale < 0.999)
    {
        std::cout << "显示缩放: " << disp_scale << "（原图宽 " << frame_size.width
                  << "，窗口限制 1600；可用 --scale 覆盖）\n";
    }

    // 亮度 gamma：用于把录制的视频亮度对齐到相机画面
    cv::Mat gamma_lut;
    if (std::abs(opt.gamma - 1.0) > 1e-3)
    {
        gamma_lut.create(1, 256, CV_8U);
        for (int i = 0; i < 256; ++i)
        {
            gamma_lut.at<uchar>(i) = cv::saturate_cast<uchar>(
                std::pow(i / 255.0, 1.0 / opt.gamma) * 255.0);
        }
        std::cout << "亮度 gamma: " << opt.gamma << "\n";
    }

    std::cout << "视频: " << opt.video_path << "\n"
              << "分辨率: " << frame_size.width << "x" << frame_size.height
              << "  帧率: " << fps << " fps"
              << "  总帧数: " << (total_frames > 0 ? std::to_string(static_cast<long>(total_frames)) : "未知")
              << "\n";

    // ---- 组件初始化 ----
    dart_aim::DartDetector detector(opt.aim_config);
    dart_aim::DartTrack    tracker(opt.aim_config, opt.camera_config);

    // 离线测试的内参与视频分辨率对齐：
    //   image_width 决定 error_x 归一化、野值阈值和死区，必须用视频实际宽度；
    //   cx 只影响 error_x 和中心参考线，配置明显不匹配时默认用画面中心。
    const float video_width = static_cast<float>(frame_size.width);
    tracker.setImageWidth(video_width);

    if (opt.cx_override > 0.0)
    {
        tracker.setPrincipalPoint(static_cast<float>(opt.cx_override));
    }
    else if (std::abs(tracker.cx() - video_width * 0.5f) > video_width * 0.1f)
    {
        std::cout << "提示: 配置 cx=" << tracker.cx() << " 与视频宽度 "
                  << frame_size.width << " 不匹配，按画面中心 cx=" << video_width * 0.5f
                  << " 计算（可用 --cx 覆盖）\n";
        tracker.setPrincipalPoint(video_width * 0.5f);
    }

    if (opt.fx_override > 0.0)
    {
        tracker.setFocalLength(static_cast<float>(opt.fx_override));
    }

    const float fx = tracker.fx();
    if (fx <= 10.0f)
    {
        std::cout << "提示: 未读到有效 fx（当前 " << fx
                  << "），target_yaw 没有物理意义，请以 error_x 为准。\n";
    }
    else
    {
        std::cout << "内参: fx=" << fx << " cx=" << tracker.cx()
                  << "  锁定所需连续帧数: " << tracker.alignFrames() << "\n";
    }

    // ---- 输出视频 ----
    // 给了 --save 时，除了标注视频，再自动输出一份黑白二值掩码视频
    cv::VideoWriter writer;
    cv::VideoWriter mask_writer;
    cv::VideoWriter compare_writer;
    const cv::Size  half_size(frame_size.width / 2, frame_size.height / 2);
    const cv::Size  compare_size(half_size.width * 2, half_size.height);

    if (!opt.save_path.empty())
    {
        ensure_parent_dir(opt.save_path);

        const int fourcc = has_suffix(opt.save_path, ".avi")
                               ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                               : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(opt.save_path, fourcc, fps, frame_size);
        if (!writer.isOpened())
        {
            std::cerr << "警告: 无法创建输出视频 " << opt.save_path
                      << "，可尝试改用 .avi 扩展名\n";
        }
        else
        {
            std::cout << "标注视频: " << opt.save_path << "\n";
        }

        const std::string mask_path = with_suffix(opt.save_path, "_mask");
        mask_writer.open(mask_path, fourcc, fps, frame_size, true);
        if (!mask_writer.isOpened())
        {
            std::cerr << "警告: 无法创建掩码视频 " << mask_path << "\n";
        }
        else
        {
            std::cout << "掩码视频: " << mask_path << "（白=选中的灯芯）\n";
        }

        const std::string compare_path = with_suffix(opt.save_path, "_compare");
        compare_writer.open(compare_path, fourcc, fps, compare_size, true);
        if (!compare_writer.isOpened())
        {
            std::cerr << "警告: 无法创建对比视频 " << compare_path << "\n";
        }
        else
        {
            std::cout << "对比视频: " << compare_path << "（左=标注，右=灯芯掩码）\n";
        }
    }

    // ---- CSV ----
    std::ofstream csv;
    if (opt.write_csv)
    {
        ensure_parent_dir(opt.csv_path);
        csv.open(opt.csv_path);
        if (!csv.is_open())
        {
            std::cerr << "警告: 无法写入 CSV " << opt.csv_path << "\n";
        }
        else
        {
            csv << "frame,video_time_s,det_valid,det_x,det_y,radius,area,circularity,"
                   "found,error_x,yaw_deg,center_x,center_y,detect_ms\n";
            csv << std::fixed << std::setprecision(3);
            std::cout << "逐帧数据: " << opt.csv_path << "\n";
        }
    }

    const std::string window_name = "dart video test";
    if (opt.show)
    {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        const int disp_w = static_cast<int>(frame_size.width * disp_scale);
        const int disp_h = static_cast<int>(frame_size.height * disp_scale);
        cv::resizeWindow(window_name, std::max(320, disp_w), std::max(240, disp_h));
    }

    // ---- 主循环 ----
    Stats              stats;
    std::vector<Sample> history;
    history.reserve(std::max(10, opt.history_len));

    cv::Mat frame = first;
    long    source_index = 0;
    bool    paused       = false;
    bool    step_once    = false;
    bool    quit         = false;
    const auto t_start   = std::chrono::steady_clock::now();

    for (;;)
    {
        if (frame.empty())
        {
            if (!cap.read(frame)) break;
        }

        if (source_index < opt.start_frame)
        {
            ++source_index;
            frame.release();
            continue;
        }
        if (opt.max_frames > 0 && stats.frames >= opt.max_frames) break;

        if (!gamma_lut.empty())
        {
            cv::LUT(frame, gamma_lut, frame);
        }

        // 实时节奏：让 wall-clock 与视频时间对齐（丢失判定用真实时间）
        // 用已处理帧数计时，--start 跳过前面的帧时不会误等
        if (opt.pace)
        {
            const auto target_time =
                t_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(stats.frames / fps));
            std::this_thread::sleep_until(target_time);
        }

        const double video_time = source_index / fps;

        // 1. 检测（空窗口名 = 不触发 SDL 调试窗口）
        const auto t_det_0 = std::chrono::steady_clock::now();
        const dart_aim::DartTarget target = detector.detect(frame, "");
        const double detect_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_det_0).count();

        // 2. 追踪
        const dart_aim::DartAimResult aim = tracker.update(target);

        // 3. 统计
        stats.frames++;
        stats.sum_detect_ms += detect_ms;
        stats.max_detect_ms = std::max(stats.max_detect_ms, detect_ms);

        if (target.valid)
        {
            stats.det_frames++;
            stats.cur_gap = 0;
            stats.det_min_x = std::min(stats.det_min_x, target.center.x);
            stats.det_max_x = std::max(stats.det_max_x, target.center.x);
        }
        else
        {
            stats.cur_gap++;
            stats.max_gap = std::max(stats.max_gap, stats.cur_gap);
        }

        if (aim.found)
        {
            stats.found_frames++;
            stats.ema_min_x = std::min(stats.ema_min_x, aim.center_x);
            stats.ema_max_x = std::max(stats.ema_max_x, aim.center_x);

            if (target.valid && stats.has_prev_ema)
            {
                stats.ema_step_sum += std::abs(aim.center_x - stats.prev_ema_x);
                stats.ema_step_count++;
            }
            stats.prev_ema_x   = aim.center_x;
            stats.has_prev_ema = true;
        }

        // 4. 曲线采样
        Sample sample;
        sample.valid  = target.valid;
        sample.inst_x = target.center.x;
        sample.ema_x  = aim.found ? aim.center_x : (history.empty() ? tracker.cx() : history.back().ema_x);
        history.push_back(sample);
        if (static_cast<int>(history.size()) > opt.history_len)
        {
            history.erase(history.begin());
        }

        // 5. CSV
        if (csv.is_open())
        {
            csv << source_index << ',' << video_time << ','
                << (target.valid ? 1 : 0) << ',' << target.center.x << ',' << target.center.y << ','
                << target.radius << ',' << target.area << ',' << target.circularity << ','
                << (aim.found ? 1 : 0) << ',' << aim.error_x << ','
                << aim.target_yaw * 180.0 / kPi << ','
                << aim.center_x << ',' << aim.center_y << ',' << detect_ms << '\n';
        }

        // 6. 绘制 / 保存 / 显示
        cv::Mat canvas = frame.clone();
        draw_overlay(canvas, target, aim, tracker, source_index, video_time, detect_ms, history);

        if (writer.isOpened())
        {
            writer.write(canvas);
        }

        if (mask_writer.isOpened() || compare_writer.isOpened())
        {
            cv::Mat mask_bgr;
            cv::cvtColor(detector.lastMask(), mask_bgr, cv::COLOR_GRAY2BGR);

            if (mask_writer.isOpened())
            {
                mask_writer.write(mask_bgr);
            }

            if (compare_writer.isOpened())
            {
                cv::Mat left, right;
                cv::resize(canvas, left, half_size, 0, 0, cv::INTER_AREA);
                cv::resize(mask_bgr, right, half_size, 0, 0, cv::INTER_AREA);
                cv::putText(left, "annotated", cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX,
                            0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                cv::putText(right, "mask: selected core", cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX,
                            0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                cv::Mat compare;
                cv::hconcat(left, right, compare);
                compare_writer.write(compare);
            }
        }

        if (opt.show)
        {
            cv::Mat shown = canvas;
            if (std::abs(disp_scale - 1.0) > 1e-3)
            {
                cv::resize(canvas, shown, cv::Size(), disp_scale, disp_scale,
                           cv::INTER_AREA);
            }
            cv::imshow(window_name, shown);

            auto save_shot = [&]() {
                const std::string shot = "screenshot_" + std::to_string(source_index) + ".png";
                cv::imwrite(shot, canvas);
                std::cout << "已保存截图: " << shot << "\n";
            };

            // 单步：上一轮按了 n，处理完这一帧后重新暂停
            if (step_once)
            {
                step_once = false;
                paused    = true;
            }

            const int key = cv::waitKey(paused ? 30 : 1);
            if (key == 'q' || key == 'Q' || key == 27) quit = true;
            else if (key == ' ')                      paused = !paused;
            else if (key == 's' || key == 'S')        save_shot();

            // 暂停时阻塞在这里，直到空格继续 / n 单步 / q 退出
            while (paused && !quit)
            {
                const int k = cv::waitKey(30);
                if (k == 'q' || k == 'Q' || k == 27) { quit = true; break; }
                if (k == ' ')                        { paused = false; break; }
                if (k == 'n' || k == 'N')            { paused = false; step_once = true; break; }
                if (k == 's' || k == 'S')            save_shot();
            }
        }

        ++source_index;
        frame.release();
        if (quit) break;
    }

    // ---- 收尾 ----
    if (writer.isOpened())         writer.release();
    if (mask_writer.isOpened())    mask_writer.release();
    if (compare_writer.isOpened()) compare_writer.release();
    if (csv.is_open())             csv.close();
    if (opt.show)          cv::destroyAllWindows();

    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    const double det_rate = stats.frames > 0 ? 100.0 * stats.det_frames / stats.frames : 0.0;
    const double found_rate = stats.frames > 0 ? 100.0 * stats.found_frames / stats.frames : 0.0;

    std::cout << "\n================ 测试汇总 ================\n"
              << "处理帧数: " << stats.frames << "  用时 " << std::fixed << std::setprecision(2)
              << elapsed_s << " s (" << (elapsed_s > 0 ? stats.frames / elapsed_s : 0.0) << " fps)\n"
              << "检测成功率: " << std::setprecision(1) << det_rate << "%  ("
              << stats.det_frames << "/" << stats.frames << ")\n"
              << "found 比例:  " << found_rate << "%  ("
              << stats.found_frames << "/" << stats.frames << ")\n"
              << "最长连续未检出: " << stats.max_gap << " 帧 ("
              << std::setprecision(0) << stats.max_gap / fps * 1000.0 << " ms)\n"
              << std::setprecision(2)
              << "检测耗时: 平均 " << (stats.frames > 0 ? stats.sum_detect_ms / stats.frames : 0.0)
              << " ms, 最大 " << stats.max_detect_ms << " ms\n";

    if (stats.det_frames > 0)
    {
        std::cout << "瞬时圆心 x 范围: " << stats.det_min_x << " ~ " << stats.det_max_x
                  << " px (跨度 " << (stats.det_max_x - stats.det_min_x) << ")\n";
    }
    if (stats.found_frames > 0)
    {
        std::cout << "中心 x 范围: " << stats.ema_min_x << " ~ " << stats.ema_max_x
                  << " px (跨度 " << (stats.ema_max_x - stats.ema_min_x) << ")\n";
    }
    if (stats.ema_step_count > 0)
    {
        std::cout << "中心每帧变化: 平均 " << stats.ema_step_sum / stats.ema_step_count << " px\n";
    }
    std::cout << "==========================================\n";

    if (!opt.pace)
    {
        std::cout << "提示: 当前为 --fast 模式，LOST 判定基于 wall-clock，时序不代表真实情况；\n"
                     "      要评估丢失/保持行为，请去掉 --fast 重新跑。\n";
    }

    return 0;
}
