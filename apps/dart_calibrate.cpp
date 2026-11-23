// ============================================================================
// dart_calibrate —— 棋盘格相机内参标定工具（只需要 OpenCV）
//
// 用途：算出 config/hikcamera.yaml 里需要的 fx / cx / image_width。
// 本项目只用到 fx / cx（把像素偏移换算成角度），不需要手眼标定和 PnP。
//
// 输入可以是：
//   1) 一堆图片：./dart_calibrate out/photos/*.png
//   2) 一个目录：./dart_calibrate out/photos
//   3) 一段视频：./dart_calibrate chessboard.mp4 --step 15
//      视频会按 --step 抽帧，不用事先存图。
//
// 输出：终端打印 RMS、fx / fy / cx / cy，以及可直接粘贴到
//       config/hikcamera.yaml 的 calibration 段。
//
// 注意：
//   - 棋盘格参数 --pattern 是「内角点」数量，不是方格数量；
//     9x6 内角点 = 10x7 个方格。
//   - 标定结果只对「同一颗镜头 + 同一分辨率 + 同一对焦位置」有效，
//     换镜头、变焦、改分辨率后必须重标。
//   - 图片必须是原始分辨率，不要用聊天软件/看图软件另存过的图。
// ============================================================================

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef DART_CALIBRATE_WITH_CAMERA
#include "common_def.hpp"
#include "hikcamera.hpp"
#endif

namespace fs = std::filesystem;

namespace
{

constexpr int kMinImages = 10;  // 少于这个数解不稳定

enum class GridType
{
    Auto,
    Chessboard,
    Circles,
};

void printUsage()
{
    std::cout <<
        "用法:\n"
        "  dart_calibrate <图片目录 | 图片/视频文件...> [选项]\n"
        "  dart_calibrate --camera [相机配置] [选项]     # 实时相机，内存采样，不存任何文件\n"
        "\n"
        "选项:\n"
        "  --camera [yaml] 用海康相机实时标定（默认 config/hikcamera.yaml），\n"
        "                  自动采样，结果只打印，不写图片/视频到磁盘\n"
        "  --grid TYPE     标定板类型: auto(默认) / chess(棋盘格) / circles(圆点阵)\n"
        "  --pattern CxR   棋盘格=内角点数，圆点阵=圆点列数x行数\n"
        "                  chess 默认 9x6；circles 例如 11x10\n"
        "  --square  M     方格边长或圆点间距（米），默认 0.025；\n"
        "                  只影响外参，不影响 fx/cx\n"
        "  --step    N     视频抽帧间隔，默认 10（每 10 帧取 1 帧）\n"
        "  --max     N     最多采样多少张有效图像，默认 40\n"
        "  --min-move PX   实时模式：角点至少移动多少像素才收，默认 40\n"
        "  --interval MS   实时模式：两次采样的最小间隔，默认 400ms\n"
        "  --no-window     实时模式不弹预览窗口，纯自动采样\n"
        "  --show          图片/视频模式显示找到角点的画面（需要 GUI）\n"
        "  -h, --help      显示本帮助\n"
        "\n"
        "示例:\n"
        "  ./dart_calibrate out/photos\n"
        "  ./dart_calibrate out/photos/*.png\n"
        "  ./dart_calibrate chessboard.mp4 --step 15 --pattern 9x6\n"
        "  ./dart_calibrate --camera                    # 不存图片，挥动棋盘自动采样\n"
        "  ./dart_calibrate --camera --grid circles --pattern 11x10 --square 0.01\n"
        "                                               # 用 calib.io 圆点板，间距 10mm\n";
}

bool hasImageExtension(const fs::path &path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

bool hasVideoExtension(const fs::path &path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov" || ext == ".flv" || ext == ".m4v";
}

void collectImages(const fs::path &path, std::vector<fs::path> &images)
{
    std::error_code ec;
    if (fs::is_directory(path, ec))
    {
        for (const auto &entry : fs::directory_iterator(path, ec))
        {
            if (entry.is_regular_file(ec) && hasImageExtension(entry.path()))
            {
                images.push_back(entry.path());
            }
        }
    }
    else if (fs::is_regular_file(path, ec) && hasImageExtension(path))
    {
        images.push_back(path);
    }
}

// 棋盘格：优先用 SB 版本（更稳），失败再退回经典算法。
bool findChessboard(const cv::Mat &gray, const cv::Size &pattern, std::vector<cv::Point2f> &corners)
{
    bool ok = false;

#if CV_VERSION_MAJOR >= 4
    ok = cv::findChessboardCornersSB(gray, pattern, corners, cv::CALIB_CB_NORMALIZE_IMAGE);
#endif

    if (!ok)
    {
        ok = cv::findChessboardCorners(gray, pattern, corners,
                                       cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (ok)
        {
            const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-6);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
        }
    }

    return ok;
}

// 圆点网格（calib.io 那种点阵）。pattern 是「列数 x 行数」的圆点数量。
// 先试普通检测，再用带聚类的版本兜底（透视较强时更稳）。
bool findCircles(const cv::Mat &gray, const cv::Size &pattern, std::vector<cv::Point2f> &corners)
{
    if (cv::findCirclesGrid(gray, pattern, corners, cv::CALIB_CB_SYMMETRIC_GRID))
    {
        return true;
    }
    return cv::findCirclesGrid(gray, pattern, corners,
                               cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING);
}

bool findCorners(const cv::Mat &gray, const cv::Size &pattern, GridType grid,
                 std::vector<cv::Point2f> &corners)
{
    switch (grid)
    {
    case GridType::Chessboard:
        return findChessboard(gray, pattern, corners);
    case GridType::Circles:
        return findCircles(gray, pattern, corners);
    case GridType::Auto:
    default:
        return findChessboard(gray, pattern, corners) || findCircles(gray, pattern, corners);
    }
}

void printYaml(const cv::Mat &K, const cv::Mat &dist, int width)
{
    // 本模块读的是 camera_matrix[0] = fx、camera_matrix[2] = cx（行优先）。
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\n把下面这段填进 config/hikcamera.yaml：\n\n";
    std::cout << "calibration:\n";
    std::cout << "  image_width: " << width << "\n";
    std::cout << "  camera_matrix: [" << fx << ", 0, " << cx << ",\n";
    std::cout << "                  0, " << fy << ", " << cy << ",\n";
    std::cout << "                  0, 0, 1]\n";

    std::cout << "  distort_coeffs: [";
    for (int i = 0; i < 5 && i < static_cast<int>(dist.total()); ++i)
    {
        if (i > 0) std::cout << ", ";
        std::cout << dist.at<double>(0, i);
    }
    std::cout << "]\n";
}

} // namespace

int main(int argc, char **argv)
{
    cv::Size    pattern(9, 6);
    GridType    grid   = GridType::Auto;
    double      square = 0.025;
    int         step   = 10;
    int         max_images = 40;
    bool        show   = false;
    bool        live   = false;
    bool        no_window = false;
    float       min_move_px = 40.0f;
    int         sample_interval_ms = 400;
    std::string camera_config = "config/hikcamera.yaml";
    std::vector<fs::path> inputs;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help")
        {
            printUsage();
            return 0;
        }
        else if (arg == "--show")
        {
            show = true;
        }
        else if (arg == "--no-window")
        {
            no_window = true;
        }
        else if (arg == "--min-move" && i + 1 < argc)
        {
            min_move_px = std::max(1.0f, static_cast<float>(std::stod(argv[++i])));
        }
        else if (arg == "--interval" && i + 1 < argc)
        {
            sample_interval_ms = std::max(0, std::stoi(argv[++i]));
        }
        else if (arg == "--camera")
        {
            live = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
            {
                camera_config = argv[++i];
            }
        }
        else if (arg == "--grid" && i + 1 < argc)
        {
            const std::string value = argv[++i];
            if (value == "auto") grid = GridType::Auto;
            else if (value == "chess" || value == "chessboard") grid = GridType::Chessboard;
            else if (value == "circles" || value == "circle") grid = GridType::Circles;
            else
            {
                std::cerr << "--grid 只能是 auto / chess / circles\n";
                return 1;
            }
        }
        else if (arg == "--pattern" && i + 1 < argc)
        {
            const std::string value = argv[++i];
            const auto pos = value.find('x');
            if (pos == std::string::npos)
            {
                std::cerr << "--pattern 格式应为 9x6\n";
                return 1;
            }
            pattern = cv::Size(std::stoi(value.substr(0, pos)), std::stoi(value.substr(pos + 1)));
        }
        else if (arg == "--square" && i + 1 < argc)
        {
            square = std::stod(argv[++i]);
        }
        else if (arg == "--step" && i + 1 < argc)
        {
            step = std::max(1, std::stoi(argv[++i]));
        }
        else if (arg == "--max" && i + 1 < argc)
        {
            max_images = std::max(kMinImages, std::stoi(argv[++i]));
        }
        else if (!arg.empty() && arg[0] == '-')
        {
            std::cerr << "未知参数: " << arg << "\n";
            printUsage();
            return 1;
        }
        else
        {
            inputs.emplace_back(arg);
        }
    }

    if (inputs.empty() && !live)
    {
        printUsage();
        return 1;
    }

    // 棋盘格三维坐标：与 OpenCV 角点顺序一致（行优先，左上角开始）
    std::vector<cv::Point3f> object_points;
    for (int r = 0; r < pattern.height; ++r)
    {
        for (int c = 0; c < pattern.width; ++c)
        {
            object_points.emplace_back(static_cast<float>(c) * static_cast<float>(square),
                                       static_cast<float>(r) * static_cast<float>(square), 0.0f);
        }
    }

    std::vector<std::vector<cv::Point3f>> object_points_all;
    std::vector<std::vector<cv::Point2f>> image_points_all;
    cv::Size image_size;

    int used = 0;
    int seen = 0;

    auto processFrame = [&](const cv::Mat &frame, const std::string &name) -> bool {
        ++seen;
        if (frame.empty()) return false;

        if (image_size.empty())
        {
            image_size = frame.size();
        }
        else if (frame.size() != image_size)
        {
            std::cout << "跳过(尺寸不一致) " << name << "  " << frame.cols << "x" << frame.rows << "\n";
            return false;
        }

        cv::Mat gray;
        if (frame.channels() == 3)
        {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = frame;
        }

        std::vector<cv::Point2f> corners;
        if (!findCorners(gray, pattern, grid, corners))
        {
            std::cout << "未找到棋盘格   " << name << "\n";
            return false;
        }

        object_points_all.push_back(object_points);
        image_points_all.push_back(corners);
        ++used;

        std::cout << "OK            " << name << "  (" << used << ")\n";

        if (show)
        {
            cv::Mat vis;
            if (frame.channels() == 3)
            {
                frame.copyTo(vis);
            }
            else
            {
                cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
            }
            cv::drawChessboardCorners(vis, pattern, corners, true);
            cv::imshow("dart_calibrate", vis);
            cv::waitKey(1);
        }

        return true;
    };

    if (live)
    {
#ifdef DART_CALIBRATE_WITH_CAMERA
        std::cout << "实时标定模式：挥动棋盘格，工具在内存中自动采样，不写任何文件。\n"
                  << "采样 " << max_images << " 张后自动结束，也可以按 q / ESC 提前结束。\n";

        hikcamera::HikCamera camera(camera_config);
        if (!camera.openCamera())
        {
            std::cout << "相机打开失败，等待自动重连……（10 秒内没有图像则退出）\n";
        }

        bool gui_ok = !no_window;
        if (no_window)
        {
            std::cout << "已禁用预览窗口，采满 " << max_images << " 张后自动结束。\n";
        }

        std::vector<cv::Point2f> last_corners;
        auto last_accept = std::chrono::steady_clock::time_point{};
        const auto start_time = std::chrono::steady_clock::now();

        const auto  kSampleInterval = std::chrono::milliseconds(sample_interval_ms);
        const float kMinMovePx      = min_move_px;  // 与上一张相比角点至少移动多少像素才收

        while (used < max_images)
        {
            CameraFrame data = camera.getImage();
            const auto  now  = std::chrono::steady_clock::now();

            if (data.frame.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                if (used == 0 && now - start_time > std::chrono::seconds(10))
                {
                    std::cerr << "10 秒内没有拿到图像，退出。检查相机连接或配置: "
                              << camera_config << "\n";
                    break;
                }
                continue;
            }

            ++seen;

            if (used == 0 && now - start_time > std::chrono::seconds(120))
            {
                std::cerr << "120 秒内没有成功检测到棋盘格，退出。检查 --pattern 是否与实际内角点数量一致。\n";
                break;
            }

            cv::Mat gray;
            if (data.frame.channels() == 3)
            {
                cv::cvtColor(data.frame, gray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                gray = data.frame;
            }

            if (image_size.empty()) image_size = data.frame.size();
            const bool size_ok = (data.frame.size() == image_size);

            std::vector<cv::Point2f> corners;
            const bool found = size_ok && findCorners(gray, pattern, grid, corners);

            bool accept = false;
            if (found)
            {
                if (object_points_all.empty())
                {
                    // 第一张直接收
                    accept = true;
                }
                else if (now - last_accept >= kSampleInterval)
                {
                    // 与上一张相比棋盘位置/姿态要有明显变化，避免同一位置重复采样
                    float max_move = 1e9f;
                    if (last_corners.size() == corners.size())
                    {
                        max_move = 0.0f;
                        for (size_t k = 0; k < corners.size(); ++k)
                        {
                            max_move = std::max(max_move,
                                                static_cast<float>(cv::norm(corners[k] - last_corners[k])));
                        }
                    }
                    accept = (max_move >= kMinMovePx);
                }
            }

            if (accept)
            {
                object_points_all.push_back(object_points);
                image_points_all.push_back(corners);
                last_corners = corners;
                last_accept  = now;
                ++used;

                const cv::Point2f center = corners[corners.size() / 2];
                std::cout << "采样 " << used << "/" << max_images
                          << "   棋盘中心 (" << cvRound(center.x) << ", " << cvRound(center.y) << ")\n";
            }

            if (gui_ok)
            {
                try
                {
                    cv::Mat vis;
                    if (data.frame.channels() == 3)
                    {
                        data.frame.copyTo(vis);
                    }
                    else
                    {
                        cv::cvtColor(data.frame, vis, cv::COLOR_GRAY2BGR);
                    }
                    if (found)
                    {
                        cv::drawChessboardCorners(vis, pattern, corners, true);
                    }

                    char info[160];
                    std::snprintf(info, sizeof(info), "samples %d/%d  %s   q/ESC = finish",
                                  used, max_images, found ? "board OK" : "no board");
                    cv::putText(vis, info, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                                found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
                    cv::imshow("dart_calibrate (live)", vis);

                    const int key = cv::waitKey(1) & 0xFF;
                    if (key == 27 || key == 'q' || key == 'Q') break;
                }
                catch (const cv::Exception &)
                {
                    gui_ok = false;
                    std::cout << "没有可用显示（headless），继续自动采样，采满即结束。\n";
                }
            }
        }

        camera.closeCamera();
        if (gui_ok)
        {
            cv::destroyAllWindows();
        }
        std::cout << "实时采样结束，共 " << used << " 张（没有写入任何图片/视频）。\n";
#else
        std::cerr << "--camera 需要完整模式编译（不要加 -DDART_VISION_OFFLINE=ON），"
                     "并安装海康 MVS SDK。\n";
        return 1;
#endif
    }
    else
    {
        for (const auto &input : inputs)
        {
            if (used >= max_images) break;

        std::error_code ec;
        if (fs::is_regular_file(input, ec) && hasVideoExtension(input))
        {
            cv::VideoCapture cap(input.string());
            if (!cap.isOpened())
            {
                std::cout << "打不开视频: " << input.string() << "\n";
                continue;
            }

            int index = 0;
            cv::Mat frame;
            while (used < max_images && cap.read(frame))
            {
                if (index % step == 0)
                {
                    processFrame(frame, input.filename().string() + "#" + std::to_string(index));
                }
                ++index;
            }
            std::cout << "视频处理完毕: " << input.string() << "  共 " << index << " 帧\n";
        }
        else
        {
            std::vector<fs::path> images;
            collectImages(input, images);
            std::sort(images.begin(), images.end());

            if (images.empty())
            {
                std::cout << "没有找到图片: " << input.string() << "\n";
                continue;
            }

            for (const auto &file : images)
            {
                if (used >= max_images) break;
                cv::Mat img = cv::imread(file.string());
                processFrame(img, file.filename().string());
            }
        }
    }
    }

    if (show)
    {
        cv::destroyAllWindows();
    }

    std::cout << "\n有效图像: " << used << " / 检查过 " << seen << " 张\n";

    if (used < kMinImages)
    {
        std::cerr << "可用图像太少（至少 " << kMinImages << " 张，建议 15~25 张）。\n"
                  << "检查：棋盘格内角点数量是否填对、棋盘是否完整入镜、是否模糊/反光/只拍了画面中央。\n";
        return 1;
    }

    cv::Mat K, dist;
    std::vector<cv::Mat> rvecs, tvecs;

    const double rms = cv::calibrateCamera(object_points_all, image_points_all, image_size,
                                           K, dist, rvecs, tvecs);

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nRMS 重投影误差: " << rms << " px"
              << "   (<0.5 很好, 0.5~1 可用, >1 建议重拍)\n";
    std::cout << "图像尺寸: " << image_size.width << " x " << image_size.height << "\n";
    std::cout << "fx=" << fx << "  fy=" << fy << "  cx=" << cx << "  cy=" << cy << "\n";

    if (std::abs(cx - image_size.width * 0.5) > image_size.width * 0.1)
    {
        std::cout << "提示: cx 偏离画面中心超过 10%，检查是否每张图都完整覆盖了画面各处。\n";
    }

    printYaml(K, dist, image_size.width);
    return 0;
}
