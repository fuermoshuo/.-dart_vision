#include "dart_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "display.hpp"
#include "mas_log.hpp"
#include "yaml-cpp/yaml.h"

namespace dart_aim
{

namespace
{
constexpr double PI = 3.14159265358979323846;

// 兜底检测（白色亮核 + 绿色光环）相关策略
// 近处目标过曝面积很大，绿色光晕会和被照亮的支架/背景粘在一起，
// 主路径的面积/填充率会把真正的灯筛掉，只剩金属件上的小反光点。
// 灯泡的白核明显比这些反光点大，所以用"核面积"来裁决。
constexpr double kRingOverrideAreaRatio = 2.0;     // 兜底核面积超过主结果这么多倍时，改用兜底结果
constexpr double kFallbackMaxCoreArea    = 50000.0; // 兜底核最大面积（近处过曝的灯可以很大）
constexpr float  kFallbackMaxCoreRadius  = 200.0f;  // 兜底核最大半径
} // namespace

DartDetector::DartDetector(const std::string &config_path)
{
    try
    {
        std::ifstream file(config_path);
        if (!file.good())
        {
            MAS_LOG_WARN("Config file not found: {}, using default values", config_path.c_str());
            return;
        }

        YAML::Node config = YAML::LoadFile(config_path);
        YAML::Node node   = config["dart_vision"]["detector"];

        debug_              = node["debug"].as<bool>(false);
        hue_low_            = node["hue_low"].as<int>(35);
        hue_high_           = node["hue_high"].as<int>(85);
        sat_low_            = node["sat_low"].as<int>(60);
        sat_high_           = node["sat_high"].as<int>(255);
        val_low_            = node["val_low"].as<int>(60);
        val_high_           = node["val_high"].as<int>(255);
        morph_kernel_size_  = node["morph_kernel_size"].as<int>(3);
        min_area_           = node["min_area"].as<double>(150.0);
        max_area_           = node["max_area"].as<double>(20000.0);
        min_circularity_    = node["min_circularity"].as<double>(0.10);
        max_aspect_ratio_   = node["max_aspect_ratio"].as<double>(2.2);
        min_fill_           = node["min_fill"].as<double>(0.30);
        min_mean_sat_       = node["min_mean_sat"].as<double>(120.0);
        min_mean_val_       = node["min_mean_val"].as<double>(95.0);
        core_val_min_       = node["core_val_min"].as<double>(240.0);
        core_max_sat_       = node["core_max_sat"].as<double>(90.0);
        core_min_area_      = node["core_min_area"].as<double>(50.0);
        core_min_radius_    = node["core_min_radius"].as<double>(8.0);
        core_green_ring_min_ = node["core_green_ring_min"].as<double>(0.60);
        big_core_radius_     = node["big_core_radius"].as<double>(25.0);
        big_core_ring_min_   = node["big_core_ring_min"].as<double>(0.35);

        MAS_LOG_INFO("dart_detector yaml loaded successfully");
        MAS_LOG_INFO("detector params: hue=[{},{}] sat_low={} val_low={} area=[{:.0f},{:.0f}] "
                     "big_core_radius={:.1f} big_core_ring_min={:.2f} core_green_ring_min={:.2f}",
                     hue_low_, hue_high_, sat_low_, val_low_, min_area_, max_area_,
                     big_core_radius_, big_core_ring_min_, core_green_ring_min_);
    }
    catch (const std::exception &e)
    {
        MAS_LOG_WARN("Failed to load detector config, using defaults: {}", e.what());
    }
}

std::vector<DartTarget> DartDetector::detectAll(const cv::Mat &bgr_img, const std::string &window_name) noexcept
{
    std::vector<DartTarget> all;
    detectImpl(bgr_img, window_name, &all);
    return all;
}

DartTarget DartDetector::detect(const cv::Mat &bgr_img, const std::string &window_name) noexcept
{
    return detectImpl(bgr_img, window_name, nullptr);
}

DartTarget DartDetector::detectImpl(const cv::Mat &bgr_img, const std::string &window_name,
                                    std::vector<DartTarget> *all_out) noexcept
{
    DartTarget result;

    // 每处理一帧就算一次，窗口里显示的 FPS 就是实际的检测/处理帧率
    updateFps();

    if (bgr_img.empty() || bgr_img.channels() != 3)
    {
        return result;
    }

    // 1. BGR -> HSV，绿色阈值
    cv::cvtColor(bgr_img, hsv_, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_, cv::Scalar(hue_low_, sat_low_, val_low_), cv::Scalar(hue_high_, sat_high_, val_high_), mask_);

    // 2. 形态学去噪：开运算去小噪点，闭运算填小空洞
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_kernel_size_, morph_kernel_size_));
    cv::morphologyEx(mask_, morph_, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(morph_, morph_, cv::MORPH_CLOSE, kernel);

    // 3. 提取轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 4. 过滤并选最可靠目标
    //
    // 只靠"绿色 + 圆度"很容易被绿色垃圾桶、绿色招牌、屏幕里其他绿色物体干扰。
    // 这里再加三个维度的判据：
    //   - fill（填充率）：细长/镂空的绿色区域填充率低，圆的填充率高；
    //   - 平均饱和度/亮度：暗绿色物体（垃圾桶、背景）平均亮度低；
    //   - 面积上限：排除大面积绿色背景。
    // 评分同时偏向"亮、饱和、填充率高、接近正圆"的候选。
    DartTarget              best;
    double                  best_score = -1.0;
    std::vector<DartTarget> main_targets;   // 主路径的所有有效候选（多目标输出用）

    for (const auto &contour : contours)
    {
        if (contour.size() < 5) continue;

        const double area      = cv::contourArea(contour);
        if (area < min_area_ || area > max_area_) continue;

        const double perimeter   = cv::arcLength(contour, true);
        if (perimeter < 1e-6) continue;
        const double circularity = 4.0 * PI * area / (perimeter * perimeter);
        if (circularity < min_circularity_) continue;

        // 外接旋转椭圆的长短轴比
        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        const double    major   = std::max(ellipse.size.width, ellipse.size.height);
        const double    minor   = std::min(ellipse.size.width, ellipse.size.height);
        const double    aspect  = (minor > 1e-6) ? major / minor : 1e6;
        if (aspect > max_aspect_ratio_) continue;

        // 绿色外圈的圆心/半径，只用于填充率过滤和评分
        cv::Point2f outer_center;
        float       outer_radius = 0.0f;
        cv::minEnclosingCircle(contour, outer_center, outer_radius);
        if (outer_radius < 1e-6f) continue;

        const double fill = area / (PI * outer_radius * outer_radius);
        if (fill < min_fill_) continue;

        // 只在包围盒 ROI 上算平均 HSV，避免全图开销
        const cv::Rect roi = cv::boundingRect(contour) & cv::Rect(0, 0, morph_.cols, morph_.rows);
        if (roi.width < 3 || roi.height < 3) continue;

        cv::Mat local_mask = cv::Mat::zeros(roi.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> shifted{contour};
        for (auto &p : shifted[0]) p -= roi.tl();
        cv::drawContours(local_mask, shifted, 0, cv::Scalar(255), cv::FILLED);

        // 平均 S/V 只统计绿色像素，不把中间巨大的白色过曝核算进去。
        // 否则灯越亮、白洞越大，平均饱和度越低，反而会把真灯筛掉。
        cv::Mat green_only;
        cv::bitwise_and(local_mask, morph_(roi), green_only);
        if (cv::countNonZero(green_only) < 10)
        {
            green_only = local_mask;
        }

        const cv::Scalar mean_hsv = cv::mean(hsv_(roi), green_only);
        if (mean_hsv[1] < min_mean_sat_) continue;   // 太"灰"的绿色
        if (mean_hsv[2] < min_mean_val_) continue;   // 太暗的绿色（垃圾桶、背景）

        // 灯芯：绿色外圈内部的高亮核心。
        //
        // 两种典型情况：
        //   - 相机/高曝光：灯芯过曝成白色（S 低），外面是一大片绿色光晕；
        //   - 视频/低曝光：灯芯仍是绿色的高亮小圆（S 高）。
        // 所以先尝试"高 V + 低 S"的严格白核；找不到再退回"只看高 V"，
        // 兼容这两种画面。光晕若是绿色，S 高，会被严格模式排除。
        cv::Mat v_channel, s_channel;
        cv::extractChannel(hsv_(roi), v_channel, 2);
        cv::extractChannel(hsv_(roi), s_channel, 1);

        cv::Mat strict_mask;
        cv::threshold(v_channel, strict_mask, core_val_min_, 255, cv::THRESH_BINARY);
        strict_mask &= local_mask;

        cv::Mat sat_cap;
        cv::threshold(s_channel, sat_cap, core_max_sat_, 255, cv::THRESH_BINARY_INV);
        strict_mask &= sat_cap;

        cv::Mat loose_mask;
        cv::threshold(v_channel, loose_mask, core_val_min_, 255, cv::THRESH_BINARY);
        loose_mask &= local_mask;

        // 灯芯可能被反光分成几块：选"面积够大、离绿色光晕中心近"的那块，
        // 避免选中旁边更亮/更大的白色反光点。
        auto select_core = [&](const cv::Mat &mask, double &out_area, cv::Point2f &out_center,
                               float &out_radius) -> bool {
            std::vector<std::vector<cv::Point>> cs;
            cv::findContours(mask, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            double best_score = -1.0;
            for (const auto &c : cs)
            {
                const double a = cv::contourArea(c);
                if (a < core_min_area_) continue;

                cv::Point2f cc;
                float       cr = 0.0f;
                cv::minEnclosingCircle(c, cc, cr);
                if (cr < core_min_radius_) continue;

                const cv::Point2f cc_img = cc + cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
                const double      dist   = cv::norm(cc_img - outer_center);
                const double      s      = a / (1.0 + dist);

                if (s > best_score)
                {
                    best_score = s;
                    out_area   = a;
                    out_center = cc_img;
                    out_radius = cr;
                }
            }
            return best_score >= 0.0;
        };

        double      core_area   = 0.0;
        cv::Point2f core_center;
        float       core_radius = 0.0f;

        if (!select_core(strict_mask, core_area, core_center, core_radius) &&
            !select_core(loose_mask, core_area, core_center, core_radius))
        {
            continue;
        }

        // 评分：绿环的填充率 + 亮度 + 饱和度，再按长宽比惩罚细长形状
        const double score = fill
                           * std::min(mean_hsv[2] / 200.0, 1.0)
                           * std::min(mean_hsv[1] / 200.0, 1.0)
                           / std::max(1.0, aspect);

        DartTarget cand;
        cand.center      = core_center;   // 用灯芯中心，而不是外圈光晕中心
        cand.radius      = core_radius;   // 用灯芯半径，框更贴合实际光斑
        cand.area        = static_cast<float>(core_area);
        cand.circularity = static_cast<float>(circularity);
        cand.confidence  = static_cast<float>(std::clamp(score, 0.0, 1.0));
        cand.valid       = true;
        main_targets.push_back(cand);

        if (score > best_score)
        {
            best_score = score;
            best       = cand;
        }
    }

    // "白色亮核 + 绿色光环"通道始终跑一次：
    //   - 绿色外圈被打碎、主路径全部失败时，它是唯一的退路；
    //   - 近处大灯过曝时，主路径可能只抓到支架上的小反光点，
    //     而真正的灯芯面积要大得多，这时用兜底结果覆盖主结果。
    std::vector<DartTarget> ring_targets;
    findCoresByGreenRing(ring_targets);

    DartTarget ring_best;
    double     ring_best_score = -1.0;
    for (const auto &r : ring_targets)
    {
        const double s = static_cast<double>(r.confidence) * r.area;
        if (s > ring_best_score)
        {
            ring_best_score = s;
            ring_best       = r;
        }
    }
    if (ring_best.valid && (!best.valid || ring_best.area > best.area * kRingOverrideAreaRatio))
    {
        best = ring_best;
    }

    // 合并两条通道的结果并去重：同一盏灯可能被两条通道各报一次
    std::vector<DartTarget> merged;
    auto                    add_unique = [&merged](const DartTarget &t) {
        for (auto &m : merged)
        {
            const float d = static_cast<float>(cv::norm(m.center - t.center));
            // 2.5 倍核半径内的两个检测视为同一盏灯（灯旁边的金属反光点常常挨得很近），
            // 保留面积更大的那个，避免把一个灯拆成两个目标导致左右身份判错。
            if (d < std::max(8.0f, 2.5f * std::max(m.radius, t.radius)))
            {
                if (t.area > m.area) m = t;
                return;
            }
        }
        merged.push_back(t);
    };
    for (const auto &t : main_targets) add_unique(t);
    for (const auto &t : ring_targets) add_unique(t);
    std::sort(merged.begin(), merged.end(),
              [](const DartTarget &a, const DartTarget &b) { return a.center.x < b.center.x; });

    // 多目标模式：把合并后的列表交给调用方
    if (all_out != nullptr)
    {
        *all_out = merged;
    }

    // 黑白掩码：单目标模式只画 best；多目标模式画所有灯芯
    if (detection_mask_.size() != morph_.size())
    {
        detection_mask_ = cv::Mat::zeros(morph_.size(), CV_8UC1);
    }
    detection_mask_.setTo(0);
    if (all_out != nullptr)
    {
        for (const auto &t : merged)
        {
            const cv::Point c(static_cast<int>(t.center.x), static_cast<int>(t.center.y));
            cv::circle(detection_mask_, c, std::max(1, static_cast<int>(t.radius)),
                       cv::Scalar(255), cv::FILLED);
        }
    }
    else if (best.valid)
    {
        const cv::Point c(static_cast<int>(best.center.x), static_cast<int>(best.center.y));
        cv::circle(detection_mask_, c, std::max(1, static_cast<int>(best.radius)),
                   cv::Scalar(255), cv::FILLED);
    }

    result = best;

    if (debug_)
    {
        std::vector<DartTarget> display_targets;
        if (all_out != nullptr)
        {
            display_targets = merged;
        }
        else if (best.valid)
        {
            display_targets.push_back(best);
        }
        showResult(bgr_img, detection_mask_, display_targets, window_name);
    }

    return result;
}

void DartDetector::updateFps() noexcept
{
    using clock = std::chrono::steady_clock;

    const auto now                   = clock::now();
    frame_stamps_[frame_stamp_head_] = now;
    frame_stamp_head_                = (frame_stamp_head_ + 1) % kFpsWindow;
    if (frame_stamp_count_ < kFpsWindow) ++frame_stamp_count_;

    // 只数最近 1 秒内的帧（环形缓冲里越新的样本越靠近 frame_stamp_head_）
    const auto window_start = now - std::chrono::seconds(1);
    size_t     valid        = 0;
    for (size_t i = 0; i < frame_stamp_count_; ++i)
    {
        const size_t idx = (frame_stamp_head_ + kFpsWindow - 1 - i) % kFpsWindow;
        if (frame_stamps_[idx] < window_start) break;
        ++valid;
    }
    fps_ = static_cast<float>(valid);
}

void DartDetector::findCoresByGreenRing(std::vector<DartTarget> &out)
{
    if (hsv_.empty()) return;

    cv::Mat v_channel, s_channel;
    cv::extractChannel(hsv_, v_channel, 2);
    cv::extractChannel(hsv_, s_channel, 1);

    cv::Mat core_mask;
    cv::threshold(v_channel, core_mask, core_val_min_, 255, cv::THRESH_BINARY);
    cv::Mat sat_cap;
    cv::threshold(s_channel, sat_cap, core_max_sat_, 255, cv::THRESH_BINARY_INV);
    core_mask &= sat_cap;

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(core_mask, core_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(core_mask, core_mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> cores;
    cv::findContours(core_mask, cores, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat green_mask;
    cv::inRange(hsv_, cv::Scalar(35, 100, 80), cv::Scalar(85, 255, 255), green_mask);

    for (const auto &c : cores)
    {
        const double area = cv::contourArea(c);
        if (area < core_min_area_ || area > kFallbackMaxCoreArea) continue;

        cv::Point2f center;
        float       radius = 0.0f;
        cv::minEnclosingCircle(c, center, radius);
        if (radius < core_min_radius_ || radius > kFallbackMaxCoreRadius) continue;

        // 环只需要在核附近计算：每个核都建整幅图的 fill/dilated 太慢
        const int pad = static_cast<int>(std::ceil(radius)) + 2;
        cv::Rect  roi = cv::boundingRect(c);
        roi.x      -= pad;
        roi.y      -= pad;
        roi.width  += 2 * pad;
        roi.height += 2 * pad;
        roi = roi & cv::Rect(0, 0, hsv_.cols, hsv_.rows);
        if (roi.width < 3 || roi.height < 3) continue;

        cv::Mat fill = cv::Mat::zeros(roi.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> shifted{c};
        for (auto &p : shifted[0]) p -= roi.tl();
        cv::drawContours(fill, shifted, 0, cv::Scalar(255), cv::FILLED);

        const int k = std::max(3, static_cast<int>(radius * 2.0f) | 1);
        cv::Mat   dilated;
        cv::dilate(fill, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
        cv::Mat ring;
        cv::subtract(dilated, fill, ring);

        const double ring_n = static_cast<double>(cv::countNonZero(ring));
        if (ring_n < 1.0) continue;

        const double green_n = static_cast<double>(cv::countNonZero(green_mask(roi) & ring));
        const double ratio   = green_n / ring_n;
        // 近处大灯的白核很大，光晕会被支架和黑背景切断，绿环占比天然偏低（实测约 0.5）；
        // 小核仍用配置里的严格阈值，避免把普通白灯/小反光点误认成靶灯。
        const double ring_min = (radius >= big_core_radius_) ? big_core_ring_min_ : core_green_ring_min_;
        if (ratio < ring_min) continue;

        const double per = cv::arcLength(c, true);

        DartTarget t;
        t.center      = center;
        t.radius      = radius;
        t.area        = static_cast<float>(area);
        t.circularity = (per > 1e-6) ? static_cast<float>(4.0 * PI * area / (per * per)) : 0.0f;
        t.confidence  = static_cast<float>(std::clamp(ratio, 0.0, 1.0));
        t.valid       = true;
        out.push_back(t);
    }
}

void DartDetector::showResult(const cv::Mat &bgr_img, const cv::Mat &bin_img,
                              const std::vector<DartTarget> &targets, const std::string &window_name) const
{
    if (window_name.empty()) return;

    auto &display = rm_utils::Display::getInstance();

    std::vector<rm_utils::DisplayPoint> points;
    std::vector<rm_utils::DisplayLine>  lines;
    std::vector<rm_utils::DisplayText>  texts;

    // 和多目标身份逻辑保持一致：取面积最大的两个，再按 x 分 L/R
    int left_big = -1, right_big = -1;
    if (targets.size() >= 2)
    {
        size_t big1 = 0, big2 = 1;
        if (targets[big2].area > targets[big1].area) std::swap(big1, big2);
        for (size_t i = 2; i < targets.size(); ++i)
        {
            if (targets[i].area > targets[big1].area)
            {
                big2 = big1;
                big1 = i;
            }
            else if (targets[i].area > targets[big2].area)
            {
                big2 = i;
            }
        }
        left_big  = static_cast<int>((targets[big1].center.x <= targets[big2].center.x) ? big1 : big2);
        right_big = static_cast<int>((left_big == static_cast<int>(big1)) ? big2 : big1);
    }

    for (size_t i = 0; i < targets.size(); ++i)
    {
        const DartTarget &target = targets[i];
        const int         cx     = static_cast<int>(target.center.x);
        const int         cy     = static_cast<int>(target.center.y);
        const int         r      = std::max(1, static_cast<int>(target.radius));

        // 圆心
        rm_utils::DisplayPoint p;
        p.x     = cx;
        p.y     = cy;
        p.color = {255, 0, 0, 255};
        p.size  = 4;
        points.push_back(p);

        // 中心十字
        rm_utils::DisplayLine cross_h;
        cross_h.x1 = cx - 10; cross_h.y1 = cy;
        cross_h.x2 = cx + 10; cross_h.y2 = cy;
        cross_h.color = {255, 255, 0, 255}; cross_h.thickness = 2;
        lines.push_back(cross_h);

        rm_utils::DisplayLine cross_v;
        cross_v.x1 = cx; cross_v.y1 = cy - 10;
        cross_v.x2 = cx; cross_v.y2 = cy + 10;
        cross_v.color = {255, 255, 0, 255}; cross_v.thickness = 2;
        lines.push_back(cross_v);

        // 方形框（内切于灯芯半径，稍微收紧一点），和圆一起画方便对照
        const int box = std::max(2, static_cast<int>(r * 0.85));
        const int x1 = cx - box, y1 = cy - box;
        const int x2 = cx + box, y2 = cy + box;
        const int box_edges[4][4] = {
            {x1, y1, x2, y1}, {x2, y1, x2, y2},
            {x2, y2, x1, y2}, {x1, y2, x1, y1}};
        for (const auto &e : box_edges)
        {
            rm_utils::DisplayLine line;
            line.x1 = e[0]; line.y1 = e[1]; line.x2 = e[2]; line.y2 = e[3];
            line.color = {0, 255, 255, 255}; line.thickness = 2;
            lines.push_back(line);
        }

        // 只画方框，不再画外接圆

        // 两个以上目标时：面积最大的两个里，偏左 = L（前哨站），偏右 = R（基地），其余用序号
        std::string tag;
        if (targets.size() == 1)
            tag = "T";
        else if (static_cast<int>(i) == left_big)
            tag = "L";
        else if (static_cast<int>(i) == right_big)
            tag = "R";
        else
            tag = "#" + std::to_string(i);

        rm_utils::DisplayText t;
        t.content = tag + " (" + std::to_string(cx) + "," + std::to_string(cy) +
                    ") r=" + std::to_string(static_cast<int>(target.radius));
        t.x       = cx + r + 5;
        t.y       = cy - 8;
        t.size    = 20;
        t.color   = {0, 255, 255, 255};
        texts.push_back(t);

        rm_utils::DisplayText t2;
        t2.content = "circ=" + std::to_string(target.circularity).substr(0, 4);
        t2.x       = cx + r + 5;
        t2.y       = cy + 16;
        t2.size    = 20;
        t2.color   = {0, 255, 0, 255};
        texts.push_back(t2);
    }

    const bool found = !targets.empty();

    rm_utils::DisplayText st;
    st.content = "target: " + std::string(found ? "FOUND" : "NONE");
    if (found && targets.size() > 1)
    {
        st.content += " (" + std::to_string(targets.size()) + ")";
    }
    st.x       = 20;
    st.y       = 20;
    st.size    = 24;
    st.color   = found ? SDL_Color{0, 255, 0, 255} : SDL_Color{255, 0, 0, 255};
    texts.push_back(st);

    // 实时帧率（右上角）：直接反映检测线程每秒处理了多少帧，
    // 相机断流/重连时这个值会掉下来，便于肉眼确认卡顿时间点。
    char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "FPS %.0f", static_cast<double>(fps_));

    rm_utils::DisplayText fps_label;
    fps_label.content = fps_buf;
    fps_label.x       = 220;   // 放在 "target: ..." 右边，避免被另一个窗口盖住
    fps_label.y       = 20;
    fps_label.size    = 24;
    fps_label.color   = {255, 255, 0, 255};

    texts.push_back(fps_label);

    rm_utils::DisplayText binary_fps = fps_label;
    binary_fps.x                     = 20;
    std::vector<rm_utils::DisplayText> binary_texts{binary_fps};
    display.display_add(window_name + "_binary", bin_img, binary_texts);
    display.display_add(window_name, bgr_img, texts, points, lines);
}

} // namespace dart_aim
