#include "dart_track.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>

#include "mas_log.hpp"
#include "yaml-cpp/yaml.h"

namespace dart_aim
{

DartTrack::DartTrack(const std::string &aim_config_path, const std::string &camera_config_path)
{
    // 读相机内参（fx / cx / 图像宽）
    try
    {
        std::ifstream file(camera_config_path);
        if (file.good())
        {
            YAML::Node config = YAML::LoadFile(camera_config_path);
            auto       matrix = config["calibration"]["camera_matrix"].as<std::vector<double>>();
            if (matrix.size() >= 9)
            {
                fx_ = static_cast<float>(matrix[0]);
                cx_ = static_cast<float>(matrix[2]);
            }
            image_width_ = config["calibration"]["image_width"].as<double>(1280.0);
        }
    }
    catch (const std::exception &e)
    {
        MAS_LOG_WARN("Failed to load camera intrinsics, using defaults: {}", e.what());
    }

    // 读瞄准参数
    try
    {
        std::ifstream file(aim_config_path);
        if (file.good())
        {
            YAML::Node node = YAML::LoadFile(aim_config_path)["dart_vision"]["track"];

            debug_                = node["debug"].as<bool>(false);
            align_tolerance_deg_  = node["align_tolerance_deg"].as<float>(0.3f);
            align_frames_         = std::max(1, node["align_frames"].as<int>(5));
            yaw_sign_             = node["yaw_sign"].as<float>(1.0f);
            role_match_norm_      = node["role_match_norm"].as<float>(0.15f);
            max_lost_ms_          = node["max_lost_ms"].as<double>(500.0);
            detect_hold_ms_       = std::max(0, node["detect_hold_ms"].as<int>(300));
            default_target_       = static_cast<uint8_t>(std::clamp(node["default_target"].as<int>(0), 0, 2));
            swap_sides_           = node["swap_sides"].as<bool>(false);
            hold_enable_          = node["hold_enable"].as<bool>(true);

            MAS_LOG_INFO("dart_track yaml loaded successfully");
            MAS_LOG_INFO("track params: default_target={} align_tolerance_deg={:.2f} align_frames={} "
                         "hold_enable={} detect_hold_ms={} yaw_sign={:.1f} swap_sides={}",
                         static_cast<int>(default_target_), align_tolerance_deg_, align_frames_,
                         hold_enable_, detect_hold_ms_, yaw_sign_, swap_sides_);
        }
    }
    catch (const std::exception &e)
    {
        MAS_LOG_WARN("Failed to load track config, using defaults: {}", e.what());
    }

    command_ = default_target_;
    state_   = (command_ == TARGET_NONE) ? AIM_IDLE : AIM_ACQUIRE;
}

DartAimResult DartTrack::update(const std::vector<DartTarget> &targets) noexcept
{
    DartAimResult result;
    result.target = command_;
    result.state  = state_;
    
    // 没有目标选择：不输出瞄准误差
    if (command_ == TARGET_NONE)
    {
        state_       = AIM_IDLE;
        align_count_ = 0;
        result.state = AIM_IDLE;
        return result;
    }

    // 已锁定：无论目标之后怎么动，都保持上一次的对准结果
    if (state_ == AIM_HOLD)
    {
        result.found    = true;
        result.state    = AIM_HOLD;
        result.center_x = hold_x_;
        result.center_y = hold_y_;
        return result;
    }

    if (targets.empty())
    {
        state_       = AIM_ACQUIRE;
        align_count_ = 0;
        result.state = AIM_ACQUIRE;

        // 短暂丢失（灯太亮/过曝导致几帧检不到）时，沿用上一次的角度，
        // 避免发给电控的 found 在 0/1 之间抖。超过 detect_hold_ms 才真的丢。
        applyDetectHold(result, std::chrono::steady_clock::now());
        return result;
    }

    // 长时间看不到任何灯，清掉身份位置记忆，避免拿旧位置乱匹配
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_any_seen_).count() > max_lost_ms_)
    {
        has_last_.fill(false);
    }
    last_any_seen_ = now;

    std::vector<uint8_t> roles(targets.size(), TARGET_NONE);
    assignRoles(targets, roles);

    int idx = -1;
    for (size_t i = 0; i < targets.size(); ++i)
    {
        if (roles[i] == command_)
        {
            idx = static_cast<int>(i);
            break;
        }
    }

    // 只剩一个灯时，不管它在画面什么位置，都直接当作当前选中的目标。
    // 左右身份是两盏灯之间的相对关系，不能用画面绝对位置判断；
    // 画面里只剩一个灯时也没有别的可瞄，直接用它。
    if (idx < 0 && targets.size() == 1)
    {
        roles[0]            = command_;
        last_x_[command_]   = targets[0].center.x;
        has_last_[command_] = true;
        idx                 = 0;
    }

    if (idx < 0)
    {
        state_       = AIM_ACQUIRE;
        align_count_ = 0;
        result.state = AIM_ACQUIRE;
        applyDetectHold(result, std::chrono::steady_clock::now());
        return result;
    }

    return updateSelected(targets[idx], command_);
}

DartAimResult DartTrack::update(const DartTarget &target) noexcept
{
    // 离线视频工具兼容：不看串口命令，直接把这一帧的目标当作选中目标
    DartAimResult result;
    result.target = command_;

    if (state_ == AIM_HOLD)
    {
        result.found    = true;
        result.state    = AIM_HOLD;
        result.center_x = hold_x_;
        result.center_y = hold_y_;
        return result;
    }

    if (!target.valid)
    {
        state_       = AIM_ACQUIRE;
        align_count_ = 0;
        result.state = AIM_ACQUIRE;
        applyDetectHold(result, std::chrono::steady_clock::now());
        return result;
    }

    last_any_seen_ = std::chrono::steady_clock::now();
    return updateSelected(target, command_);
}

void DartTrack::assignRoles(const std::vector<DartTarget> &targets, std::vector<uint8_t> &roles) noexcept
{
    const size_t n = targets.size();
    if (n == 0) return;

    // 按 x 排序（detectAll 已经排过，这里再兜底一次）
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return targets[a].center.x < targets[b].center.x; });

    const uint8_t left_role  = swap_sides_ ? TARGET_BASE : TARGET_OUTPOST;
    const uint8_t right_role = swap_sides_ ? TARGET_OUTPOST : TARGET_BASE;

    if (n >= 2)
    {
        // 先从所有检测里取面积最大的两个（排除小反光点/误检），
        // 再按这两个的 x 分左右：两者中偏左 = 前哨站，偏右 = 基地。
        std::vector<size_t> by_area(order);
        std::sort(by_area.begin(), by_area.end(),
                  [&](size_t a, size_t b) { return targets[a].area > targets[b].area; });

        const size_t big1 = by_area[0];
        const size_t big2 = by_area[1];
        const size_t left_idx  = (targets[big1].center.x <= targets[big2].center.x) ? big1 : big2;
        const size_t right_idx = (left_idx == big1) ? big2 : big1;

        roles[left_idx]  = left_role;
        roles[right_idx] = right_role;

        last_x_[left_role]  = targets[left_idx].center.x;
        last_x_[right_role] = targets[right_idx].center.x;
        has_last_[left_role]  = true;
        has_last_[right_role] = true;
    }
    else
    {
        // 只剩一个：离上次已知位置最近的灯继承身份
        const float   x      = targets[order.front()].center.x;
        uint8_t       best   = TARGET_NONE;
        float         best_d = role_match_norm_ * std::max(1.0f, image_width_);

        for (uint8_t role : {static_cast<uint8_t>(TARGET_OUTPOST), static_cast<uint8_t>(TARGET_BASE)})
        {
            if (!has_last_[role]) continue;
            const float d = std::abs(x - last_x_[role]);
            if (d < best_d)
            {
                best_d = d;
                best   = role;
            }
        }
        roles[order.front()] = best;

        // 单灯期间也要更新它的历史位置，否则灯移动后身份会丢
        if (best != TARGET_NONE)
        {
            last_x_[best]    = x;
            has_last_[best]  = true;
        }
    }
}

DartAimResult DartTrack::updateSelected(const DartTarget &target, uint8_t role) noexcept
{
    DartAimResult result;
    result.target = role;

    const float dx = target.center.x - cx_;
    result.error_x    = dx / std::max(1.0f, image_width_);
    result.target_yaw = yaw_sign_ * static_cast<float>(std::atan2(static_cast<double>(dx),
                                                                static_cast<double>(std::max(1.0f, fx_))));
    result.center_x = target.center.x;
    result.center_y = target.center.y;
    result.found    = true;
    result.state    = AIM_ACQUIRE;
    state_          = AIM_ACQUIRE;

    // 记录最近一次有效输出，供短暂丢失时保持
    has_last_out_  = true;
    last_out_x_    = result.center_x;
    last_out_y_    = result.center_y;
    last_out_yaw_  = result.target_yaw;
    last_out_time_ = std::chrono::steady_clock::now();

    const float align_tolerance_rad = align_tolerance_deg_ * 3.14159265f / 180.0f;

    if (hold_enable_ && std::abs(result.target_yaw) <= align_tolerance_rad)
    {
        ++align_count_;
        if (align_count_ >= align_frames_)
        {
            // 粗对准完成：锁定，之后不再跟随目标移动
            state_    = AIM_HOLD;
            hold_x_   = target.center.x;
            hold_y_   = target.center.y;
            align_count_     = 0;
            result.state     = AIM_HOLD;
            result.error_x   = 0.0f;
            result.target_yaw = 0.0f;
        }
    }
    else
    {
        align_count_ = 0;
    }

    return result;
}

bool DartTrack::applyDetectHold(DartAimResult &result, std::chrono::steady_clock::time_point now) noexcept
{
    if (!has_last_out_ || detect_hold_ms_ <= 0) return false;

    const auto age_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_out_time_).count();
    if (age_ms > detect_hold_ms_) return false;

    result.found      = true;
    result.state      = AIM_ACQUIRE;
    result.target     = command_;
    result.center_x   = last_out_x_;
    result.center_y   = last_out_y_;
    result.target_yaw = last_out_yaw_;
    return true;
}

void DartTrack::setCommand(uint8_t cmd) noexcept
{
    if (cmd == kCmdRecapture)
    {
        // 保持当前目标，重新粗对准
        if (command_ != TARGET_NONE)
        {
            state_       = AIM_ACQUIRE;
            align_count_ = 0;
        }
        return;
    }

    if (cmd > TARGET_BASE) return;
    if (cmd == command_) return;

    command_ = cmd;
    state_   = (cmd == TARGET_NONE) ? AIM_IDLE : AIM_ACQUIRE;
    align_count_ = 0;
    hold_x_      = 0.0f;
    hold_y_      = 0.0f;
    has_last_out_ = false;   // 换目标后不要沿用上一个目标的角度
}

void DartTrack::reset() noexcept
{
    command_     = default_target_;
    state_       = (command_ == TARGET_NONE) ? AIM_IDLE : AIM_ACQUIRE;
    align_count_ = 0;
    hold_x_      = 0.0f;
    hold_y_      = 0.0f;
    has_last_out_ = false;
    has_last_.fill(false);
}

} // namespace dart_aim
