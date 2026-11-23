#ifndef _DART_TRACK_H_
#define _DART_TRACK_H_

#include <array>
#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "dart_types.hpp"

namespace dart_aim
{

// 串口发送包的 target 字段
enum DartTargetId : uint8_t
{
    TARGET_NONE    = 0,
    TARGET_OUTPOST = 1,   // 两个最大目标中偏左的引导灯 = 前哨站
    TARGET_BASE    = 2,   // 两个最大目标中偏右的引导灯 = 基地
};

// 串口发送包的 state 字段
enum DartAimState : uint8_t
{
    AIM_IDLE    = 0,   // 没有目标选择，视觉不输出有效误差
    AIM_ACQUIRE = 1,   // 正在粗对准，target_yaw 有效
    AIM_HOLD    = 2,   // 已对准并锁定，电控应保持云台不动，交给制导镖
};

/**
 * @brief 瞄准输出结果（发给电控）
 */
struct DartAimResult
{
    bool    found      = false; // 选中目标是否可见/已锁定
    uint8_t target     = TARGET_NONE; // 0=无, 1=前哨站, 2=基地
    uint8_t state      = AIM_IDLE;    // 0=idle, 1=acquire, 2=hold
    float   error_x    = 0.0f;  // 归一化水平误差
    float   target_yaw = 0.0f;  // 目标 yaw 角（rad）
    float   center_x   = 0.0f;  // 选中目标中心 x（像素）
    float   center_y   = 0.0f;  // 选中目标中心 y（像素）
};

/**
 * @brief 目标选择 + 粗瞄准锁定（原周期中心追踪已移除）
 *
 * 新规则下基地目标不是持续摆动，而是"静止 → 事件触发换位置 → 静止"，
 * 所以底座只需要：
 *   1. 取面积最大的两个灯，按左右分前哨站/基地，再根据串口命令选择；
 *   2. 把云台转到选中目标，误差进入死区并保持 align_frames 帧后进入 HOLD；
 *   3. HOLD 之后不再输出角度（target_yaw = 0），
 *      电控保持当前角度，后续基地的随机移动交给制导镖。
 *
 * 不依赖 PnP、世界坐标或云台角度。
 */
class DartTrack
{
  public:
    DartTrack(const std::string &aim_config_path = "config/dart_vision.yaml",
              const std::string &camera_config_path = "config/hikcamera.yaml");

    /**
     * @brief 多目标输入（实车使用）
     * @param targets 按 x 排序的所有检测目标
     */
    DartAimResult update(const std::vector<DartTarget> &targets) noexcept;

    /**
     * @brief 单目标输入（离线视频工具兼容）
     */
    DartAimResult update(const DartTarget &target) noexcept;

    /**
     * @brief 设置目标选择命令
     *   0 = 停止/待机
     *   1 = 瞄准前哨站（左）
     *   2 = 瞄准基地（右）
     *   3 = 保持当前目标，重新粗对准（清掉 HOLD）
     */
    void setCommand(uint8_t cmd) noexcept;

    // 回到初始状态（配置里的 default_target）
    void reset() noexcept;

    bool debug() const { return debug_; }

    // 内参/参数只读访问（离线测试工具用）
    float fx() const { return fx_; }
    float cx() const { return cx_; }
    float imageWidth() const { return image_width_; }
    int   alignFrames() const { return align_frames_; }

    // 离线测试用：按视频分辨率覆盖内参
    void setImageWidth(float width) { image_width_ = width; }
    void setPrincipalPoint(float cx) { cx_ = cx; }
    void setFocalLength(float fx) { fx_ = fx; }

    static constexpr uint8_t kCmdRecapture = 3;

  private:
    // 对选中的目标做粗对准/锁定
    DartAimResult updateSelected(const DartTarget &target, uint8_t role) noexcept;

    // 目标短暂丢失时，用上一次的输出保持 detect_hold_ms_，避免 found 抖动
    bool applyDetectHold(DartAimResult &result, std::chrono::steady_clock::time_point now) noexcept;

    // 按左右给目标分配前哨站/基地身份
    void assignRoles(const std::vector<DartTarget> &targets, std::vector<uint8_t> &roles) noexcept;

    // 相机内参
    float fx_ = 1.0f;
    float cx_ = 640.0f;
    float image_width_ = 1280.0f;

    // 瞄准参数
    float  align_tolerance_deg_ = 0.3f;   // 进入锁定的角度阈值（度）
    int    align_frames_        = 5;      // 连续多少帧在阈值内才锁定
    float  yaw_sign_             = 1.0f;
    float  role_match_norm_      = 0.15f; // 只剩一个灯时，按上次位置匹配身份的阈值
    double max_lost_ms_          = 500.0; // 丢失超过该时间后清空身份位置记忆
    int    detect_hold_ms_       = 300;   // 选中目标短暂丢失时，继续输出上一次角度的时间
    uint8_t default_target_      = TARGET_NONE; // 未收到串口命令时的默认目标
    bool   swap_sides_           = false; // 相机左右装反时置 true
    bool   hold_enable_          = true;  // 是否允许 HOLD 锁定；调试云台时置 false

    bool debug_ = false;

    // 状态
    uint8_t command_     = TARGET_NONE;
    uint8_t state_       = AIM_IDLE;
    int     align_count_ = 0;
    float   hold_x_      = 0.0f;
    float   hold_y_      = 0.0f;

    // 最近一次看到的目标位置（用于只剩一个灯时继承身份）
    std::array<float, 3> last_x_{};
    std::array<bool, 3>  has_last_{};
    std::chrono::steady_clock::time_point last_any_seen_{};

    // 最近一次实际输出的瞄准结果（用于短暂丢失时保持）
    bool   has_last_out_ = false;
    float  last_out_x_   = 0.0f;
    float  last_out_y_   = 0.0f;
    float  last_out_yaw_ = 0.0f;
    std::chrono::steady_clock::time_point last_out_time_{};
};

} // namespace dart_aim

#endif // _DART_TRACK_H_
