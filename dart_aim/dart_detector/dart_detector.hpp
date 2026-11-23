#ifndef _DART_DETECTOR_H_
#define _DART_DETECTOR_H_

#include <opencv2/core.hpp>
#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "dart_types.hpp"

namespace dart_aim
{

/**
 * @brief 绿色圆形目标检测器
 *
 * 思路：BGR -> HSV，用绿色阈值区间做 inRange 二值化 -> 形态学去噪 ->
 *       提取轮廓 -> 按面积、圆度、长宽比过滤 -> 取最可靠的圆作为目标。
 *
 * 相比自瞄的装甲板检测（灯条配对 + 数字识别 + PnP），这里只做单目标的
 * 颜色 + 形状检测，不依赖深度学习模型，也不做 3D 位姿解算。
 */
class DartDetector
{
  public:
    explicit DartDetector(const std::string &config_path = "config/dart_vision.yaml");

    /**
     * @brief 识别绿色圆形目标
     * @param bgr_img 输入 BGR 图像
     * @param window_name 调试窗口名称（debug 模式下使用）
     * @return 检测结果；未检测到时 valid == false
     */
    DartTarget detect(const cv::Mat &bgr_img, const std::string &window_name = "") noexcept;

    /**
     * @brief 识别画面中所有绿色引导灯（按 x 从小到大排序）
     *
     * 主路径和"白核 + 绿环"通道的结果会合并、去重；
     * 调用方（瞄准模块）负责按左右分配前哨站/基地身份。
     */
    std::vector<DartTarget> detectAll(const cv::Mat &bgr_img, const std::string &window_name = "") noexcept;

    // 调试开关
    bool debug() const { return debug_; }

    // 最近一帧的灯芯二值掩码（多目标时画所有灯芯），供离线工具保存黑白视频
    const cv::Mat &lastMask() const { return detection_mask_; }

  private:
    // 真正执行检测；all_out 非空时把合并后的所有目标写出去
    DartTarget detectImpl(const cv::Mat &bgr_img, const std::string &window_name,
                          std::vector<DartTarget> *all_out) noexcept;

    void showResult(const cv::Mat &bgr_img, const cv::Mat &bin_img, const std::vector<DartTarget> &targets,
                    const std::string &window_name) const;

    // 白核 + 绿环通道：把所有满足条件的核写进 out
    void findCoresByGreenRing(std::vector<DartTarget> &out);

    // 统计最近 1 秒内 detect() 的调用次数，供调试窗口显示实时帧率
    void updateFps() noexcept;
    static constexpr size_t kFpsWindow = 128;
    std::array<std::chrono::steady_clock::time_point, kFpsWindow> frame_stamps_{};
    size_t frame_stamp_count_ = 0;
    size_t frame_stamp_head_  = 0;
    float  fps_               = 0.0f;

    // HSV 绿色阈值
    int hue_low_    = 35;   // H 下限
    int hue_high_   = 85;   // H 上限
    int sat_low_    = 60;   // S 下限
    int sat_high_   = 255;  // S 上限
    int val_low_    = 60;   // V 下限
    int val_high_   = 255;  // V 上限

    // 形态学
    int morph_kernel_size_ = 3;   // 开/闭运算核大小

    // 轮廓过滤
    double min_area_         = 150.0;   // 最小面积（像素²）
    double max_area_         = 20000.0; // 最大面积（像素²），排除大块绿色背景
    double min_circularity_  = 0.10;    // 最小圆度（形状主要由填充率/长宽比把关）
    double max_aspect_ratio_ = 2.2;     // 外接椭圆长短轴比上限
    double min_fill_         = 0.30;    // 最小填充率 area/(πr²)，排除细长/镂空绿色
    double min_mean_sat_     = 120.0;   // 轮廓内平均饱和度下限，排除灰绿/浅绿
    double min_mean_val_     = 95.0;    // 轮廓内平均亮度下限，排除暗绿色物体

    // 灯芯（发光核心）判据：过曝的灯芯在 HSV 里可能变成白色（S 很低），
    // 所以只在绿色外圈内部按亮度 V 找亮核，用亮核定位和定半径。
    double core_val_min_     = 240.0;   // 灯芯最低亮度
    double core_max_sat_     = 90.0;    // 灯芯最高饱和度（过曝核心接近白色，S 很低）
    double core_min_area_    = 50.0;    // 灯芯最小面积（像素²）
    double core_min_radius_  = 8.0;     // 灯芯最小半径（像素）
    double core_green_ring_min_ = 0.60; // 小核的绿环占比下限
    double big_core_radius_     = 25.0; // 半径 >= 该值视为大核
    double big_core_ring_min_   = 0.35; // 大核的绿环占比下限（过曝严重时可降到 0.2）

    bool debug_ = false;

    // 缓存图像，避免反复申请内存
    cv::Mat hsv_;
    cv::Mat mask_;
    cv::Mat morph_;
    cv::Mat detection_mask_;   // 被选中的灯芯（黑白输出用）
};

} // namespace dart_aim

#endif // _DART_DETECTOR_H_
