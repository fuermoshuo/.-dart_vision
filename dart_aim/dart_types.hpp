#ifndef _DART_TYPES_H_
#define _DART_TYPES_H_

#include <opencv2/core.hpp>

namespace dart_aim
{

/**
 * @brief 检测到的绿色圆形目标（飞镖标靶上的闪烁绿灯）
 */
struct DartTarget
{
    cv::Point2f center;               // 圆心（像素坐标）
    float       radius      = 0.0f;   // 半径（像素）
    float       area        = 0.0f;   // 轮廓面积（像素²）
    float       circularity = 0.0f;   // 圆度 = 4π·area / perimeter²，越接近 1 越圆
    float       confidence  = 0.0f;   // 置信度 0~1（预留，当前按面积/圆度简单给出）
    bool        valid       = false;

    DartTarget() = default;
};

} // namespace dart_aim

#endif // _DART_TYPES_H_
