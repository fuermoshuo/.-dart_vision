#ifndef _COMMON_DEF_H_
#define _COMMON_DEF_H_

#include <opencv2/core/mat.hpp>
#include <chrono>
#include <string>

struct CameraFrame
{
    cv::Mat                                frame;
    std::chrono::steady_clock::time_point  timestamp;
    std::chrono::steady_clock::time_point  t_recv; 
};

class Base_Camera
{
  public:
    virtual ~Base_Camera() = default;

    virtual bool openCamera() = 0;

    // 返回 false 表示这次关闭没有真正完成（例如拿不到相机锁），调用方不能假装已经关掉
    virtual bool closeCamera() = 0;

    virtual CameraFrame getImage() = 0;
    
    virtual bool isConnectedStatus() const = 0;

};

#endif // _COMMON_DEF_H_
