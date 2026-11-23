#ifndef _DART_THREAD_H_
#define _DART_THREAD_H_

#include "SPSCQueue.h"
#include "common_def.hpp"
#include <atomic>
#include <memory>

extern std::atomic<bool> g_shutdown;
extern std::atomic<bool> g_save_photo;
extern std::atomic<int>  g_manual_command;

namespace threads
{
/**
 * @brief 飞镖视觉处理线程入口函数
 * @param camera_buffer 相机图像队列
 */
void dart_thread_func(std::shared_ptr<rigtorp::SPSCQueue<CameraFrame>> camera_buffer);

} // namespace threads

#endif // _DART_THREAD_H_
