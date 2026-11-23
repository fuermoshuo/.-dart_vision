#include "hikcamera.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include "mas_log.hpp"

// 守护线程：每 kDaemonPollMs 检查一次连接状态和采集心跳。
//   - 真掉线：按退避完整重连；
//   - 采集卡死（长时间没有新帧）：先软重启取流（StopGrabbing/StartGrabbing），
//     仍没有帧再完整 close/open。软重启通常只要几百毫秒，而完整重连实测
//     要 1.2~1.4 s，期间一帧都没有，表现就是明显的周期性卡顿。
void hikcamera::HikCamera::daemonLoop()
{
    while (running_.load(std::memory_order_relaxed))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(kDaemonPollMs));

        if (!running_.load(std::memory_order_relaxed))
        {
            break;
        }

        if (!isConnected.load(std::memory_order_relaxed))
        {
            // 已经掉线（SDK 回调或上一次完整重连失败）：按退避重连
            stall_recoveries_ = 0;
            TryConnect(false);
            continue;
        }

        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
        const int64_t last_ms = last_frame_tick_ms_.load(std::memory_order_relaxed);
        const int64_t gap_ms  = now_ms - last_ms;

        // 软重启后只给一个较短的等待窗口：有帧就恢复正常判断，没帧就尽快完整重连
        const int64_t timeout_ms = (stall_recoveries_ > 0) ? kStreamRestartGraceMs : kWatchdogTimeoutMs;

        if (gap_ms <= timeout_ms)
        {
            stall_recoveries_ = 0;   // 心跳正常，之前的软重启视为成功
            continue;
        }

        if (stall_recoveries_ < kMaxStreamRestarts)
        {
            ++stall_recoveries_;
            MAS_LOG_WARN("Capture heartbeat stale for over {} ms, restarting grabber ({}/{})",
                         timeout_ms, stall_recoveries_, kMaxStreamRestarts);

            if (restartGrabber())
            {
                // 从"重启完成"开始重新计时，避免刚启动就被判超时
                last_frame_tick_ms_.store(now_ms, std::memory_order_relaxed);
                continue;
            }

            MAS_LOG_WARN("Restart grabber failed, falling back to full reconnect");
            stall_recoveries_ = kMaxStreamRestarts;
        }

        // 软恢复无效：完整关闭再打开设备
        MAS_LOG_WARN("Capture did not recover after grabber restart, forcing full reconnect");
        // 心跳卡死本身就是一种"合法断线信号"，为分层延迟诊断记录信号触发时刻。
        markDisconnected(std::chrono::steady_clock::now());
        TryConnect(true);
        stall_recoveries_ = 0;
    }
}

// 轻量恢复：只停/开始取流，不关闭设备句柄。
// 用于采集卡死但设备仍在线的情况（USB 2.0 带宽抖动时很常见）。
bool hikcamera::HikCamera::restartGrabber()
{
    std::unique_lock<std::timed_mutex> lock(camera_mutex_,
                                            std::chrono::milliseconds(kRestartLockTimeoutMs));
    if (!lock.owns_lock())
    {
        MAS_LOG_WARN("restartGrabber: could not acquire camera_mutex_ within {} ms",
                     kRestartLockTimeoutMs);
        return false;
    }

    if (!isConnected.load(std::memory_order_relaxed) || handle == NULL)
    {
        return false;
    }

    int nRet = MV_CC_StopGrabbing(handle);
    if (nRet != MV_OK)
    {
        MAS_LOG_WARN("restartGrabber: StopGrabbing fail! nRet 0x{0:x}", nRet);
        return false;
    }

    nRet = MV_CC_StartGrabbing(handle);
    if (nRet != MV_OK)
    {
        MAS_LOG_WARN("restartGrabber: StartGrabbing fail! nRet 0x{0:x}", nRet);
        return false;
    }

    MAS_LOG_INFO("restartGrabber: stream restarted without reopening the device");
    return true;
}

// 重连：由 daemonLoop 周期性调用
void hikcamera::HikCamera::TryConnect(bool bypass_backoff)
{
    const auto now          = std::chrono::steady_clock::now();
    const auto connect_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_reconnect_attempt_).count();

    // 退避：上次尝试之后还没到 reconnect_backoff_ms_，这次直接跳过。
    // 注意首次调用时 last_reconnect_attempt_ 是默认值（很久以前），
    // 所以第一次一定会通过——这是故意的，启动/首次掉线要立刻尝试。
    if (!bypass_backoff && connect_time < reconnect_backoff_ms_)
    {
        return;
    }

    last_reconnect_attempt_ = now;

    // 关键修复：关不掉就不要再打开。
    // 旧逻辑 closeCamera() 失败（拿不到相机锁）后会继续 openCamera()，
    // 而 openCamera() 看到 "isConnected && handle" 会直接返回 true，
    // 于是日志打出 "reconnect successfully"，但相机根本没重开、采集仍然卡死。
    if (!closeCamera())
    {
        reconnect_backoff_ms_ = std::min(reconnect_backoff_ms_ * 2, kMaxBackoffMs);

        const auto since_last_log =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fail_log_time_).count();
        if (last_fail_log_time_.time_since_epoch().count() == 0 || since_last_log >= kFailLogThrottleMs)
        {
            MAS_LOG_WARN("Camera close failed, postpone reconnect for {} ms", reconnect_backoff_ms_);
            last_fail_log_time_ = now;
        }
        return;
    }

    const bool success = openCamera() && isConnected.load(std::memory_order_relaxed);

    if (success)
    {
        reconnect_backoff_ms_ = kInitialBackoffMs;
        MAS_LOG_INFO("Camera reconnect successfully");
    }
    else
    {
        reconnect_backoff_ms_ = std::min(reconnect_backoff_ms_ * 2, kMaxBackoffMs);

        //限制日志打印频率，避免刷屏
        const auto since_last_log =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fail_log_time_).count();
        if (last_fail_log_time_.time_since_epoch().count() == 0 || since_last_log >= kFailLogThrottleMs)
        {
            MAS_LOG_WARN("Camera reconnect failed, next attempt after {} ms", reconnect_backoff_ms_);
            last_fail_log_time_ = now;
        }
    }
}
