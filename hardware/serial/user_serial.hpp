#ifndef _USER_SERIAL_H_
#define _USER_SERIAL_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "serial.h"
#include "serial_types.hpp"

/**
 * @brief 飞镖视觉串口封装（单例）
 *
 * 负责：
 *  - 发送目标信息（found / target / state / target_yaw_deg）给电控；
 *  - 接收电控下发的单字节命令（0~3），并做断线重连。
 *
 * 相比自瞄版删除了所有 IMU 四元数相关的接收、插值与有效性判断逻辑。
 */
class UserSerial
{
  public:
    UserSerial(const UserSerial &)            = delete;
    UserSerial &operator=(const UserSerial &) = delete;

    static UserSerial &getInstance(const std::string &config_path = "config/serial_config.yaml")
    {
        static UserSerial instance(config_path);
        return instance;
    }

    ~UserSerial();

    // 关闭串口
    void closeSerial();

    // 串口是否已连接
    bool isConnectedStatus() const { return isConnected; }

    /**
     * @brief 更新待发送的视觉数据（供 dart 线程调用）
     * @param found 选中目标是否可见/已锁定
     * @param target 0=无, 1=前哨站, 2=基地
     * @param state 0=idle, 1=acquire, 2=hold
     * @param target_yaw_deg 云台需要转过的角度（度）
     */
    void sendDart(uint8_t found, uint8_t target, uint8_t state, float target_yaw_deg);

    // 目标丢失时，清空发送内容（found=0）
    void disableDart();

    // 把发送包写入串口（serial 线程以固定频率调用）
    void sendPacketToSerial();

    // 从串口读取命令字节（serial 线程调用）
    void receiveSerial();

    // 断线重连
    void reconnect();

    // 是否收到过至少一次有效命令（当前仅用于日志/心跳，不参与控制）
    bool hasReceivedData() const { return has_received_data_.load(); }

    // 最近一次接收到的命令字节
    uint8_t lastCommand() const { return last_command_.load(); }

  private:
    explicit UserSerial(const std::string &config_path = "config/serial_config.yaml");

    std::shared_ptr<serial::Serial>       serial_port;
    std::atomic<bool>                     isConnected{false};
    std::chrono::steady_clock::time_point last_reconnect_time_;

    // 发送数据包
    DartSendPacket send_packet_{};
    std::mutex     send_packet_mutex_;

    // 配置参数
    std::string           port_{"/dev/mascdc"};
    uint32_t              baudrate_{115200};
    int                   timeout_{2};
    serial::bytesize_t    bytesize_{serial::eightbits};
    serial::parity_t      parity_{serial::parity_none};
    serial::stopbits_t    stopbits_{serial::stopbits_one};
    serial::flowcontrol_t flowcontrol_{serial::flowcontrol_none};
    bool                  debug_{false};
    bool                  enable_{true};   // false = 不打开/不重连串口

    // 接收缓冲区（单字节命令，只保留一次读取的临时缓冲区）
    std::vector<uint8_t>  temp_read_buf_;
    std::atomic<bool>     has_received_data_{false};
    std::atomic<uint8_t>  last_command_{0};
};

#endif // _USER_SERIAL_H_
