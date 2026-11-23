#include "user_serial.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <stdio.h>
#include <thread>

#include "mas_log.hpp"
#include "serial_types.hpp"
#include <yaml-cpp/yaml.h>

extern std::atomic<bool> g_shutdown;

UserSerial::UserSerial(const std::string &config_path)
{
    try
    {
        std::ifstream file(config_path);
        if (!file.good())
        {
            MAS_LOG_WARN("Config file not found: {}, using default values", config_path.c_str());
        }

        YAML::Node config = YAML::LoadFile(config_path);

        if (config["serial"])
        {
            if (config["serial"]["enable"]) enable_ = config["serial"]["enable"].as<bool>(true);
            if (config["serial"]["debug"]) debug_ = config["serial"]["debug"].as<bool>(false);
            if (config["serial"]["port"]) port_ = config["serial"]["port"].as<std::string>();
            if (config["serial"]["baudrate"]) baudrate_ = config["serial"]["baudrate"].as<uint32_t>(115200);
            if (config["serial"]["timeout"]) timeout_ = config["serial"]["timeout"].as<int>(2);
            if (config["serial"]["bytesize"])
            {
                int bs = config["serial"]["bytesize"].as<int>(8);
                switch (bs)
                {
                case 5: bytesize_ = serial::fivebits; break;
                case 6: bytesize_ = serial::sixbits; break;
                case 7: bytesize_ = serial::sevenbits; break;
                case 8: bytesize_ = serial::eightbits; break;
                default: bytesize_ = serial::eightbits; break;
                }
            }
            if (config["serial"]["parity"])
            {
                std::string p = config["serial"]["parity"].as<std::string>("none");
                if (p == "none") parity_ = serial::parity_none;
                else if (p == "odd") parity_ = serial::parity_odd;
                else if (p == "even") parity_ = serial::parity_even;
                else parity_ = serial::parity_none;
            }
            if (config["serial"]["stopbits"])
            {
                int sb = config["serial"]["stopbits"].as<int>(1);
                if (sb == 1) stopbits_ = serial::stopbits_one;
                else if (sb == 2) stopbits_ = serial::stopbits_two;
                else stopbits_ = serial::stopbits_one;
            }
            if (config["serial"]["flowcontrol"])
            {
                std::string fc = config["serial"]["flowcontrol"].as<std::string>("none");
                if (fc == "none") flowcontrol_ = serial::flowcontrol_none;
                else if (fc == "software") flowcontrol_ = serial::flowcontrol_software;
                else if (fc == "hardware") flowcontrol_ = serial::flowcontrol_hardware;
                else flowcontrol_ = serial::flowcontrol_none;
            }
        }

        MAS_LOG_INFO("Serial config loaded successfully from: {}", config_path.c_str());
    }
    catch (const YAML::Exception &e)
    {
        MAS_LOG_ERROR("YAML parsing error: {}", e.what());
    }
    catch (const std::exception &e)
    {
        MAS_LOG_ERROR("Error loading serial config: {}", e.what());
    }

    temp_read_buf_.resize(256);
    has_received_data_.store(false);

    if (!enable_)
    {
        isConnected = false;
        MAS_LOG_INFO("Serial disabled by config (serial.enable = false)");
        return;
    }

    try
    {
        serial_port =
            std::make_shared<serial::Serial>(port_, baudrate_, serial::Timeout::simpleTimeout(timeout_), bytesize_, parity_, stopbits_, flowcontrol_);
        if (serial_port && serial_port->isOpen())
        {
            isConnected = true;
        }
        else
        {
            isConnected = false;
            return;
        }
    }
    catch (const std::exception &e)
    {
        MAS_LOG_ERROR("Exception while opening serial port: {}", e.what());
        isConnected = false;
    }
}

UserSerial::~UserSerial() {}

void UserSerial::closeSerial()
{
    if (serial_port)
    {
        if (serial_port->isOpen())
        {
            serial_port->close();
        }
        serial_port.reset();
    }
    isConnected = false;
    MAS_LOG_INFO("Serial port closed");
}

void UserSerial::sendDart(uint8_t found, uint8_t target, uint8_t state, float target_yaw_deg)
{
    std::lock_guard<std::mutex> lock(send_packet_mutex_);
    send_packet_.found          = found;
    send_packet_.target         = target;
    send_packet_.state          = state;
    send_packet_.target_yaw_deg = target_yaw_deg;
}

void UserSerial::disableDart()
{
    std::lock_guard<std::mutex> lock(send_packet_mutex_);
    send_packet_.found          = 0;
    send_packet_.target         = 0;
    send_packet_.state          = 0;
    send_packet_.target_yaw_deg = 0.0f;
}

void UserSerial::sendPacketToSerial()
{
    if (!enable_) return;

    while (!g_shutdown && (!isConnected || !serial_port || !serial_port->isOpen()))
    {
        reconnect();
    }
    if (g_shutdown) return;

    try
    {
        DartSendPacket packet;
        {
            std::lock_guard<std::mutex> lock(send_packet_mutex_);
            packet = send_packet_;
        }
        serial_port->write(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
    }
    catch (const std::exception &e)
    {
        MAS_LOG_ERROR("Exception while sending data: {}", e.what());
        isConnected = false;
    }
}

void UserSerial::receiveSerial()
{
    if (!enable_) return;

    while (!g_shutdown && (!isConnected || !serial_port || !serial_port->isOpen()))
    {
        reconnect();
    }
    if (g_shutdown) return;

    try
    {
        size_t bytes_available = serial_port->available();
        if (bytes_available > 0)
        {
            size_t bytes_to_read = std::min(bytes_available, temp_read_buf_.size());
            size_t bytes_read    = serial_port->read(temp_read_buf_.data(), bytes_to_read);

            if (debug_ && bytes_read > 0)
            {
                std::string hex_str;
                for (size_t i = 0; i < bytes_read; ++i)
                {
                    char buf[4];
                    snprintf(buf, sizeof(buf), "%02X ", temp_read_buf_[i]);
                    hex_str += buf;
                }
                MAS_LOG_INFO("Received {} bytes (HEX): {}", bytes_read, hex_str);
            }

            // 单字节命令：只接受 0~3，其它字节视为噪声丢弃。
            // 电控建议周期发送当前目标状态（0/1/2）；
            // command=3 是"重新粗对准"动作，只在需要时发一次。
            for (size_t i = 0; i < bytes_read; ++i)
            {
                const uint8_t cmd = temp_read_buf_[i];
                if (cmd <= 3)
                {
                    last_command_.store(cmd);
                    has_received_data_.store(true);
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        MAS_LOG_ERROR("Exception while receiving data: {}", e.what());
        isConnected = false;
    }
}

void UserSerial::reconnect()
{
    if (g_shutdown || !enable_)
    {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_reconnect_time_).count() < 5000)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }
    last_reconnect_time_ = now;

    MAS_LOG_INFO("Attempting to reconnect to serial port...");

    if (serial_port)
    {
        if (serial_port->isOpen())
        {
            serial_port->close();
        }
        serial_port.reset();
    }
    isConnected = false;

    disableDart();
    has_received_data_.store(false);

    try
    {
        serial_port =
            std::make_shared<serial::Serial>(port_, baudrate_, serial::Timeout::simpleTimeout(timeout_), bytesize_, parity_, stopbits_, flowcontrol_);
        if (serial_port && serial_port->isOpen())
        {
            isConnected = true;
            MAS_LOG_INFO("Reconnected to {}", port_.c_str());
        }
    }
    catch (const std::exception &e)
    {
        MAS_LOG_ERROR("Exception while reconnecting to {}: {}", port_.c_str(), e.what());
        isConnected = false;
    }
}
