#include "hikcamera.hpp"
#include "MvCameraControl.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <string.h>
#include <exception>
#include <fstream>

#include "mas_log.hpp"
#include <yaml-cpp/yaml.h>

// 1.初始化exposure_time_，gain_
hikcamera::HikCamera::HikCamera(const std::string &config_path)
         : handle(nullptr), isConnected(false), exposure_time_(10000.0f), gain_(0.0f)
{
    memset(serialNumber, 0, sizeof(serialNumber));
    try
    {
        // 检查文件是否存在
        std::ifstream file(config_path);
        if (!file.good())
        {
            MAS_LOG_ERROR("Config file not found: {}, using default values", config_path.c_str());
            return;
        }

        // 加载YAML文件
        YAML::Node config = YAML::LoadFile(config_path);

        if (!config["camera"])
        {
            MAS_LOG_ERROR("No 'camera' section in config file");
        }

        YAML::Node camera_config = config["camera"];

        // 读取曝光时间
        if (camera_config["exposuretime"])
        {
            exposure_time_ = camera_config["exposuretime"].as<float>();
        }
        else
        {
            MAS_LOG_ERROR("exposuretime not found, using default: {}", exposure_time_);
        }

        // 读取增益
        if (camera_config["gain"])
        {
            gain_ = camera_config["gain"].as<float>();
        }
        else
        {
            MAS_LOG_ERROR("gain not found, using default: {}", gain_);
        }

        // 读取采集帧率上限（可选）。0 或不填表示沿用相机当前设置。
        // USB 2.0 全分辨率下带宽不够时，用它把帧率降到 15~20 可显著减少断流。
        if (camera_config["frame_rate"])
        {
            frame_rate_ = camera_config["frame_rate"].as<float>();
        }

        // 读取传输延迟
        if(camera_config["transfer_latency_ms"])
        {
            transfer_latency_ms_ = camera_config["transfer_latency_ms"].as<float>();
        }
        else
        {
            MAS_LOG_ERROR("transfer_latency_ms not found, using default: {}", transfer_latency_ms_);
        }

        // 读取目标相机序列号（可选）。多台同型号相机接入同一主机时(待做)，提供多相机接口(为多相机留接口)
        // if (camera_config["serial"])
        // {
        //     target_serial_ = camera_config["serial"].as<std::string>();
        //     MAS_LOG_INFO("Target camera serial number pinned from config: {}", target_serial_.c_str());
        // }

        MAS_LOG_INFO("Config loaded successfully from: {}", config_path.c_str());
        return;
    }
    catch (const YAML::Exception &e)
    {
        MAS_LOG_ERROR("YAML parsing error: {}", e.what());
    }
    catch (const std::exception &e)
    {
        MAS_LOG_ERROR("Error loading config: {}", e.what());
    }
}

hikcamera::HikCamera::~HikCamera()
{
    // 先停止内部线程，再拆句柄，避免线程在对象析构过程中继续访问 handle
    stopThreads();

    closeCamera();

    // 仅初始化后才需要 finalize
    if (sdk_initialized_)
    {
        const int nRet = MV_CC_Finalize();
        if (nRet != MV_OK)
        {
            MAS_LOG_ERROR("Finalize SDK fail! nRet 0x{0:x}", nRet);
        }
        sdk_initialized_ = false;
    }
}

bool hikcamera::HikCamera::openCamera()
{
    std::unique_lock<std::timed_mutex> lock(camera_mutex_, std::chrono::milliseconds(kCameraLockTimeoutMs));

    if (!lock.owns_lock())
    {
        MAS_LOG_ERROR("openCamera: could not acquire camera_mutex_ within {} ms ", kCameraLockTimeoutMs);
        return false;
    }

    // 筛选已经连接且句柄不为空
    if (isConnected && handle != NULL)
    {
        MAS_LOG_WARN("Camera already open, skipping re-open");
        return true;
    }

    // 防御：句柄还在但连接标志已经掉了，先释放旧句柄再重新创建，
    // 避免 CreateHandle 直接覆盖旧指针，造成 SDK 句柄泄漏/重复打开同一设备。
    if (handle != NULL)
    {
        MAS_LOG_WARN("openCamera: stale handle found while disconnected, releasing it first");
        MV_CC_DestroyHandle(handle);
        handle = NULL;
    }

    // 重置连续SDK错误计数
    consecutive_sdk_errors_ = 0;

    int nRet = MV_OK;

    // Initialize SDK(即使已经连接且句柄不为空，也可能是 SDK 尚未初始化)
    if (!sdk_initialized_)
    {
        nRet = MV_CC_Initialize();
        if (nRet != MV_OK)
        {
            MAS_LOG_ERROR("Initialize SDK fail! nRet 0x{0:x}", nRet);
            return false;
        }
        sdk_initialized_ = true;
    }

    // 每次打开都重新推导实际曝光值，避免沿用上一次连接（可能是另一台相机）的残留值
    exposure_actual_us_            = 0.0f;
    // 设置曝光(初始化)，防止goto报错
    MVCC_FLOATVALUE stExposureTime = {0};
    // 设置gain
    MVCC_FLOATVALUE stGain         = {0};
    // 用于读取实际曝光时间
    MVCC_FLOATVALUE stExpActual    = {0};
    // 用于读取/显示当前采集帧率、分辨率与像素格式（诊断带宽是否够用）
    MVCC_FLOATVALUE stFrameRate    = {0};
    MVCC_INTVALUE   stWidth        = {0};
    MVCC_INTVALUE   stHeight       = {0};
    MVCC_ENUMVALUE  stPixelFormat  = {0};

    MV_CC_DEVICE_INFO *pDeviceInfo = NULL;

    // ch:枚举设备
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet)
    {
        MAS_LOG_ERROR("Enum Devices fail! nRet 0x{0:x}", nRet);
        goto clean_init;
    }

    if (stDeviceList.nDeviceNum > 0)
    {
        for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
        {
            MAS_LOG_INFO("[device {}]:", i);
            pDeviceInfo = stDeviceList.pDeviceInfo[i];
            if (NULL == pDeviceInfo)
            {
                break;
            }
            PrintDeviceInfo(pDeviceInfo);
        }
    }
    else
    {
        MAS_LOG_ERROR("Find No Devices!");
        goto clean_init;
    }

    // ch:选择设备并创建句柄,默认选择第一个设备
    pDeviceInfo = stDeviceList.pDeviceInfo[0];

    // 检测USB协议版本
    if (pDeviceInfo->nTLayerType == MV_USB_DEVICE)
    {
        unsigned int nbcdUSB = pDeviceInfo->SpecialInfo.stUsb3VInfo.nbcdUSB;
        if (nbcdUSB < 0x0300) // USB 2.x
        {
            MAS_LOG_WARN("WARNING: Camera is connected via USB 2.0 0x{0:x}! Performance "
                         "may be limited. Please use USB 3.0 port.",
                         nbcdUSB);
        }
        else // USB 3.x
        {
            MAS_LOG_INFO("Camera connected via USB 3.0 0x{0:x}", nbcdUSB);
        }
    }

    nRet = MV_CC_CreateHandle(&handle, pDeviceInfo);
    if (MV_OK != nRet)
    {
        MAS_LOG_ERROR("Create Handle fail! nRet 0x{0:x}", nRet);
        goto clean_init;
    }

    // ch:打开设备 | en:Open device
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet)
    {
        MAS_LOG_ERROR("Open Device fail! nRet 0x{0:x}", nRet);
        goto clean_handle;
    }

    nRet = MV_CC_RegisterExceptionCallBack(handle, &HikCamera::ExceptionCallBack, this);
    if (MV_OK != nRet)
    {
        MAS_LOG_WARN("RegisterExceptionCallBack failed! nRet 0x{0:x}; falling back to heartbeat-only disconnect detection", nRet);
    }

    // 必须关掉自动曝光/自动增益/自动白平衡：
    //  - 自动曝光/增益会让画面亮度随场景自己变化，绿灯的白核/光晕比例随之漂移，检测阈值失效；
    //  - 自动白平衡在绿色主导的画面里会把绿色"拉灰"，S 掉下去，绿色掩码不稳定。
    nRet = MV_CC_SetEnumValue(handle, "ExposureAuto", 0); // 0 = Off
    if (MV_OK != nRet) MAS_LOG_WARN("Set ExposureAuto=Off failed! nRet 0x{0:x}", nRet);

    nRet = MV_CC_SetEnumValue(handle, "GainAuto", 0); // 0 = Off
    if (MV_OK != nRet) MAS_LOG_WARN("Set GainAuto=Off failed! nRet 0x{0:x}", nRet);

    nRet = MV_CC_SetEnumValue(handle, "BalanceWhiteAuto", 0); // 0 = Off
    if (MV_OK != nRet) MAS_LOG_WARN("Set BalanceWhiteAuto=Off failed! nRet 0x{0:x}", nRet);

    // Exposure time
    nRet = MV_CC_GetFloatValue(handle, "ExposureTime", &stExposureTime);
    if (nRet != MV_OK)
    {
        MAS_LOG_WARN("Get ExposureTime failed! nRet 0x{0:x}; use camera default", nRet);
        exposure_actual_us_ = exposure_time_;
    }
    else if (exposure_time_ < stExposureTime.fMin ||
             exposure_time_ > stExposureTime.fMax)
    {
        MAS_LOG_WARN("ExposureTime {} is outside [{}, {}]; use camera default",
                        exposure_time_, stExposureTime.fMin, stExposureTime.fMax);
        exposure_actual_us_ = stExposureTime.fCurValue;
    }
    else
    {
        nRet = MV_CC_SetFloatValue(handle, "ExposureTime", exposure_time_);
        if (nRet != MV_OK)
        {
            MAS_LOG_WARN("Set ExposureTime failed! nRet 0x{0:x}; use camera default", nRet);
            exposure_actual_us_ = stExposureTime.fCurValue;
        }
        else
        {
            exposure_actual_us_ = exposure_time_;
        }

        nRet = MV_CC_GetFloatValue(handle, "ExposureTime", &stExpActual);
        if (nRet == MV_OK) 
        {
            exposure_actual_us_ = stExpActual.fCurValue;
        }
        else
        {
            MAS_LOG_WARN("ExposureTime readback after set failed! nRet 0x{0:x}", nRet);
        }
    }

    // Gain
    nRet = MV_CC_GetFloatValue(handle, "Gain", &stGain);
    if (nRet != MV_OK)
    {
        MAS_LOG_WARN("Get Gain failed! nRet 0x{0:x}; use camera default", nRet);
    }   
    else if (gain_ < stGain.fMin || gain_ > stGain.fMax)
    {
        MAS_LOG_WARN("Gain {} is outside [{}, {}]; use camera default",
                        gain_, stGain.fMin, stGain.fMax);
    }
    else
    {
        nRet = MV_CC_SetFloatValue(handle, "Gain", gain_);
        if (nRet != MV_OK)
        {
            MAS_LOG_WARN("Set Gain failed! nRet 0x{0:x}; use camera default", nRet);
        }
    }

    // ch:设置触发模式为off | en:Set trigger mode as off
    nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    if (MV_OK != nRet)
    {
        MAS_LOG_ERROR("Set Trigger Mode fail! nRet 0x{0:x}", nRet);
        goto clean_device;
    }

    // 采集帧率上限（可选）：frame_rate: 0 表示不修改相机当前设置。
    // USB 2.0 下全分辨率 + 高帧率会吃满带宽，周期性断流多时可降到 15~20。
    if (frame_rate_ > 0.0f)
    {
        nRet = MV_CC_SetBoolValue(handle, "AcquisitionFrameRateEnable", true);
        if (nRet != MV_OK)
        {
            MAS_LOG_WARN("Enable AcquisitionFrameRate fail! nRet 0x{0:x}", nRet);
        }
        else
        {
            nRet = MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", frame_rate_);
            if (nRet != MV_OK)
            {
                MAS_LOG_WARN("Set AcquisitionFrameRate={} fail! nRet 0x{:x}", frame_rate_, nRet);
            }
        }
    }

    // 打印当前分辨率/像素格式/帧率，方便判断 USB 带宽是否够用（读取失败不影响取流）
    if (MV_CC_GetIntValue(handle, "Width", &stWidth) == MV_OK &&
        MV_CC_GetIntValue(handle, "Height", &stHeight) == MV_OK)
    {
        MAS_LOG_INFO("Camera resolution = {}x{}", stWidth.nCurValue, stHeight.nCurValue);
    }
    if (MV_CC_GetEnumValue(handle, "PixelFormat", &stPixelFormat) == MV_OK)
    {
        MAS_LOG_INFO("Camera PixelFormat = 0x{:x}", stPixelFormat.nCurValue);
    }
    if (MV_CC_GetFloatValue(handle, "AcquisitionFrameRate", &stFrameRate) == MV_OK)
    {
        MAS_LOG_INFO("Camera AcquisitionFrameRate = {:.1f} fps", stFrameRate.fCurValue);
    }

    // 设置缓存节点数量
    nRet = MV_CC_SetImageNodeNum(handle, 1);
    if (MV_OK != nRet)
    {
        MAS_LOG_ERROR("SetImageNodeNum fail! nRet 0x{0:x}", nRet);
    }

    // 设置抓取策略为最新图像
    nRet = MV_CC_SetGrabStrategy(handle, MV_GrabStrategy_LatestImagesOnly);
    if (MV_OK != nRet)
    {
        MAS_LOG_ERROR("SetGrabStrategy fail! nRet 0x{0:x}", nRet);
    }

    // ch:开始取流
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet)
    {
        MAS_LOG_ERROR("Start Grabbing fail! nRet 0x{0:x}", nRet);
        goto clean_device;
    }

    isConnected = true;

    {
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();

        // 心跳清零
        last_frame_tick_ms_.store(now_ms, std::memory_order_relaxed);

        t_grab_started_ms_.store(now_ms, std::memory_order_relaxed);
    }

    stats_reset_pending_.store(true, std::memory_order_relaxed);
    first_frame_pending_.store(true, std::memory_order_relaxed);

    // 启动内部采集/守护线程；若已在运行则为 no-op
    startThreads();
    return true;

    clean_device:
        MV_CC_CloseDevice(handle);
    clean_handle:
        MV_CC_DestroyHandle(handle);
    clean_init: 
        handle = NULL;
        isConnected = false;

    startThreads();
    return false;
}

bool hikcamera::HikCamera::closeCamera()
{
    std::unique_lock<std::timed_mutex> lock(camera_mutex_,
                                            std::chrono::milliseconds(kCameraLockTimeoutMs));

    if (!lock.owns_lock())
    {
        MAS_LOG_ERROR("closeCamera: could not acquire camera_mutex_ within {} ms "
                     "(capture thread likely stuck in a blocking SDK call); close NOT performed",
                     kCameraLockTimeoutMs);
        return false;
    }

    if (handle == NULL)
    {
        isConnected = false;
        return true;
    }

    auto t_close_start = std::chrono::steady_clock::now();
    int nRetStop       = MV_CC_StopGrabbing(handle);
    int nRetClose      = MV_CC_CloseDevice(handle);

    if (nRetStop != MV_OK)
    {
        MAS_LOG_WARN("closeCamera: StopGrabbing fail! nRet 0x{0:x}", nRetStop);
    }
    if (nRetClose != MV_OK)
    {
        MAS_LOG_WARN("closeCamera: CloseDevice fail! nRet 0x{0:x}", nRetClose);
    }

    // 无论上面几步是否报错，句柄都已经释放，后续只能重新 open
    isConnected = false;
    MV_CC_DestroyHandle(handle);
    handle = NULL;

    auto close_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t_close_start)
                            .count();
    MAS_LOG_INFO("closeCamera: teardown took {} ms", close_ms);
    return true;
}

void hikcamera::HikCamera::startThreads()
{
    if (running_.exchange(true))
    {
        return; // 已在运行，避免重复创建线程
    }
    capture_thread_ = std::thread(&HikCamera::captureLoop, this);
    daemon_thread_  = std::thread(&HikCamera::daemonLoop, this);
    MAS_LOG_INFO("Capture/daemon threads started");
}

void hikcamera::HikCamera::stopThreads()
{
    if (!running_.exchange(false))
    {
        return; // 未在运行
    }
    queue_cv_.notify_all();                 // 唤醒可能阻塞在 getImage() 里的消费者

    if (capture_thread_.joinable())         // 等待采集线程退出
    {
        capture_thread_.join();
    }
    if (daemon_thread_.joinable())          // 等待守护线程退出
    {
        daemon_thread_.join();
    }
    MAS_LOG_INFO("Capture/daemon threads stopped");
}

void hikcamera::HikCamera::markDisconnected(std::chrono::steady_clock::time_point now_steady)
{
    if (isConnected.exchange(false, std::memory_order_relaxed))
    {
        t_disconnect_signal_ms_.store(
            std::chrono::duration_cast<std::chrono::milliseconds>(now_steady.time_since_epoch()).count(),
            std::memory_order_relaxed);
    }
}

void hikcamera::HikCamera::ExceptionCallBack(unsigned int nMsgType, void *pUser)
{
    auto *self = static_cast<HikCamera *>(pUser);
    if (self == nullptr)
    {
        return;
    }

    if (nMsgType == MV_EXCEPTION_DEV_DISCONNECT)
    {
        MAS_LOG_WARN("HikCamera: SDK exception callback reports device disconnect (0x{0:x}); "
                     "marking as disconnected so daemonLoop reconnects on its next poll",
                     nMsgType);
        self->markDisconnected(std::chrono::steady_clock::now());
    }
    else
    {
        MAS_LOG_WARN("HikCamera: SDK exception callback reports nMsgType 0x{0:x}", nMsgType);
    }
}

bool hikcamera::HikCamera::PrintDeviceInfo(MV_CC_DEVICE_INFO *pstMVDevInfo)
{
    if (NULL == pstMVDevInfo)
    {
        MAS_LOG_ERROR("The Pointer of pstMVDevInfo is NULL!\n");
        return false;
    }
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
    {
        int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
        int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
        int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
        int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

        // ch:打印当前相机ip和用户自定义名字
        MAS_LOG_INFO("CurrentIp: {}.{}.{}.{}", nIp1, nIp2, nIp3, nIp4);
        MAS_LOG_INFO("UserDefinedName: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName));
    }
    else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
    {
        MAS_LOG_INFO("UserDefinedName: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName));
        MAS_LOG_INFO("Serial Number: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber));
        MAS_LOG_INFO("Device Number: {}", pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber);
    }
    else if (pstMVDevInfo->nTLayerType == MV_GENTL_GIGE_DEVICE)
    {
        MAS_LOG_INFO("UserDefinedName: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName));
        MAS_LOG_INFO("Serial Number: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stGigEInfo.chSerialNumber));
        MAS_LOG_INFO("Model Name: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName));
    }
    else if (pstMVDevInfo->nTLayerType == MV_GENTL_CAMERALINK_DEVICE)
    {
        MAS_LOG_INFO("UserDefinedName: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stCMLInfo.chUserDefinedName));
        MAS_LOG_INFO("Serial Number: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stCMLInfo.chSerialNumber));
        MAS_LOG_INFO("Model Name: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stCMLInfo.chModelName));
    }
    else if (pstMVDevInfo->nTLayerType == MV_GENTL_CXP_DEVICE)
    {
        MAS_LOG_INFO("UserDefinedName: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stCXPInfo.chUserDefinedName));
        MAS_LOG_INFO("Serial Number: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stCXPInfo.chSerialNumber));
        MAS_LOG_INFO("Model Name: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stCXPInfo.chModelName));
    }
    else if (pstMVDevInfo->nTLayerType == MV_GENTL_XOF_DEVICE)
    {
        MAS_LOG_INFO("UserDefinedName: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stXoFInfo.chUserDefinedName));
        MAS_LOG_INFO("Serial Number: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stXoFInfo.chSerialNumber));
        MAS_LOG_INFO("Model Name: {}", reinterpret_cast<const char *>(pstMVDevInfo->SpecialInfo.stXoFInfo.chModelName));
    }
    else
    {
        MAS_LOG_ERROR("Not support.");
    }

    return true;
}
