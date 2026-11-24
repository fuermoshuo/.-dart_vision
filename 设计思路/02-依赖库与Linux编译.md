# 02 依赖库与 Linux 编译

目标平台为 Linux（Jetson / 工控机，x86_64 或 aarch64）。以下命令基于 Ubuntu/Debian。

## 1. 基础工具与系统库

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake build-essential ninja-build git pkg-config libssl-dev \
    clang clangd clang-format libspdlog-dev \
    libgtk2.0-dev \
    libsdl2-2.0-0 libsdl2-dev libsdl2-image-2.0-0 libsdl2-image-dev \
    libsdl2-ttf-2.0-0 libsdl2-ttf-dev libopencv-dev
```

> 说明：本模块已移除 Eigen、nlohmann-json、Ceres 依赖，所以无需再装它们。
> `libopencv-dev` 用系统自带即可；若系统源版本过旧，可改用源码编译 OpenCV 4.x。

## 2. 海康相机驱动（MVS）

1. 到海康机器人官网下载对应平台（x86_64 / arm）的 MVS 驱动：
   <https://www.hikrobotics.com/cn/machinevision/service/download/?module=0>
2. 安装 deb 包（版本号以实际为准）：
   ```bash
   sudo dpkg -i MVS-4.6.1_x86_64_20251217.deb
   ```
3. 安装后 SDK 默认位于 `/opt/MVS`，`hardware/hikcamera/CMakeLists.txt` 里已按此路径配置。
   若安装到其它位置，修改该 CMakeLists 中的 `MVS_ROOT`。

## 3. yaml-cpp

```bash
git clone https://gitee.com/mirrors/yaml-cpp.git
cd yaml-cpp
mkdir build && cd build
cmake .. && make -j8 && sudo make install
```

> `find_package(yaml-cpp REQUIRED)` 会找到安装后的 `yaml-cpp-config.cmake`，
> 它同时提供 `yaml-cpp::yaml-cpp` 目标与 `${YAML_CPP_LIBRARIES}` 变量，两种写法都可链接。

## 4. 编译

```bash
cd dart_vision
mkdir build && cd build
cmake ..
make -j8
```

### 换机器 / 从 U 盘拷贝后：必须重新编译

**不要拷贝 `build/`、`build_camera/`、`build_calib/`。**
`CMakeCache.txt` 里记的是上一台机器的绝对路径，且不同 OpenCV 版本
（`libopencv_core.so.4.5` vs `4.10`）的可执行文件不能直接互换，
拷过去大概率是 `error while loading shared libraries` 或链接到错误的库。

U 盘只带源码和配置：

```
CMakeLists.txt  apps/  dart_aim/  hardware/  rm_utils/  threads/
scripts/  config/  设计思路/  README.MD  .gitignore  .gitattributes
```

不带：`build*/`、`out/`、`logs/`、`.vs/`。最省事的方式是用 git 打包：

```bash
git archive --format=zip HEAD -o dart_vision.zip
```

到新机器后：

```bash
unzip dart_vision.zip -d ~/dart_vision
cd ~/dart_vision

# 1. 确认依赖和 OpenCV 版本
pkg-config --modversion opencv4
pkg-config --modversion yaml-cpp spdlog
ls /opt/MVS/include/MvCameraControl.h

# 2. 全新编译
rm -rf build build_camera build_calib
mkdir build && cd build
cmake ..
make -j$(nproc)

# 3. 确认链接的是本机 OpenCV
ldd ./dart_calibrate | grep -i opencv
```

- OpenCV 版本不同**不用改代码**，重新编译会自动链接新版本；
- 系统里有多个 OpenCV 时，显式指定：
  `cmake .. -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4`；
- OpenCV 3.x 也能编（`dart_calibrate` 里对 `findChessboardCornersSB`
  做了版本保护），但仍建议 4.x；
- 经 Windows U 盘（exFAT/NTFS）拷贝后，脚本的可执行位可能丢失：
  `chmod +x scripts/*.sh`；
- 如果 `.sh` 在 Windows 编辑器里改过，确认换行符仍是 LF。

## 5. 运行

```bash
cd build
./dart_vision
```

- 程序会生成 `logs/dart_vision.log` 日志；
- `config/` 在 build 目录下以软链接形式可用；
- 需要先保证相机已连接、串口设备存在（见下）。

## 6. 串口设备准备

飞镖架通过串口（USB 转串口 / 板载 UART）连接。项目提供两个脚本：

```bash
# 生成 /dev/mascdc 软链接（自动识别设备号）
sudo ./scripts/setup_serial_device.sh

# 开机自启（可选，按需修改后使用）
sudo ./scripts/mas_auto_start.sh
```

默认串口配置在 `config/serial_config.yaml`：`/dev/mascdc`、115200、8N1。

> `/dev/mascdc` 是 udev 软链接，设备名可能变成 `ttyACM0` / `ttyUSB0`，
> 报错码、排查步骤、稳定路径（`/dev/serial/by-id/...`）见
> [04-串口通信协议.md 第 5 节](04-串口通信协议.md#5-串口设备与联调排查踩坑记录)。

## 7. 相机标定（得到 fx / cx）

本模块只用到内参 `fx`、`cx`（和图像宽），把棋盘格标定结果填入
`config/hikcamera.yaml` 的 `calibration.camera_matrix`：

- `fx = camera_matrix[0]`
- `cx = camera_matrix[2]`

标定可用：
- 原自瞄代码里的 `./base calibrate_camera` 命令；
- 项目自带的 C++ 工具 `dart_calibrate`（只用 OpenCV，不需要 Python）：

  ```bash
  cd build
  cmake .. -DDART_VISION_OFFLINE=ON      # 只要标定工具的话，离线模式即可
  make -j8 dart_calibrate

  # 输入目录（运行主程序后按 s 存到 out/photos）
  ./dart_calibrate ../out/photos

  # 或者直接喂一段棋盘格视频，按 --step 抽帧，不用事先存图
  ./dart_calibrate ../chessboard.mp4 --step 15

  # 或者直接连海康相机实时采样：不写任何图片/视频，结果只打印
  # 需要完整模式编译（不加 DART_VISION_OFFLINE）并装好 MVS SDK
  ./dart_calibrate --camera

  # 棋盘格内角点不是 9x6 时
  ./dart_calibrate ../out/photos --pattern 8x5 --square 0.03

  # 用圆点标定板（calib.io 那种）时：pattern = 圆点列数 x 行数
  # 例如 11x10、间距 10mm、直径 6mm 的板子：
  ./dart_calibrate --camera --grid circles --pattern 11x10 --square 0.01
  ```

  工具会打印 RMS 重投影误差和一段可直接粘贴的 `calibration:`，复制进
  `config/hikcamera.yaml` 即可。图片必须是原始分辨率，不能是聊天软件另存过的。

  `--camera` 是**手动采样**模式：

  - 窗口里会一直显示 `FPS`、`samples N/M` 和检测状态；
  - 绿色 `BOARD OK` 表示当前帧识别到了标定板，这时按 **`s`** 采一张；
  - 红色 `NO BOARD` 时按 `s` 不会采样，并会在窗口里提示；
  - 检测短暂丢失时，角点会保留 800ms 并显示 `BOARD LOST`，避免画面一直闪；
  - 采满 `--max` 张（实时模式不写 `--max` 时默认 20）后自动计算；
  - 按 `q` / `ESC` 提前结束；
  - 整个过程只在内存里，`out/photos` 不会多出任何文件。

  **没有角度/位移阈值**：你看着窗口，觉得这一帧姿态好就按 `s`，不会再出现
  "摆了半天角度也不采样"的情况。

  `--no-window` 用于没有显示环境的情况：没有键盘可按，改为检测到标定板后
  每 1 秒自动采一张，采满即结束。

- 或任意 OpenCV 棋盘格标定脚本，输出内参矩阵后填入即可。

> 即使不标定，先用 `fx` 约等于焦距（像素）的估计值也能跑通方向，
> 只是角度绝对值不精确；建议尽快用真实内参替换。
