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

  `--camera` 模式下挥动棋盘格即可：工具会自动挑选位置/姿态差别足够大的帧，
  采满 `--max`（默认 40）张后自动计算，按 `q` / `ESC` 可提前结束。
  整个过程只在内存里，`out/photos` 不会多出任何文件。

  采样阈值可以现场调：

  - `--min-move 40`：当前帧与上一张相比，角点至少移动多少像素才收（默认 40px，
    约等于 1280 宽画面的 3%）；平移、倾斜、靠近/远离都会让角点移动；
  - `--interval 400`：两次采样最小间隔（毫秒），防止手抖连拍；
  - `--no-window`：不弹预览窗口，纯自动采样；
  - `--max 20`：采 20 张就停，一般 15~20 张足够。

- 或任意 OpenCV 棋盘格标定脚本，输出内参矩阵后填入即可。

> 即使不标定，先用 `fx` 约等于焦距（像素）的估计值也能跑通方向，
> 只是角度绝对值不精确；建议尽快用真实内参替换。
