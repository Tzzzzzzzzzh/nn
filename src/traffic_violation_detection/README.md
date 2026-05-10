# CARLA 交通标志自动采集与违章检测系统

基于 CARLA 模拟器的交通标志数据集自动化采集与交通违章检测工具，支持多天气场景切换、自动驾驶巡航、交通标志实时检测、VOC 格式标注自动生成，以及超速和红灯违章信息记录。

本项目适用于交通标志识别、驾驶违章检测、自动驾驶视觉感知等相关模型训练数据集的制作。

---

## 📋 项目简介

本项目基于 **CARLA 0.9.11** 自动驾驶仿真平台，实现交通标志数据的自动化采集与标注，并扩展了车辆违章检测功能。

主要功能包括：

- 自动巡航采集道路场景，无需持续手动驾驶
- 晴天、雨天、雾天、夜间 4 种天气自动循环切换
- 自动检测交通标志并生成 VOC 格式 XML 标注文件
- 基于 CARLA 仿真真值判断交通灯状态，降低红灯误检风险
- 实时检测车辆速度、超速状态和红灯违章状态
- 支持自动驾驶模式和手动控制模式切换
- 相机画面实时可视化，数据自动保存

---

## ✨ 核心功能

1. **自动化数据采集**  
   车辆在 CARLA 场景中自动驾驶巡航，自动捕获交通标志图像。

2. **多天气场景切换**  
   支持晴天、雨天、雾天、夜间 4 种天气自动循环切换，增强数据多样性。

3. **交通标志自动标注**  
   根据 CARLA 场景中的交通标志 3D 包围框，投影计算图像中的 2D 边界框，并生成 VOC 标准 XML 标注文件。

4. **交通标志框后处理**  
   使用距离过滤、视野范围过滤、面积过滤、宽高比过滤和 NMS 非极大值抑制，减少无效框和重复框。

5. **超速检测与统计**  
   实时计算车辆速度，当速度超过设定阈值时，记录超速违章，并统计超速次数。

6. **红灯违章检测优化**  
   红灯判断由原来的图像红色像素检测方式，改为基于 CARLA 交通灯状态的仿真真值判断。只有当车辆处于交通灯影响区域、交通灯为红色、车辆仍在移动时，才判定为红灯违章。

7. **自动驾驶 / 手动控制模式切换**  
   支持按 `M` 键在自动驾驶模式和手动控制模式之间切换，避免自动驾驶和键盘控制同时作用造成控制冲突。

8. **同步仿真运行**  
   CARLA 使用同步模式运行，保证传感器数据和仿真帧之间的稳定对应关系。

9. **资源安全清理**  
   程序退出时会销毁车辆、相机和背景车辆，并恢复 CARLA 原始仿真设置。

---

## 🧰 环境依赖

### 基础环境

- Windows 系统
- Python 3.7
- CARLA 0.9.11（WindowsNoEditor 版本）

### Python 库安装

```bash
pip install pygame opencv-python numpy
```

### CARLA Python API

如果运行时出现：

```bash
ModuleNotFoundError: No module named 'carla'
```

需要在代码开头加入本地 CARLA egg 文件路径，例如：

```python
import sys
sys.path.append(r"D:\CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.11-py3.7-win-amd64.egg")
```

请根据自己电脑中 CARLA 的实际安装路径修改。

---

## 📁 项目结构

```text
项目根目录/
├── OutPut/
│   └── data01/                         # 自动生成：图片 + XML 标注文件
├── src/
│   └── traffic_violation_detection/
│       ├── traffic_violation_detection.py
│       └── README.md
└── requirements.txt
```

---

## 🚀 快速开始

### 1. 启动 CARLA 模拟器

进入 CARLA 根目录，运行：

```bash
CarlaUE4.exe
```

等待 CARLA 场景窗口完全启动后，再运行 Python 程序。

---

### 2. 检查 CARLA Python API

在终端中测试：

```bash
python -c "import carla; print('carla import ok')"
```

如果提示 `carla import ok`，说明环境配置正常。

如果提示找不到 `carla` 模块，需要将 CARLA egg 路径加入代码或 Python 环境变量。

---

### 3. 运行采集程序

在项目根目录下运行：

```bash
python src/traffic_violation_detection/traffic_violation_detection.py
```

---

### 4. 程序运行说明

程序启动后会自动完成以下操作：

- 加载 `Town05` 地图
- 生成主驾驶车辆
- 生成背景车辆
- 创建车载 RGB 相机
- 默认开启自动驾驶模式
- 自动切换晴天、雨天、雾天和夜间场景
- 实时检测交通标志并绘制边界框
- 实时显示车辆速度、交通灯状态、超速次数和红灯违章次数
- 检测到交通标志后，自动保存图像和 VOC XML 标注文件

---

## 🎮 控制快捷键

| 按键 | 功能 |
|------|------|
| `M`  | 切换自动驾驶模式 / 手动控制模式 |
| `W`  | 手动模式下前进或加速 |
| `S`  | 手动模式下刹车 |
| `A`  | 手动模式下左转 |
| `D`  | 手动模式下右转 |
| `R`  | 手动模式下切换倒车模式 |
| `X`  | 退出程序并清理资源 |

说明：

- 程序默认处于 **AUTO 自动驾驶模式**。
- 在自动驾驶模式下，车辆由 CARLA Traffic Manager 控制，键盘不会直接控制车辆。
- 按 `M` 键切换到 **MANUAL 手动控制模式** 后，程序会关闭 autopilot，并使用键盘控制车辆。
- 再次按 `M` 键可切换回自动驾驶模式。

---

## 📊 数据集格式

采集结果默认保存到：

```text
项目根目录/OutPut/data01/
```

每次检测到交通标志后，程序会保存：

- 一张 PNG 图像
- 一个对应的 VOC XML 标注文件

---

### 1. 图像文件

图像格式：

```text
image_时间戳.png
```

默认图像参数：

| 参数 | 默认值 |
|------|--------|
| 分辨率 | 1024 × 1024 |
| 格式 | PNG |
| 相机类型 | RGB Camera |
| 视场角 | 70° |

---

### 2. XML 标注文件

XML 文件包含以下信息：

| 字段 | 含义 |
|------|------|
| `filename` | 图像文件名 |
| `size` | 图像宽度、高度和通道数 |
| `weather` | 当前天气类别 |
| `object` | 交通标志目标框 |
| `bndbox` | 交通标志边界框坐标 |
| `violation` | 当前帧车辆违章信息 |

天气类别说明：

| 数值 | 含义 |
|------|------|
| `0` | 雨天 |
| `1` | 晴天 |
| `2` | 雾天 |
| `3` | 夜间或其他弱光场景 |

违章信息字段说明：

| 字段 | 含义 |
|------|------|
| `speeding` | 是否超速 |
| `red_light` | 是否红灯违章 |
| `ignore_sign` | 是否忽略交通标志，当前预留 |
| `current_speed` | 当前车辆速度，单位 km/h |
| `traffic_light_state` | 当前交通灯状态，如 Red、Yellow、Green |
| `at_traffic_light` | 车辆是否处于交通灯影响区域 |
| `control_mode` | 当前控制模式，AUTO 或 MANUAL |

XML 示例结构：

```xml
<annotation>
    <filename>image_1710000000000.png</filename>
    <size>
        <width>1024</width>
        <height>1024</height>
        <depth>3</depth>
    </size>
    <weather>
        <condition>1</condition>
    </weather>
    <violation>
        <speeding>False</speeding>
        <red_light>False</red_light>
        <ignore_sign>False</ignore_sign>
        <current_speed>35.42</current_speed>
        <traffic_light_state>Green</traffic_light_state>
        <at_traffic_light>False</at_traffic_light>
        <control_mode>AUTO</control_mode>
    </violation>
    <object>
        <name>TrafficSign</name>
        <bndbox>
            <xmin>420</xmin>
            <ymin>210</ymin>
            <xmax>470</xmax>
            <ymax>265</ymax>
        </bndbox>
    </object>
</annotation>
```

---

## 🔴 红灯违章检测逻辑

原始图像颜色检测方法容易受到以下因素干扰：

- 夜间车尾灯
- 红色车辆
- 红色广告牌
- 红色建筑或路牌
- 低光照和雨雾天气

因此，当前版本使用 CARLA 提供的交通灯状态进行判断。

红灯违章判断条件如下：

```python
vehicle.is_at_traffic_light()
vehicle.get_traffic_light_state() == carla.TrafficLightState.Red
speed > RED_LIGHT_SPEED_THRESHOLD
```

也就是说，只有同时满足以下条件时，才判定为红灯违章：

1. 车辆处于交通灯影响区域；
2. 当前交通灯状态为红灯；
3. 车辆速度大于设定阈值。

默认速度阈值为：

```python
RED_LIGHT_SPEED_THRESHOLD = 3.0
```

单位为 km/h。

---

## 🚗 自动驾驶与手动控制逻辑

程序默认启动自动驾驶模式：

```python
vehicle.set_autopilot(True, traffic_manager.get_port())
```

为了避免自动驾驶和键盘控制同时作用，当前版本加入了模式切换机制。

### AUTO 模式

AUTO 模式下：

- 车辆由 CARLA Traffic Manager 控制；
- 不执行键盘 `apply_control`；
- 适合自动采集数据。

### MANUAL 模式

MANUAL 模式下：

- 程序关闭 autopilot；
- 使用键盘控制车辆；
- 适合人工调整车辆位置或测试特定违章场景。

切换方式：

```text
按 M 键切换 AUTO / MANUAL
```

---

## 🔧 自定义配置

可在代码开头修改以下参数：

```python
HOST = "localhost"
PORT = 2000
TM_PORT = 8000

MAP_NAME = "Town05"
NUM_BACKGROUND_VEHICLES = 10

IMAGE_W = 1024
IMAGE_H = 1024
CAMERA_FOV = 70

SPEED_LIMIT = 50.0
RED_LIGHT_SPEED_THRESHOLD = 3.0

DISTANCE_THRESHOLD = 50.0
WEATHER_TRANSITION_INTERVAL = 10
CAPTURE_COOLDOWN = 5
```

参数说明：

| 参数 | 含义 |
|------|------|
| `HOST` | CARLA 服务地址 |
| `PORT` | CARLA 服务端口 |
| `TM_PORT` | Traffic Manager 端口 |
| `MAP_NAME` | 加载的 CARLA 地图 |
| `NUM_BACKGROUND_VEHICLES` | 背景车辆数量 |
| `IMAGE_W` | 相机图像宽度 |
| `IMAGE_H` | 相机图像高度 |
| `CAMERA_FOV` | 相机视场角 |
| `SPEED_LIMIT` | 超速判断阈值，单位 km/h |
| `RED_LIGHT_SPEED_THRESHOLD` | 红灯状态下车辆移动阈值，单位 km/h |
| `DISTANCE_THRESHOLD` | 交通标志检测距离，单位 m |
| `WEATHER_TRANSITION_INTERVAL` | 天气切换间隔，单位 s |
| `CAPTURE_COOLDOWN` | 同一交通标志重复采集冷却时间，单位 s |

---

## 🌦️ 天气类别

当前支持 4 类天气场景：

| 天气名称 | 说明 |
|----------|------|
| `sunny` | 晴天场景 |
| `rainy` | 雨天场景 |
| `foggy` | 雾天场景 |
| `night` | 夜间弱光场景 |

程序会按照设定时间间隔自动循环切换天气：

```python
weather_conditions = ["rainy", "sunny", "night", "foggy"]
```

默认每 10 秒切换一次：

```python
WEATHER_TRANSITION_INTERVAL = 10
```

---

## 🧪 输出效果

运行过程中，OpenCV 窗口会实时显示：

- 当前控制模式：AUTO 或 MANUAL
- 当前车辆速度
- 当前交通灯状态
- 是否处于交通灯影响区域
- 超速违章次数
- 红灯违章次数
- 交通标志检测框

当检测到交通标志时，程序会自动保存图像和 XML 标注文件。

---

## ⚠️ 注意事项

1. **CARLA 版本需要与 Python API 匹配**  
   推荐使用 CARLA 0.9.11 和 Python 3.7。版本不匹配可能导致 `import carla` 失败。

2. **运行代码前必须先启动 CARLA 模拟器**  
   如果 CARLA 没有启动，程序无法连接到 `localhost:2000`。

3. **红灯检测基于 CARLA 交通灯状态**  
   当前版本判断的是“车辆处于红灯影响区域且仍在移动”，不是严格的停止线穿越检测。

4. **如果需要更严格的闯红灯判断**  
   后续可以加入停止线或路口入口线的几何穿越检测。

5. **如果无法生成车辆或相机**  
   可能是出生点被占用，可减少背景车辆数量，或重启 CARLA 后重新运行程序。

6. **低配置电脑建议降低负载**  
   可减少背景车辆数量，或降低相机分辨率。

7. **退出程序建议使用 `X` 键**  
   程序会在退出时清理车辆、相机和背景车辆，并恢复 CARLA 原始同步设置。

---

## 📌 项目用途

本项目可用于：

- 交通标志识别模型训练
- 驾驶违章检测系统数据集制作
- 自动驾驶视觉感知算法验证
- CARLA 仿真数据集自动生成
- 多天气交通场景感知研究
- 课程项目与开源项目功能扩展

---

## 🔄 本次功能改进说明

本版本主要改进如下：

1. 将红灯检测逻辑由图像红色像素阈值判断，改为 CARLA 交通灯状态判断；
2. 新增红灯状态、是否处于交通灯影响区域等信息显示；
3. 新增 `M` 键切换自动驾驶和手动控制模式；
4. 修复自动驾驶和键盘手动控制同时作用的问题；
5. XML 标注文件中新增车辆速度、红灯状态和控制模式等违章信息；
6. 程序退出时恢复 CARLA 原始同步设置，提高程序稳定性。

---