# 数字仪表读数系统
基于双阶段 YOLO 的端到端数字仪表自动读数系统。面向无人机巡检场景，具备抗抖动、抗视角倾斜能力。

---

## 系统架构
```
视频流->前端过滤->阶段一OBB定位->OBB横向拓展->裁剪仪表区域->CLAHE增强->阶段二单字识别->时序投票筛选->输出
```
---

## 目录结构

```
Yolo_digital_meter/
├── dataset/                    # 原始数据集（用户提供，YOLO 标准格式）
│   └── <your_dataset>/
│       ├── train/images/       # 训练图片
│       ├── train/labels/       # 训练标签（含 class_id=2 大框 + 单字小框）
│       ├── valid/...
│       └── test/...
├── dataset_obb/                # 由 prepare_obb_dataset.py 生成的 OBB 屏幕定位数据集
├── dataset_digits/             # 由 prepare_digit_dataset.py 生成的单字数据集
├── runs/                       # 训练产物
│   ├── obb/train/weights/      # 阶段一模型权重
│   └── digit/train/weights/    # 阶段二模型权重
├── debug/                      # 调试工具与输出
│   ├── debug_visualize_yolo.py # 双 YOLO 可视化调试
│   └── output_yolo/            # 调试输出图片
├── inference_pipeline.py       # 核心 Pipeline
├── prepare_obb_dataset.py      # 从原始数据集生成 OBB 屏幕定位数据集
├── prepare_digit_dataset.py    # 从原始数据集生成单字训练集
├── train_obb.py                # 阶段一：OBB 屏幕定位训练
├── train_digit.py              # 阶段二：单字检测训练
├── export_engine.py            # 模型导出（TensorRT / ONNX）
└── README.md
```

---

## 环境搭建

```bash
conda create -n d2l python=3.9 -y
conda activate d2l

#深度学习环境根据显卡配置

# ------- 一般平台  X86_64 架构 ------------
# 安装 PyTorch
pip install torch==1.13
pip install torchvision==0.14.1
#安装 TensorRT 及其必备依赖     可选，用于边缘设备推理加速
pip install tensorrt==8.2.5.1  
#-----------------------------------------

#--------- Jetson 平台 边缘设备  ------------
# torch 使用whl文件
wget [https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl](https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl)
pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
# tochvision需要下载源码，手动编译
git clone http://github.com/pytorch/vision.git
git checkout v0.14.1
cd vision
#脚本编译
python setup.py install

# 配置 TensorRT (将系统底层预装的 TensorRT 软链接至 Conda 虚拟环境中) 
#注意根据实际路径对应修改
ln -s /usr/lib/python3.8/dist-packages/tensorrt* ~/miniconda3/envs/d2l/lib/python3.8/site-packages/
ln -s /usr/lib/python3.8/dist-packages/uff* ~/miniconda3/envs/d2l/lib/python3.8/site-packages/
ln -s /usr/lib/python3.8/dist-packages/graphsurgeon* ~/miniconda3/envs/d2l/lib/python3.8/site-packages/
#-------------------------------------------
 

# 安装 Ultralytics
pip install Ultralytics

#ONNX依赖
pip install onnx==1.12.0  
pip install onnxsim


#其他常规依赖
pip install tqdm pyyaml opencv-python
```

---

## 快速开始

### 1. 准备数据集

将你的数据集放入 `dataset/` 目录。要求标准 YOLO 格式，标签中包含：

- **`class_id=2`** 的行代表数显区域大框（正框）
- 其余行代表单个数字字符（class_id 对应原始 names 列表中的数字）

原始数据集 `data.yaml` 示例：

```yaml
train: train/images
val: valid/images
test: test/images
nc: 11
names: ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
#       ↑    ↑    ↑       ← class_id=2 即 '10' 是大框，其余是单字
```

标签文件 `.txt` 示例：

```
0 0.378 0.362 0.031 0.043   ← 单字 '0'
6 0.421 0.376 0.034 0.050   ← 单字 '5'
2 0.381 0.371 0.216 0.115   ← 大框 (class_id=2, w/h 明显更大)
```

### 2. 生成训练数据集

```bash
#默认为增量添加，可选择--clean参数覆盖原有数据集
# 生成 OBB 屏幕定位数据集（提取大框，转为 OBB 四角点格式）
python prepare_obb_dataset.py --src dataset/<your_dataset> --dst dataset_obb

# 生成单字检测数据集（裁剪大框区域，重映射单字标签）
python prepare_digit_dataset.py --src dataset/<your_dataset> --dst dataset_digits
```

```bash
python prepare_digit_dataset.py --src dataset/<your_dataset> --dst dataset_digits
```

脚本会自动：
- 按 `class_id=2` 大框裁剪数显区域
- 将单字标签坐标重映射到裁剪图
- 修正 class_id 为标准 0-9
- 在 `dataset_digits/` 下生成 `data.yaml`

### 3. 训练双模型

```bash
#可添加--exist_ok覆盖训练结果，默认为增量更新
# 阶段一：OBB 屏幕定位
python train_obb.py --data dataset_obb/data.yaml

# 阶段二：单字检测
python train_digit.py --data dataset_digits/data.yaml
```

训练产物分别保存在 `runs/obb/` 和 `runs/digit/`

### 4. 导出模型

支持直接导出engine模型，tensorrt加速推理
```bash
python export_engine.py --weights runs/obb/train/weights/best.pt --imgsz 640
python export_engine.py --weights runs/digit/train/weights/best.pt --imgsz 416
```

### 5. 推理调用

```python
from inference_pipeline import DigitalMeterPipeline

# 初始化（默认不保存文件，CLAHE 增强默认开启）
pipeline = DigitalMeterPipeline(
    obb_weights="runs/obb/train/weights/best.pt",
    digit_weights="runs/digit/train/weights/best.pt",
    enhance_enabled=True,   # CLAHE 增强预处理（默认开启，可设 False 对比）
)

# 处理单帧，返回读数字符串与 OBB 中心像素坐标
reading, obb_center = pipeline.process_frame(frame)
print(reading, obb_center)  # 例如 "33442" (120, 88)

# 可选：开启文件记录
pipeline_with_log = DigitalMeterPipeline(
    obb_weights="runs/obb/train/weights/best.pt",
    digit_weights="runs/digit/train/weights/best.pt",
    enhance_enabled=True,   # CLAHE 增强，提升实机光照不均下的文字对比度
    output_dir="results",   # 写入 JSON/CSV
)
```

> **性能优化**：两阶段 YOLO 推理均使用 `torch.no_grad()` 包裹，并立即 `.cpu().item()` 释放显存引用，避免隐式梯度记录导致的显存累积，Jetson 等显存受限平台运行更稳定。

### 6. 可视化调试

```bash
# 默认开启 CLAHE 增强（与主 Pipeline 一致）
python debug/debug_visualize_yolo.py \
    --source dataset/<your_dataset>/test/images/ \
    --show

# 关闭增强，用于 A/B 对比增强前后效果
python debug/debug_visualize_yolo.py \
    --source test.jpg \
    --no-enhance
```

输出四阶段对比图到 `debug/output_4/`：
- 左上：原图 + OBB 检测框
- 右上：裁剪的数显长条（**原始裁剪**）
- 左下：增强后图像 + 单字检测框 + 类别标签（**真实推理输入**）
- 右下：拼接读数结果

| 参数 | 说明 |
|------|------|
| `--enhance` | 启用 CLAHE 增强预处理（默认开启） |
| `--no-enhance` | 关闭增强，便于 A/B 对比效果 |

---

## API 参考

### `DigitalMeterPipeline`

```python
pipeline = DigitalMeterPipeline(
    obb_weights="...",          # OBB 模型权重
    digit_weights="...",        # 单字检测权重
    blur_threshold=50.0,        # 拉普拉斯模糊阈值
    obb_conf=0.5,               # OBB 置信度阈值
    digit_conf=0.3,             # 单字检测置信度阈值
    obb_imgsz=640,              # OBB 输入尺寸
    digit_imgsz=416,            # 单字检测输入尺寸
    voting_window=18,           # 时序投票滑动窗口大小
    stability_threshold=3,      # 连续一致帧数判定稳定
    quorum_ratio=0.5,           # 单位投票最少支持帧比例，低于则该位判为不可靠
    min_consistency=0.6,        # 窗口内位数一致比例下限，低于则整体判为不可靠
    enhance_enabled=True,       # CLAHE 裁剪图增强预处理（默认开启）
    obb_expand_ratio=0.4,       # OBB 框横向拓展比例（默认 0.4，左右各 20%）
    output_dir=None,            # None=不保存文件, "results"=写 JSON/CSV
    device="0",                 # 推理设备
)

reading, obb_center = pipeline.process_frame(frame, uav_lat=0, uav_lon=0, uav_alt=0)
# 返回: (reading, obb_center)
#   reading: 读数字符串 或 None
#   obb_center: OBB 中心像素坐标 (cx, cy) 或 None
```

### `enhance_crop`

CLAHE 裁剪图增强函数

```python
from inference_pipeline import enhance_crop

# 对裁剪图做 LAB 颜色空间的亮度通道 CLAHE 均衡
enhanced = enhance_crop(crop_img, enable=True, clip_limit=2.0, tile_grid=8)
# enable=False 时原样返回，便于 A/B 对比
```

### `ScreenDetector`

阶段一 OBB 检测器

```python
detector = ScreenDetector(weights="best.pt", device="0", imgsz=640, conf=0.5)
corners, crop = detector.detect(frame)
# corners: (4,2) 归一化角点, crop: 裁剪的数显区域图像
```

### `DigitDetector`

阶段二 单字 检测器

```python
detector = DigitDetector(weights="best.pt", device="0", imgsz=416, conf=0.3)
reading, avg_conf, digit_count = detector.detect(crop_image)
# reading: "33442", avg_conf: 0.92, digit_count: 5
```

---

## 数据集规范

### 原始数据集要求

用户提供的数据集需满足：

1. **目录结构**：标准 YOLO 格式
   ```
   <dataset>/
   ├── train/images/*.jpg
   ├── train/labels/*.txt
   ├── valid/images/*.jpg
   ├── valid/labels/*.txt
   └── test/images/*.jpg, test/labels/*.txt
   ```

2. **标签格式**：每行 `class_id x_c y_c w h`（YOLO 正框，归一化坐标）

3. **大框标注**：必须有一个 `class_id` 专门表示数显区域的大框（默认 `class_id=2`）

4. **单字标注**：其余 `class_id` 代表 0-9 数字字符

5. **data.yaml**：包含正确的 `names` 列表，大框类别名建议为 `'10'` 或类似标识

### OBB 屏幕定位数据集

由 `prepare_obb_dataset.py` 自动生成，包含：
- `nc: 1`，`names: ['digital_screen']`
- 图片为原图 symlink，标签仅保留大框行并转为 OBB 四角点格式

### 单字检测数据集

由 `prepare_digit_dataset.py` 自动生成，包含：
- `nc: 10`，`names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']`
- 图像为裁剪后的数显长条，标签已重映射

---

## 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `blur_threshold` | 50.0 | 降低保留更多帧；升高过滤更严格 |
| `obb_conf` | 0.5 | OBB 检出阈值 |
| `digit_conf` | 0.3 | 单字检出阈值，建议 0.3-0.5 |
| `voting_window` | 18 | 时序投票滑动窗口大小，增大更稳定但延迟高 |
| `stability_threshold` | 3 | 连续一致帧数判定稳定，必须小于 voting_window |
| `quorum_ratio` | 0.5 | 单位（单个数字位）投票最少支持帧比例（0-1），低于则该位判为不可靠 |
| `min_consistency` | 0.6 | 窗口内位数一致的帧比例下限（0-1），低于则整体判为不可靠 |
| `obb_imgsz` | 640 | obb模型输入图像尺寸 |
| `digit_imgsz` | 416 | 单字识别输入尺寸 |
| `enhance_enabled` | True | CLAHE 裁剪图增强预处理；设 False 可关闭对比效果 |
| `obb_expand_ratio` | 0.4 | OBB 框横向拓展比例（每侧 20%，整体宽度 +40%），避免裁掉边缘数字 |

### 增强预处理说明

`enhance_crop()` 采用 **LAB 颜色空间 + CLAHE**：
- 仅对亮度通道 L 做自适应直方图均衡化，**避免颜色畸变**
- 提升数显 LCD 文字对比度，对**实机环境光照不均、反光**更鲁棒
- 默认开启，可通过 `enhance_enabled=False` 或调试脚本 `--no-enhance` 关闭以 A/B 对比