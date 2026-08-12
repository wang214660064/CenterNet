# 双目检测第一阶段：CenterNet 2D检测 + SGBM测距

## 1. 阶段目标

当前阶段不训练新的 3D 网络，先把系统拆成两条可独立验证的链路：

```text
左图 image_2 ──> CenterNet COCO 2D检测 ──> 类别、置信度、二维框
左右图 + P2/P3 ──> StereoSGBM ──> 视差 ──> 深度
二维框 + 深度 ──> 框内稳健统计 ──> 目标距离、相机坐标、风险提示
```

入口为 `src/stereo_kitti_demo.py`，输出检测叠加图、彩色视差图和 JSON。

## 2. 数据目录

```text
data/kitti/
├── training/
│   ├── image_2/       # 左彩色图，已具备
│   ├── image_3/       # 右彩色图，需要补充下载
│   ├── calib/         # 每帧 P2、P3
│   └── label_2/       # 仅用于评估，不参与当前推理
└── testing/
    ├── image_2/
    ├── image_3/
    └── calib/
```

从 KITTI Object Detection 页面下载 `right color images of object data set`。压缩包通常解出 `training/image_3` 和 `testing/image_3`，不要覆盖 `image_2`。

下载完成后可检查：

```bash
find data/kitti/training/image_3 -name '*.png' | wc -l  # 应为 7481
find data/kitti/testing/image_3 -name '*.png' | wc -l   # 应为 7518
```

## 3. 深度计算

KITTI 的图像已经过极线校正，同一物点原则上位于左右图的同一行。脚本从每帧 `P2/P3` 计算焦距、主点差和基线：

```text
baseline = abs(P2[0,3] / P2[0,0] - P3[0,3] / P3[0,0])
depth = fx * baseline / (disparity - principal_point_offset)
```

SGBM 在无纹理、反光、遮挡边界和远距离处会失效。脚本不会取整个二维框的深度平均值，而是取框内中下部中心区域，过滤无效值和离群值后计算中位数。

## 4. 运行

在项目根目录执行：

```bash
python src/stereo_kitti_demo.py \
  --split training \
  --image-id 000008
```

程序默认优先使用CUDA，没有CUDA时自动回退CPU。默认输出到：

```text
exp/stereo_stage1/training_000008_detections.jpg
exp/stereo_stage1/training_000008_disparity.jpg
exp/stereo_stage1/training_000008_results.json
```

常用参数：

- `--score-thresh`：CenterNet 检测阈值，默认 0.35；
- `--num-disparities`：SGBM 搜索范围，必须为 16 的倍数；
- `--block-size`：匹配窗口，必须为正奇数；
- `--max-depth`：超过该距离的深度记为无效；
- `--warning-depth`：目标距离小于该值时用红框显示。

## 5. JSON字段

每个目标包含：

- `class_name`、`score`、`bbox`：CenterNet 结果；
- `distance_m`：框内稳健深度中位数；
- `camera_xyz_m`：以左彩色相机为参考的近似位置；
- `median_disparity_px`：该距离对应的视差；
- `valid_depth_ratio`：采样区域内有效深度比例；
- `sample_roi`：实际用于测距的区域。

`stereo: null` 表示框内缺少足够可靠的视差，系统明确返回不可用，而不是编造距离。

## 6. 当前边界与后续记录

- COCO 预训练模型只负责通用人、车、自行车等类别，尚未适配园区摆渡车、锥桶等自定义类别。
- 当前输出是逐帧目标表面距离，不是目标中心真值，也没有速度和轨迹。
- SGBM 是学习和基线方案；确认数据链路后，可换成立体深度网络而保持检测与输出接口不变。
- 下一阶段应使用 KITTI `label_2` 统计距离误差，并增加连续帧跟踪、TTC 和园区数据微调。
- 未经实车标定、延迟测试和安全冗余验证，不得把当前结果直接作为唯一制动依据。
