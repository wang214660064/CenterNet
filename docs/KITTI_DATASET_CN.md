# KITTI Object 数据集中文说明

## 当前已下载内容

```text
data/kitti/
├── training/
│   ├── image_2/    左彩色相机，7481 帧
│   ├── image_3/    右彩色相机，双目阶段需要补充
│   ├── label_2/    训练集目标标签，7481 份
│   └── calib/      每帧传感器标定，7481 份
├── testing/
│   ├── image_2/    左彩色相机，7518 帧
│   ├── image_3/    右彩色相机，双目阶段需要补充
│   └── calib/      测试集标定，7518 份
├── ImageSets_3dop/ 训练/验证编号划分
└── annotations/    转换后的 CenterNet/COCO 格式标注
```

`testing` 没有公开标签，需要将预测提交到 KITTI 官方评测。学习和本地验证主要使用 `training` 的 3DOP train/val 划分。

## label_2 每行含义

```text
type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y [score]
```

- `type`：Car、Pedestrian、Cyclist、Van、Truck、DontCare 等；
- `truncated`：目标被图像边界截断的比例；
- `occluded`：遮挡等级，0 到 3；
- `alpha`：相对观察角，范围约为 `[-pi, pi]`；
- `x1 y1 x2 y2`：左彩色图上的二维框；
- `h w l`：三维框高、宽、长，单位米；
- `x y z`：物体在校正后左彩色相机坐标系中的位置，单位米；
- `rotation_y`：物体绕相机 Y 轴的朝向；
- `score`：仅预测结果需要，真值标签一般没有。

相机坐标约定为 X 向右、Y 向下、Z 向前。标签中的 `(x,y,z)` 位于三维框底面中心，不是几何中心。

## calib 标定字段

- `P0`、`P1`：校正后的左右灰度相机投影矩阵；
- `P2`、`P3`：校正后的左右彩色相机投影矩阵；
- `R0_rect`：参考相机的极线校正旋转；
- `Tr_velo_to_cam`：Velodyne 激光雷达到参考相机的外参；
- `Tr_imu_to_velo`：IMU 到激光雷达的外参。

三维点投影到左彩色图的大致过程为：

```text
X_velodyne -> Tr_velo_to_cam -> R0_rect -> P2 -> image_2 像素
```

`P2/P3` 同时包含相机内参和校正后的平移信息，双目脚本通过它们计算焦距、主点和基线。不要对不同帧的矩阵求平均。

## 标定和标签是怎么来的

标定参数来自车载相机、Velodyne HDL-64E 激光雷达和 IMU/GPS 之间的离线几何标定，不是模型逐帧训练或预测出来的。KITTI Object 把多个采集序列重新编号混合成独立样本，因此每帧都附带一份标定文件；实际数据中会出现少数几组重复参数，对应不同采集或标定批次。

3D 标签主要是人工/半自动标注，标注者会利用同步图像和激光点云确定三维框。点云帮助形成标签，但训练时是否使用点云取决于模型路线：当前单目/双目代码只读取图像、标签和 `P2/P3`，并不把激光点云送入网络。

## 本项目工具

显示二维框和投影后的三维框：

```bash
python src/tools/visualize_kitti_gt.py --image-id 000008
```

把压缩的 COCO JSON 导出为容易阅读的单帧 JSON：

```bash
python src/tools/inspect_kitti_annotation.py --image-id 000008
```

生成 CenterNet 使用的 3DOP 标注：

```bash
python src/tools/convert_kitti_to_coco.py
```

双目第一阶段见 [STEREO_STAGE1_CN.md](STEREO_STAGE1_CN.md)。

## 常见误区

- `image_2/image_3` 是彩色左右目，`P2/P3` 与之对应；
- KITTI 已完成极线校正，不应再用原始畸变模型重复校正；
- 视差越小，距离误差越敏感，因此远距离双目测距会明显变差；
- 标签深度描述物体位置，SGBM 框内深度通常落在可见表面，两者存在系统差异；
- 不能把训练集标签或点云生成的深度泄漏到实际推理输入中。

官方说明：[KITTI 3D Object Detection Evaluation](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)。
