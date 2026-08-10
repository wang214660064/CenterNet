# 项目版本架构图

本目录按项目主线版本保存SVG架构图。建议初学者先看“演进总览”，再按编号依次阅读。

## 阅读标准

- 箭头表示数据或特征流动方向；
- 红色虚线表示训练梯度，只在训练阶段存在；
- 灰色模块表示继承但冻结，不参与当前版本参数更新；
- 每张图底部的“初学者一句话”说明该版本最重要的变化；
- 蓝色是输入，橙色是神经网络，绿色是传统几何，紫色是融合，青色是输出，红色是损失。

## 版本列表

| 顺序 | 版本 | 主要变化 | 状态 |
|---:|---|---|---|
| 0 | 演进总览 | 展示主线和未采用的消融旁支 | 已完成 |
| 1 | 阶段1 | CenterNet 2D检测 + SGBM双目测距 | 已完成 |
| 2 | Stereo DDD基线 | 增加3D属性头和SGBM深度offset | 已完成 |
| 3 | Fusion Gate v2 | 双尺度、质量编码、目标上下文和学习gate | 已完成 |
| 4 | Campus Gate v3 | 园区距离分层与代价加权Focal gate | 已完成 |
| 5 | Geometry Offset v4 | 尺寸/朝向几何先验 + 学习残差 | 已完成并评估 |
| 6 | Projected Center v5 | 分离2D框中心和3D中心投影点 | 已完成并评估 |
| 7 | Projected Center v6a | 增加相机XY一致性损失 | 已评估，未显著优于v5 |
| 8 | Projected Center v7 | 尺度归一化中心重叠代理损失 | 已评估，未优于v5 |
| 9 | Dimension v6b | 只训练尺寸头 + 相对尺寸Smooth L1 | 代码完成，待训练 |

## 0. 演进总览

![版本演进总览](00_evolution_overview.svg)

## 1. 阶段1：CenterNet + SGBM

![阶段1架构](01_stage1_centernet_sgbm.svg)

## 2. Stereo DDD基线

![Stereo DDD基线](02_stereo_ddd_baseline.svg)

## 3. Fusion Gate v2

![Fusion Gate v2](03_fusion_gate_v2.svg)

## 4. Campus Gate v3

![Campus Gate v3](04_campus_gate_v3.svg)

## 5. Geometry Offset v4

![Geometry Offset v4](05_geometry_offset_v4.svg)

## 6. Projected Center v5

![Projected Center v5](06_projected_center_v5.svg)

## 7. Projected Center v6a

![Projected Center v6a](07_projected_center_v6a.svg)

## 8. Projected Center v7

![Projected Center v7](08_projected_center_v7_iou.svg)

## 9. Dimension v6b

![Dimension v6b](09_dimension_v6b.svg)

## 重新生成

架构图由Python标准库生成，不需要安装额外绘图库：

```bash
python src/tools/generate_architecture_svgs.py
```

修改架构图时应同时更新生成脚本，避免手工SVG和项目结构说明长期不一致。
