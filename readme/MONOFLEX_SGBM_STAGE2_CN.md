# 第二阶段：MonoFlex思想 + SGBM深度的双目3D检测

## 1. 技术路线

第二阶段参考 [MonoFlex 解读](https://blog.csdn.net/qq_41204464/article/details/134000725)及其原论文，将现有 CenterNet 从“2D框 + 外挂距离”升级为“3D属性检测头 + 多源深度融合”。与原始 MonoFlex 不同，本项目已有双目相机，因此增加 SGBM 几何深度候选。

```text
左图 ──> DLA-34 / FPN ──> CenterNet中心热图
                         ├─ 2D框宽高
                         ├─ 2D中心到3D投影中心偏移
                         ├─ 3D尺寸
                         ├─ MultiBin朝向
                         ├─ 10个投影关键点
                         ├─ 网络直接回归深度及不确定性
                         └─ 关键点几何深度及不确定性

左右图 ──> SGBM ──> 目标区域深度及质量指标

直接深度 + 关键点几何深度 + SGBM深度
                ↓ 不确定性加权/有效性门控
              最终3D中心深度
                ↓
       (x, y, z, h, w, l, yaw) 3D框
```

## 2. 为什么不直接用SGBM替换深度头

完整 3DOP 验证集的真值二维框评测结果：

| 指标 | 结果 |
|---|---:|
| 帧数 | 3769 |
| 目标数 | 17554 |
| 有效深度 | 16993 |
| 有效率 | 96.80% |
| MAE | 2.68 m |
| Median AE | 1.40 m |
| RMSE | 4.77 m |
| 平均相对误差 | 14.32% |
| 有符号偏差 | -1.62 m |

无遮挡目标 MAE 为 1.36m，遮挡等级2的目标 MAE 上升到 6.32m。SGBM 对多数目标有效，但容易在遮挡边界、无纹理区域、反光表面和小目标上取到背景或前景深度。

此外，KITTI 标签 `z` 是3D框位置，SGBM框内中位数更接近车辆可见表面，两者天然存在偏差。因此合理用法是：

1. 把 SGBM 作为第三个深度估计器，而不是唯一深度；
2. 同时输入有效深度比例、视差离散度等质量指标；
3. SGBM无效或质量差时退回网络直接深度和关键点几何深度；
4. 训练网络预测 SGBM 到3D中心的残差及不确定性。

建议的残差形式：

```text
delta_z = z_label - z_sgbm
z_stereo_corrected = z_sgbm + delta_z_pred
```

当前代码将其命名为 `depth_offset`，监督标签为 `offset_target_m`。同时预测 `depth_log_variance`，使遮挡、反光和低纹理目标能够降低SGBM修正分支的融合权重。实现入口为 `src/lib/models/networks/stereo_depth_offset.py`。

融合形式沿用 MonoFlex 的不确定性思想：

```text
weight_i = 1 / uncertainty_i
z_final = sum(weight_i * z_i) / sum(weight_i)
```

无效的深度候选通过掩码移出融合，不使用人为填充值参与训练。

## 3. 阶段拆分

### 2A：SGBM量化基线，已完成

工具：`src/tools/evaluate_stereo_depth.py`

快速检查：

```bash
/opt/miniconda3/envs/clip/bin/python src/tools/evaluate_stereo_depth.py \
  --limit 50 \
  --output exp/stereo_stage2/sgbm_depth_metrics_50.json
```

完整验证集：

```bash
/opt/miniconda3/envs/clip/bin/python src/tools/evaluate_stereo_depth.py
```

默认输出 `exp/stereo_stage2/sgbm_depth_metrics.json`。该评测使用真值二维框，作用是隔离检测误差、评估SGBM上限，不代表完整系统性能。

### 2B：构建MonoFlex式训练目标

需要从 KITTI 标签生成并可视化：

- 类别中心热图；
- 2D框宽高；
- 2D框中心到3D中心投影的偏移；
- 3D尺寸残差；
- MultiBin朝向；
- 8个3D框顶点与顶面/底面中心，共10个关键点；
- SGBM采样深度、有效掩码、质量特征及深度残差标签；
- 截断目标的边界中心表示。

先检查标签可视化，再开始训练，避免坐标系或下采样尺度错误进入模型。

### 2C：最小3D头，已完成训练闭环

当前已经实现：

```text
hm + wh + center_offset + dim + rot + direct_depth
```

后续依次加入：

1. 10关键点与几何深度；
2. 三路深度不确定性融合；
3. Edge Fusion与截断目标解耦。

SGBM残差修正、有效性门控和直接深度回退已经接入 `stereo_ddd`。模型结构见 [MODEL_ARCHITECTURE_CN.md](MODEL_ARCHITECTURE_CN.md)。

这种顺序能分别测量 SGBM、关键点和边缘模块的真实收益。

## 4. 评测要求

每次模型升级都固定使用 3DOP train/val 划分，并至少报告：

- CenterNet 2D检测 AP；
- 深度有效率、MAE、Median AE、RMSE；
- 按距离、遮挡和截断分桶的深度误差；
- KITTI Car 3D AP_R40，Easy/Moderate/Hard；
- 单帧推理耗时；
- SGBM失败时的回退成功率。

对比组固定为：第一阶段 SGBM、直接回归深度、直接深度+SGBM修正、完整多源融合。

## 5. 当前限制

- 当前SGBM基线使用真值二维框，尚未计入 CenterNet 漏检和框偏差；
- 当前测量区域是通用启发式规则，不同类别可能需要不同采样策略；
- 论文中的10关键点、Edge Fusion和不确定性融合尚未写入当前旧版 CenterNet；
- 本阶段仍是教学和技术验证，不是车辆安全控制模块。
