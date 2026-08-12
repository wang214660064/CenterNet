# Geometry Offset v4 + Projected Center v6a 模型结构

## 1. 总体结构

```text
左图 RGB [B,3,384,1280]
          │
          ▼
    DLA-34 + DLAUp + IDAUp
          │
图像特征 [B,64,96,320]
    ├─ hm [3] / wh [2] / reg [2]
    ├─ direct dep [1] / dim [3] / rot [8]
    ├─ proj_center_offset [2]
    │
    └───────────────────────────────────────────┐
                                                │
左右图 ── StereoSGBM ──> 深度图 + 有效比例图   │
                           │                    │
                           ▼                    │
       深度/有效比例/局部离散度/梯度质量编码器  │
                           │                    │
              1/4细尺度 + 1/8粗尺度融合 <───────┘
                           │
                       ECA注意力
                           │
       按预测框大小选择3×3/7×7/15×15目标上下文
                  ┌────────┼───────────┐
                  ▼        ▼           ▼
              offset  uncertainty  learned_gate
                  │        │           │
                  └────────┴───────────┘
                           │
       z_final = gate*z_stereo + (1-gate)*z_direct
                           │
          无效或越过安全边界时强制使用direct dep
```

## 2. 五项结构改造

1. `depth_gate`学习选择SGBM修正深度或网络直接深度；
2. 使用1/4细尺度和1/8粗尺度的轻量双尺度融合；
3. 根据预测2D框大小，在3×3、7×7和15×15邻域间自适应聚合目标特征；
4. 独立编码SGBM深度、有效比例、局部离散度和深度梯度；
5. 融合特征经过ECA轻量通道注意力。

第三项采用CenterNet兼容的目标中心聚合方式：保留稠密输出形式，但每个中心位置会根据预测框大小读取不同范围的目标上下文，不再只依赖中心单像素。

## 3. 输出头

输入为`384×1280`，下采样倍数为4，全部输出尺寸为`96×320`。

| 输出头 | 通道 | 作用 | 监督 |
|---|---:|---|---|
| `hm` | 3 | Pedestrian、Car、Cyclist中心热图 | Focal Loss |
| `wh` | 2 | 2D框宽高 | L1 |
| `reg` | 2 | 中心亚像素偏移 | L1 |
| `dep` | 1 | 网络直接深度 | L1 |
| `dim` | 3 | 3D框尺寸 | L1 + 可选相对尺寸Smooth L1 |
| `rot` | 8 | MultiBin朝向 | BinRot Loss |
| `depth_offset` | 1 | SGBM表面到3D框中心的深度修正 | Masked Huber |
| `depth_log_variance` | 1 | offset不确定性 | 校准损失 |
| `depth_gate` | 1 | 两种深度的融合权重 | BCE + 融合深度Huber |
| `proj_center_offset` | 2 | 2D框中心到3D几何中心投影点的偏移 | 像素Smooth L1 + 可选XY/重叠代理损失 |

模型总参数量为`22,825,540`。新投影中心头为`148,226`个参数。使用`--train_projected_center_only`时冻结v4全部已验证分支，只更新这个新头。

### Projected Center v5

KITTI的`location`表示3D框底面中心。生成监督时先向上移动半个目标高度，再通过当帧`P2`投影到左图：

```text
center_3d = [x, y - height / 2, z]
projected_center = P2 × center_3d
proj_center_offset = projected_center - bbox_center_2d
```

推理时分为两条路径：

```text
bbox_center_2d + wh                 -> 2D框
bbox_center_2d + proj_center_offset -> 3D中心投影点
3D中心投影点 + depth + P2 -> 相机坐标(x,y,z)
```

训练沿用园区距离权重：0～15m为2.0，15～30m为1.5，30～50m为0.5，50m以上不训练该3D偏移。

对极端边缘车辆，真实3D中心可能远在画面外。当偏移超过64个输出特征格（原图256像素）时暂不参与该头监督，推理也做同样的向量限幅。该阈值在`project2000`中保留训练98.66%、验证98.52%的目标；剩余边缘目标需要后续Edge Fusion处理。

### Projected Center v6a

v6a不增加输出通道，也不改变推理解码。训练时将预测投影中心从输出特征坐标还原到原图，再结合当帧`P2`和最终融合深度反投影：

```text
center_output = bbox_center + proj_center_offset
center_image = inverse_affine(center_output)
pred_camera_xy = unproject(center_image, detach(z_final), P2)
L_center = L_pixel + 1.0 × SmoothL1(pred_camera_xy, gt_camera_xy)
```

`XY`损失的Smooth L1转折点为`0.2m`，并沿用园区距离权重。`z_final`显式执行`detach()`，训练模式仍为`--train_projected_center_only`，因此只有`proj_center_offset`头更新。

### Projected Center v7

v6a优化绝对米制XY误差，但同样`0.2m`偏移对小车和大车的3D IoU影响不同。v7根据车辆尺寸和朝向计算相机X方向有效宽度，并用车辆高度归一化纵向误差：

```text
extent_x = |cos(rotation_y)| × width + |sin(rotation_y)| × length
error_normalized = |pred_camera_xy - gt_camera_xy| / [extent_x, height]
L_v7 = L_pixel + 0.2 × (L_overlap + 0.1 × L_huber_fallback)
```

`L_overlap`模拟中心偏移造成的3D框重叠下降；当预测与真值完全错开时，`L_huber_fallback`继续提供梯度。尺寸和朝向只生成训练监督，模型仍只更新`proj_center_offset`头，推理解码与v5完全相同。

v7在`project2000`400帧评估中的Car Moderate BEV/3D AP_R40为`45.86/41.99`，低于v5的`45.87/42.24`。它只在0～15m的`IoU≥0.7`比例有小幅增加，15～50m没有改善；因此该代理损失作为已验证未采用的消融保留，当前最佳权重仍为v5。

### Dimension v6b

v6b只更新已有`dim`头，不增加参数。原始米制L1会使长度维度的绝对误差通常大于高度和宽度；v6b改为相对Smooth L1：

```text
relative_error = (pred_dim - gt_dim) / max(|gt_dim|, 0.1)
L_dim_v6b = SmoothL1(relative_error, beta=0.1)
```

训练脚本关闭原始`dim_weight`，只启用`dimension_aware_weight=1.0`，并冻结深度、中心、朝向、2D检测头和骨干网络。该分支已在同一400帧完成评估：Car Moderate 3D AP_R40为`42.18`，低于v5的`42.24`；尺寸MAE为`0.16041m`，也略高于v5的`0.15987m`，因此不替代v5主线。

## 4. 门控监督与安全边界

```text
offset_target = z_label - z_sgbm
z_stereo = z_sgbm + limited_offset
gate = sigmoid(depth_gate)
z_final = gate * z_stereo + (1 - gate) * z_direct
```

训练期间比较`z_stereo`和`z_direct`与真值深度的误差，误差较小的一方生成gate监督。真值只参与训练损失，推理阶段不读取标签。

可学习gate外面仍保留硬安全边界：

- SGBM深度必须有效，质量必须达到阈值；
- offset不超过8m、SGBM深度的15%，近距离上限不低于2m；
- 30m以上要求质量不低于0.8、预测标准差不超过3m；
- 条件不满足时gate强制为0，回退到网络直接深度。

园区距离策略进一步规定：

| 距离 | 双目损失权重 | 推理用途 |
|---|---:|---|
| 0～15m | 2.0 | 核心3D检测与近距风险感知 |
| 15～30m | 1.5 | 核心3D检测与规划提前量 |
| 30～50m | 0.5 | 粗深度、跟踪和远距预警 |
| 50m以上 | 0.0 | 不训练双目修正，只保留2D观察 |

gate不再使用普通BCE。v3采用Focal调制，并按两种候选深度的误差差增加`regret_weight`；误差差小于0.2m的模糊样本不参与gate分类。50m以上强制gate为0，但KITTI离线评估仍保留直接深度预测，避免人为删预测影响对比。

### Geometry Offset v4

SGBM框内中位数更接近物体可见表面，表面到3D中心的偏移与尺寸和观察角有关：

```text
geometry_offset = 0.5 × (length × |cos(alpha)| + width × |sin(alpha)|)
depth_offset = quality × learned_geometry_gate × geometry_offset
             + learned_residual_offset
```

`dim`和`rot`在进入几何分支前均执行`detach()`。因此尺寸、朝向参与offset推理，但offset损失不会通过该路径修改尺寸头和朝向头。几何项不是硬编码车长一半：SGBM质量差时其贡献自动减小，车窗、侧面、遮挡和框内背景造成的剩余误差由残差头学习。

## 5. 代码入口

- 模型：`src/lib/models/networks/stereo_pose_dla_dcn.py`
- 深度融合：`src/lib/models/networks/stereo_depth_offset.py`
- 损失：`src/lib/trains/stereo_ddd.py`
- 正式训练：`experiments/stereo_ddd_project2000.sh`
- 投影中心A/B：`experiments/stereo_ddd_projected_center_v5.sh`
- 投影中心XY一致性：`experiments/stereo_ddd_projected_center_v6a.sh`
- 投影中心尺度重叠代理：`experiments/stereo_ddd_projected_center_v7_iou.sh`
- 相对尺寸头消融：`experiments/stereo_ddd_dimension_v6b.sh`
- 结构打印：`src/tools/print_stereo_model.py`

```bash
/opt/miniconda3/envs/clip/bin/python src/tools/print_stereo_model.py
```

## 6. 当前验证状态

- 30项项目测试通过，10个输出头前向成功，`proj_center_offset`形状为`[B,2,96,320]`；
- 新头最后一层零初始化，旧v4权重保持原始中心行为；
- 本地真实KITTI样本完成标签生成、前向、总损失和反向传播；
- Geometry Offset v4和Projected Center v5均已完成`project2000`正式评估；
- v5 Car Moderate BEV/3D AP_R40为`45.87/42.24`，v4为`43.44/30.36`；
- v5不改变2D AP、深度和尺寸误差，改善来自中心`x/y`恢复，并主要集中在0～30m。
- v6a已完成30轮训练和`project2000` 400帧评估；Car Moderate BEV/3D AP_R40为`45.85/42.29`，与v5的`45.87/42.24`基本持平。
- v6a中心XY MAE仅由v5的`0.28344m`降至`0.28336m`，平均3D IoU略降，未形成可重复的实质收益；当前最佳模型仍为v5。
