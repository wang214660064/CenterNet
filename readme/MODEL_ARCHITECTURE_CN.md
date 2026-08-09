# Stereo Campus Gate v3 模型结构

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
| `dim` | 3 | 3D框尺寸 | L1 |
| `rot` | 8 | MultiBin朝向 | BinRot Loss |
| `depth_offset` | 1 | SGBM表面到3D框中心的深度修正 | Masked Huber |
| `depth_log_variance` | 1 | offset不确定性 | 校准损失 |
| `depth_gate` | 1 | 两种深度的融合权重 | BCE + 融合深度Huber |
模型总参数量为`22,677,314`。使用`--train_stereo_only`时只训练新增双目分支`2,061,991`个参数。

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
- 结构打印：`src/tools/print_stereo_model.py`

```bash
/opt/miniconda3/envs/clip/bin/python src/tools/print_stereo_model.py
```

## 6. 当前验证状态

- 9个输出头前向成功，形状正确；
- 20项项目测试通过；
- 本地真实KITTI样本完成一次前向、总损失和反向传播；
- 旧模型可加载DLA和常规检测头，新增模块重新初始化；
- 尚未完成Geometry Offset v4正式训练，因此暂时不能宣称3D AP得到提升。
