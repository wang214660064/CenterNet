# Stereo DDD + SGBM Offset 模型结构

## 1. 总体结构

```text
左图 RGB [B,3,384,1280]
          │
          ▼
    DLA-34 Backbone
          │
      DLAUp + IDAUp
          │
图像特征 [B,64,96,320]
    ├──────────────────────────────────────────────────────┐
    │                                                      │
    ├─ hm [3]             类别中心热图                     │
    ├─ wh [2]             2D框宽高                         │
    ├─ reg [2]            中心亚像素偏移                   │
    ├─ dep [1]            无SGBM时的直接深度回退           │
    ├─ dim [3]            3D框尺寸                         │
    └─ rot [8]            MultiBin朝向                     │
                                                           │
左右图 ── StereoSGBM ──> 深度图 + 有效/质量图              │
                           │                               │
                           ▼                               │
      concat(图像特征, log归一化深度, 质量) <───────────────┘
                           │
                  3x3 Conv + ReLU × 2
                     ├──────────────┐
                     ▼              ▼
             depth_offset [1]  depth_log_variance [1]
                     │              │
                     └─────门控─────┘
                           │
          z_corrected = z_sgbm + depth_offset
                           │
          SGBM无效或不确定性过大时使用direct dep
                           │
                           ▼
               最终KITTI 3D检测框
```

## 2. 输出头

输入分辨率为 `384×1280`，下采样倍数为4，所有输出头尺寸为 `96×320`。

| 输出头 | 通道 | 作用 | 监督 |
|---|---:|---|---|
| `hm` | 3 | Pedestrian、Car、Cyclist中心热图 | Focal Loss |
| `wh` | 2 | 2D框宽高 | L1 |
| `reg` | 2 | 中心点小数偏移 | L1 |
| `dep` | 1 | 网络直接深度，SGBM失败回退 | L1 |
| `dim` | 3 | 3D框高宽长 | L1 |
| `rot` | 8 | 两组方向分类和残差 | BinRot Loss |
| `depth_offset` | 1 | SGBM表面深度到3D框中心的修正 | Masked Laplace Loss |
| `depth_log_variance` | 1 | offset预测不确定性 | 与offset联合优化 |

模型实测总参数量为 `21,358,237`，全部可训练。

## 3. Offset监督与融合

```text
offset_target = z_label - z_sgbm
z_corrected = z_sgbm + offset_pred
uncertainty = exp(0.5 * depth_log_variance)
```

仅当以下条件同时成立时使用修正深度：

- SGBM深度大于0；
- SGBM质量不低于 `stereo_min_quality`；
- 修正后深度大于0；
- 预测不确定性不超过 `depth_offset_max_uncertainty`。

否则回退到 `dep` 直接深度头。训练时SGBM无效目标的offset mask为0，不参与offset损失。

## 4. 代码入口

- 双目数据集：`src/lib/datasets/sample/stereo_ddd.py`
- DLA双目模型：`src/lib/models/networks/stereo_pose_dla_dcn.py`
- Offset独立模块：`src/lib/models/networks/stereo_depth_offset.py`
- 损失与训练器：`src/lib/trains/stereo_ddd.py`
- 模型结构打印：`src/tools/print_stereo_model.py`
- 训练脚本：`experiments/stereo_ddd_3dop.sh`

打印完整模型结构：

```bash
/opt/miniconda3/envs/clip/bin/python src/tools/print_stereo_model.py \
  > exp/stereo_stage2/model_structure.txt
```

## 5. 已验证内容

- 8个输出头前向成功，形状全部符合设计；
- 真实KITTI双目样本可生成SGBM深度、质量图和offset标签；
- 一个真实样本完成总损失计算和反向传播；
- offset卷积层获得非零梯度；
- `src/main.py stereo_ddd` 已完成1迭代训练冒烟测试。

当前没有训练完成的 `stereo_ddd` 权重。现有 `ctdet_coco_dla_2x.pth` 只能初始化/运行2D检测，不是该模型的最终3D权重。
