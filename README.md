# 园区低速车双目 CenterNet 检测案例

本项目基于 Objects as Points CenterNet。第一阶段已完成左图2D检测与SGBM双目测距；当前进入第二阶段，参考 MonoFlex 增加3D属性头，并将SGBM作为几何深度候选参与融合。

## 当前状态

- [X] KITTI 左图、标注、标定文件和 3DOP 划分；
- [X] KITTI 标注可视化与人性化导出；
- [X] CenterNet COCO DLA-34 2D 模型；
- [X] 双目 SGBM、目标框测距、风险着色和 JSON 输出代码；
- [X] KITTI `image_3` 右彩色图，训练/测试集左右帧完整配对；
- [X] 在 `000008` 上完成 CenterNet + SGBM 端到端运行；
- [X] 使用 KITTI 3DOP 验证集完成SGBM深度基线统计；
- [X] 生成SGBM深度、质量图和目标级offset监督；
- [X] 实现可训练的 `stereo_ddd` 模型、8个输出头和联合损失；
- [X] 完成真实样本前向、反向传播和单迭代训练验证；
- [ ] 在GPU环境完成3DOP训练并评估KITTI 3D AP_R40。

## 快速运行

右图下载并解压到 `data/kitti/training/image_3` 后：

```bash
python src/stereo_kitti_demo.py --image-id 000008 --gpus -1
```

输出位于 `exp/stereo_stage1`：

- `*_detections.jpg`：类别、置信度、距离和风险颜色；
- `*_disparity.jpg`：SGBM 彩色视差；
- `*_results.json`：检测框、距离、相机坐标和有效深度比例。

详细原理、参数和数据检查见[双目第一阶段项目记录](readme/STEREO_STAGE1_CN.md)。
每次改造与验证结论记录在[开发日志](readme/PROJECT_LOG_CN.md)。
第二阶段设计和评测结果见 [MonoFlex + SGBM 路线](readme/MONOFLEX_SGBM_STAGE2_CN.md)。
完整网络结构见 [Stereo DDD模型结构](readme/MODEL_ARCHITECTURE_CN.md)。

打印模型结构：

```bash
/opt/miniconda3/envs/clip/bin/python src/tools/print_stereo_model.py
```

开始训练：

```bash
/opt/miniconda3/envs/clip/bin/python src/main.py stereo_ddd \
  --dataset kitti --arch stereo_dla_34 \
  --exp_id stereo_sgbm_offset \
  --batch_size 4 --num_workers 2 --gpus -1
```

CPU仅适合冒烟验证；正式训练应在NVIDIA GPU环境把 `--gpus -1` 改为 `--gpus 0`。

训练时终端会显示类似 Ultralytics 的动态进度，包括 epoch、显存、总损失、检测/深度/尺寸/朝向/offset 损失、当前批次目标数和输入尺寸。每轮结束后，`exp/stereo_ddd/<exp_id>/` 自动生成：

- `results.csv`：逐轮结构化指标，可用表格软件继续分析；
- `results.png`：总损失、深度、二维检测和三维属性四组曲线；
- `training_summary.json`：最近一轮、学习率、训练/验证指标和产物路径；
- `model_last.pth`：最近权重；按验证间隔运行后还会生成 `model_best.pth`；
- `opt.txt` 和 `logs_*/log.txt`：完整参数与文本日志。

在日志重定向或CI环境中可添加 `--no_progress_bar` 关闭动态进度条；`--print_iter N` 可每 N 次迭代额外输出一行固定文本。

## 数据学习工具

```bash
# 显示 KITTI 2D/3D 真值框
python src/tools/visualize_kitti_gt.py --image-id 000008

# 将某帧 COCO JSON 转成人类可读结构
python src/tools/inspect_kitti_annotation.py --image-id 000008
```

KITTI 标签、坐标系和标定矩阵说明见 [KITTI 数据集中文文档](readme/KITTI_DATASET_CN.md)。

## 目录约定

```text
src/stereo_kitti_demo.py       第一阶段主入口
src/lib/                       原 CenterNet 推理与训练底层
src/tools/                     KITTI 转换、检查和可视化工具
data/kitti/                    本地数据，不提交 Git
models/                        本地权重，不提交 Git
exp/stereo_stage1/             运行结果，不提交 Git
readme/STEREO_STAGE1_CN.md     当前阶段项目记录
```

原始 CenterNet 项目的安装细节保留在 [INSTALL.md](readme/INSTALL.md)，原论文和许可证信息见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)。

## 安全边界

当前代码是教学和技术验证原型。双目在弱纹理、强反光、遮挡、雨雾和远距离场景会产生无效或错误深度，不能未经验证直接作为车辆唯一制动依据。
