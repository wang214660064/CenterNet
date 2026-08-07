# 项目记录

## 2026-08-07：双目第一阶段代码完成

目标：在不训练新 3D 模型的情况下，先建立可以解释和逐段验证的园区低速车感知基线。

已完成：

- 将根 README 从原始论文介绍收束为本项目入口、状态与安全边界；
- 新增 `src/stereo_kitti_demo.py`；
- 使用 COCO CenterNet 检测人、自行车、汽车、摩托车、公交车、卡车、交通灯和停止标志；
- 使用 KITTI `image_2/image_3` 和逐帧 `P2/P3` 运行 StereoSGBM；
- 从检测框中下部中心区域提取有效深度，使用中位数和 MAD 过滤离群点；
- 输出检测叠加图、彩色视差图与结构化 JSON；
- 新增标定/深度几何单元测试；
- 删除 `src/cache/debug` 中十张旧调试图，并清理 `.DS_Store`、`__pycache__` 和旧 demo 输出；这些项目已移入系统废纸篓，可恢复；
- 保留原 CenterNet `src/lib` 训练/推理底层，避免过早破坏后续微调能力。

验证结果：

- 合成标定参数的基线与深度公式测试通过；
- 当前 KITTI `000008` 标定解析得到 `fx=721.5377 px`、`baseline=0.5327 m`；
- Python 语法检查与 `git diff --check` 通过；
- 初次检查时默认 Python 缺少 OpenCV/PyTorch，后续确认 Conda `base` 环境已包含两者。

下一步：

1. 检查 `training/image_3` 为 7481 张、`testing/image_3` 为 7518 张；
2. 安装并确认 CenterNet 原生扩展、OpenCV 和 PyTorch 可用；
3. 先运行 `000008`，检查左右图是否同尺寸、视差方向是否正确；
4. 将双目距离与 `label_2` 的 `z` 分桶比较，记录 5 m、10 m、20 m、40 m 内误差；
5. 根据园区目标大小调整 SGBM 和框内采样区域。

## 2026-08-07：右图接入和首次端到端验证

- `data_object_image_3.zip` 完整性校验通过，原压缩包保留；
- 解压得到训练右图 7481 张、测试右图 7518 张；
- 训练集左右文件名交集为 7481，对应关系完整；
- 安装旧 CenterNet 缺失的轻量依赖 `progress`、`easydict`；
- 使用 Conda `base` 环境和 CPU 成功运行 `000008`；
- CenterNet 输出 12 个候选目标，其中包括 11 个汽车候选和 1 个低置信度交通灯候选；
- 生成检测叠加图、视差图和结构化结果到 `exp/stereo_stage1`。

首次结果与该帧六个 Car 标签按二维框匹配后，SGBM 表面深度分别为 `4.13、6.30、5.03、12.84、33.07、19.28 m`，标签 Z 为 `3.68、7.86、6.15、14.44、33.20、19.96 m`。绝对误差分别约为 `0.45、1.56、1.12、1.60、0.13、0.68 m`。这只是单帧链路检查：标签 Z 表示三维框位置，而框内视差更接近可见表面，不能把这六个值当成正式精度结论。

当前待办缩减为：在 3DOP 验证集上批量统计误差，并处理远处小框、DontCare 区域和低置信度重复候选。

## 2026-08-07：建立仓库级开发约束

- 新增根目录 `AGENTS.md`，适用于整个项目；
- 固化中文沟通、问号进度检查、先检查工作区再修改等协作规则；
- 明确当前双目第一阶段及后续演进顺序，避免无依据地切换传感器路线；
- 明确数据、模型、标定、结果目录及禁止真值泄漏规则；
- 建立语法、单元测试、真实样例、视觉检查和误差统计等验证门槛；
- 约束清理操作保持可恢复，Git 操作不自动提交且任何位置不得包含 AI 署名。

## 2026-08-07：启动 MonoFlex + SGBM 第二阶段

- 根据用户选择，第二阶段不复现 Stereo CenterNet 十分支结构，改为参考 MonoFlex 的3D属性头和多深度融合；
- 保留 SGBM 作为可解释的双目几何深度候选，并计划学习 SGBM 到3D框中心的残差与不确定性；
- 新增 `src/tools/evaluate_stereo_depth.py`，使用真值二维框隔离检测误差，批量评估SGBM；
- 使用 `clip` 环境完成全部 3769 帧 3DOP 验证集评测；
- 共评估17554个目标，16993个有效，有效率96.80%；
- 整体 MAE 2.68m、Median AE 1.40m、RMSE 4.77m、平均相对误差14.32%；
- 无遮挡目标 MAE 1.36m，遮挡等级2目标 MAE 6.32m，证明需要质量门控和不确定性融合；
- 新增 `readme/MONOFLEX_SGBM_STAGE2_CN.md`，记录模型结构、融合方式、实施顺序和验收指标；
- 下一步生成中心偏移、3D尺寸、朝向、10关键点和SGBM残差等训练目标并逐帧可视化。

深度修正定义进一步固定为 `z_final = z_sgbm + depth_offset`：

- 新增 `StereoDepthOffsetHead`，融合图像特征、归一化SGBM深度和SGBM质量；
- 同时输出 `depth_offset` 与 `depth_log_variance`；
- 新增带有效掩码的 Laplace 不确定性损失，SGBM无效区域不参与残差训练；
- 批量评测明细新增 `offset_target_m = z_label - z_sgbm`、深度MAD和IQR质量字段；
- offset由训练集监督学习，禁止使用验证集平均误差作为固定补偿。

## 2026-08-07：完成Stereo DDD模型训练闭环

- 新增 `stereo_ddd` 任务并注册数据集、模型和训练器；
- 数据集在线计算SGBM，生成下采样深度图、质量图、offset标签和有效掩码；
- 新增 `StereoDLASeg`，DLA-34图像特征与SGBM深度/质量在offset分支融合；
- 保留原 `hm/wh/reg/dep/dim/rot` 六个输出，新增 `depth_offset/depth_log_variance`；
- 最终深度使用 `z_sgbm + depth_offset`，质量差或不确定性过大时回退到直接深度；
- 新增 `StereoDddLoss`，总损失加入带掩码的Laplace offset不确定性损失；
- 新增模型结构打印工具和 `experiments/stereo_ddd_3dop.sh` 训练脚本；
- `clip` 环境补充安装 `pycocotools`，并写入requirements；
- 实测模型参数量21,358,237，8个输出头形状正确；
- 真实KITTI样本完成前向与反向传播，offset层平均梯度绝对值为0.0889；
- `src/main.py stereo_ddd` 完成1次训练迭代；当前CPU单迭代约24秒，不适合直接完成70轮训练；
- 修复日志复制命令对包含空格的项目路径不兼容问题。
- 将旧版PyTorch损失参数更新为 `reduction='sum'`，消除弃用警告；
- 非CUDA环境关闭DataLoader的 `pin_memory`，避免无意义警告。

## 2026-08-07：增加可观察的训练进度与成果

- 将旧式终端进度替换为动态 `tqdm` 进度条；
- 每个批次展示显存、总损失、热力图、深度、尺寸、朝向和SGBM offset损失；
- 同时展示当前批次有效目标数和网络输入尺寸，便于发现空标注或尺寸错误；
- 每轮自动写入 `results.csv` 和 `training_summary.json`；
- 每轮自动生成 `results.png`，分面观察总损失、深度、二维检测和三维属性收敛趋势；
- 保留 TensorBoard（环境安装 `tensorboardX` 时启用）、文本日志和权重文件；
- 新增 `--no_progress_bar`，适配日志重定向和CI运行。

验证：使用真实KITTI数据完成2轮、每轮1次迭代的CPU冒烟训练；成功生成CSV、JSON、2080×1280训练曲线图和约241MB的最近权重，4项自动化测试全部通过。该短训练只验证链路和展示效果，不代表模型已经收敛。
