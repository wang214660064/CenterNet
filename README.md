# 园区低速车双目 CenterNet 检测案例

本项目基于 Objects as Points CenterNet。第一阶段已完成左图2D检测与SGBM双目测距；当前进入第二阶段，参考 MonoFlex 增加3D属性头，并将SGBM作为几何深度候选参与融合。

## 当前状态

### 项目边界

本仓库现在只负责**单帧双目3D目标检测**：输入同步左右图和当帧标定，输出类别、置信度、2D框、3D尺寸、相机坐标中心和朝向。连续帧跟踪、速度/TTC、预测、规划、控制和服务化由后续模块负责，不再纳入本项目的网络优化范围。

- [X] KITTI 左图、标注、标定文件和 3DOP 划分；
- [X] KITTI 标注可视化与人性化导出；
- [X] CenterNet COCO DLA-34 2D 模型；
- [X] 双目 SGBM、目标框测距、风险着色和 JSON 输出代码；
- [X] KITTI `image_3` 右彩色图，训练/测试集左右帧完整配对；
- [X] 在 `000008` 上完成 CenterNet + SGBM 端到端运行；
- [X] 使用 KITTI 3DOP 验证集完成SGBM深度基线统计；
- [X] 生成SGBM深度、质量图和目标级offset监督；
- [X] 实现可训练的 `stereo_ddd` 模型、9个输出头和联合损失；
- [X] 完成Stereo Fusion Gate v2：双尺度融合、SGBM质量编码、目标感知聚合、ECA注意力和可学习深度门控；
- [X] 完成园区Gate v3：0～30m重点训练、30～50m远距预警、50m以上仅保留2D用途；gate采用代价加权Focal监督；
- [X] 完成Geometry Offset v4：尺寸/朝向几何先验、SGBM质量门控与学习残差；
- [X] 完成Geometry Offset v4的400帧正式评估：Car Moderate 3D AP_R40为30.36，优于v3的28.92；
- [X] 完成3D中心投影改造：保留2D框中心，新增独立`proj_center_offset`头；
- [X] 完成Projected Center v5训练及400帧评估：Car Moderate 3D AP_R40为42.24，v4为30.36；
- [X] 完成Projected Center v6a训练及400帧评估：新增相机坐标XY一致性损失，但Car Moderate 3D AP_R40仅由42.24变为42.29，未形成稳定收益；
- [X] 完成Projected Center v7训练及400帧评估：尺度归一化重叠代理未优于v5，Car Moderate 3D AP_R40降至41.99；
- [X] 完成Dimension v6b训练与400帧评估：相对尺寸Smooth L1未优于v5，保留为消融；
- [X] 完成3D属性头小学习率解冻消融：Car Moderate 3D AP_R40降至29.44，该分支不采用；当前最佳模型为Projected Center v5（42.24）；
- [X] 正式验证新增目标级深度诊断：分解Direct/SGBM/Offset/Gate/不确定性及回退原因；
- [X] 实现只进入Backbone的左目外观增强，SGBM始终使用原始左右图；
- [X] 完成LightStereo候选A/B并确定继续只使用SGBM；
- [X] 完成真实样本前向、反向传播和单迭代训练验证；
- [x] 在GPU环境完成3DOP训练并评估KITTI 3D AP_R40（最佳权重第10轮，Car Moderate 3D AP_R40 30.88）。

## 快速运行

右图下载并解压到 `data/kitti/training/image_3` 后：

```bash
python src/stereo_kitti_demo.py --image-id 000008
```

输出位于 `exp/stereo_stage1`：

- `*_detections.jpg`：类别、置信度、距离和风险颜色；
- `*_disparity.jpg`：SGBM 彩色视差；
- `*_results.json`：检测框、距离、相机坐标和有效深度比例。

详细原理、参数和数据检查见[双目第一阶段项目记录](docs/STEREO_STAGE1_CN.md)。
每次改造与验证结论记录在[开发日志](docs/PROJECT_LOG_CN.md)。
第二阶段设计和评测结果见 [MonoFlex + SGBM 路线](docs/MONOFLEX_SGBM_STAGE2_CN.md)。
完整网络结构见 [Stereo DDD模型结构](docs/MODEL_ARCHITECTURE_CN.md)。
论文形式的网络框架与创新说明见 [双目3D检测网络方法](docs/PAPER_NETWORK_METHOD_CN.md)。
各版本的初学者SVG图解见 [项目版本架构图](docs/architecture/README_CN.md)。
GitHub开源方案筛选见 [双目3D改进路线调研](docs/GITHUB_STEREO_IMPROVEMENTS_CN.md)。

打印模型结构：

```bash
/opt/miniconda3/envs/clip/bin/python src/tools/print_stereo_model.py
```

开始训练：

```bash
/opt/miniconda3/envs/clip/bin/python src/main.py stereo_ddd \
  --dataset kitti --arch stereo_dla_34 \
  --exp_id stereo_sgbm_offset \
  --batch_size 4 --num_workers 2
```

程序默认优先使用CUDA；没有CUDA时自动回退CPU，不需要填写GPU参数。CPU仅适合冒烟验证；`--gpus -1`只用于需要强制CPU的调试场景。

训练时终端会显示类似 Ultralytics 的动态进度，包括 epoch、显存、总损失、检测/深度/尺寸/朝向/offset/gate损失、投影中心像素损失`proj`、米制XY损失`xy`、尺度重叠代理损失`piou`、几何offset均值`geo`、学习残差均值`res`、当前批次目标数和输入尺寸。每轮结束后，`exp/stereo_ddd/<exp_id>/` 自动生成：

- `results.csv`：逐轮结构化指标，可用表格软件继续分析；
- `results.png`：总损失、深度、二维检测和三维属性四组曲线；
- `training_summary.json`：最近一轮、学习率、训练/验证指标和产物路径；
- `model_last.pth`：最近权重；按验证间隔运行后还会生成 `model_best.pth`；
- `opt.txt` 和 `logs_*/log.txt`：完整参数与文本日志。

在日志重定向或CI环境中可添加 `--no_progress_bar` 关闭动态进度条；`--print_iter N` 可每 N 次迭代额外输出一行固定文本。

正式训练脚本默认每5轮验证一次；若连续2次验证损失未改善至少0.01，训练会自动早停并保留 `model_best.pth`，避免训练集继续下降而验证集恶化。

当前训练脚本已启用距离感知的SGBM门控：质量图只由局部有效视差比例生成；offset采用绝对值和深度比例双重限幅；30m以上使用更严格的质量与不确定性条件。正式脚本会加载旧 `model_best.pth` 作为初始化，但使用新的 `exp_id`、较低学习率并从第1轮重新训练；不能添加 `--resume` 继承旧优化器和轮次。

Geometry Offset v4将SGBM可见表面深度修正拆成“尺寸/朝向几何先验 + 学习残差”。几何先验乘以SGBM局部质量和可学习门控。尺寸头与朝向头使用`detach()`单向供给，不会被offset损失带偏。

Projected Center v5不替换现有2D中心热力图，而是预测“2D框中心到3D几何中心投影点”的偏移。恢复相机坐标和`rotation_y`时使用修正后的投影中心，2D框继续使用原中心，两条路径不混用。旧v4权重加载后新头默认输出0，保持向后兼容。

Projected Center v5正式评估已完成：Car Moderate BEV/3D AP_R40由v4的`43.44/30.36`提升到`45.87/42.24`；2D AP、深度和尺寸误差保持不变。改善主要集中在0～30m，其中0～15m中心`x/y`平面误差由0.361m降至0.247m。

3D属性头小学习率解冻消融已经完成：平均深度、尺寸和朝向误差略有下降，但Car Moderate BEV/3D AP_R40分别由43.44/30.36降至42.32/29.44。项目主线已回退到冻结常规头的Geometry Offset v4，解冻入口不再保留。

本项目正式使用固定的 `project2000` 数据集，其中1600帧用于训练、400帧用于验证：

```bash
python src/tools/create_kitti_project_split.py
bash experiments/stereo_ddd_project2000.sh
```

从v4最佳权重开始单独训练3D中心投影头：

```bash
bash experiments/stereo_ddd_projected_center_v5.sh
```

从v5最佳权重继续训练v6a，只更新投影中心头：

```bash
bash experiments/stereo_ddd_projected_center_v6a.sh
```

v6a保留v5的像素偏移Smooth L1，并新增相机坐标`x/y`一致性损失。最终融合深度参与反投影前执行`detach()`，不会修改深度、gate、尺寸、朝向或骨干网络。终端中的`proj`表示像素偏移损失，`xy`表示相机坐标一致性损失，单位为米。

v6a正式评估已完成：Car Moderate BEV/3D AP_R40为`45.85/42.29`，v5为`45.87/42.24`；中心XY MAE仅由`0.28344m`降至`0.28336m`，平均3D IoU由`0.53321`变为`0.53310`，`IoU≥0.7`比例由`26.07%`降至`25.80%`。变化量接近统计波动，当前不以v6a替换v5最佳模型。

v7从v5最佳权重重新开始，保留像素投影中心损失、关闭v6a米制XY损失，并新增`0.2`权重的尺度归一化重叠代理损失。该损失根据车辆高度、宽度、长度和朝向衡量中心误差，因此小目标对同样的偏移受到更大惩罚；完全不重叠时使用归一化Huber项保持梯度。训练入口：

```bash
bash experiments/stereo_ddd_projected_center_v7_iou.sh
```

v7正式评估已完成：Car Moderate BEV/3D AP_R40为`45.86/41.99`，低于v5的`45.87/42.24`；平均中心XY误差由`0.28344m`变为`0.28391m`，平均3D IoU由`0.53321`变为`0.53276`。0～15m的`IoU≥0.7`比例由`37.43%`升至`38.50%`，但15～50m均未改善，当前不采用v7权重。

Dimension v6b从v5最佳权重开始，只训练已有`dim`尺寸头。训练关闭原始米制`dim_loss`，改用按真实高度、宽度、长度归一化的相对Smooth L1，避免车长主导损失；其他头和骨干均冻结。训练入口：

```bash
bash experiments/stereo_ddd_dimension_v6b.sh
```

后续训练、验证、AP和距离分桶均以 `project2000` 为准，原3DOP划分仅作为历史资料保留，不再作为项目默认评测口径。

Backbone外观增强通过`--backbone_photo_aug`显式开启，只对训练集中进入
DLA-34的左图副本调整亮度、对比度、Gamma、饱和度和轻微色温。
SGBM视差、深度、质量图和Offset监督仍由原始左右图计算，验证和推理
不做增强。该功能不改变数据字段、张量尺寸、标注或标定。启动正式训练前可视化检查：

```bash
python src/tools/visualize_backbone_augmentation.py --image-id 000008
```

从v5最佳权重出发，使用`1e-6`学习率微调全模型的单变量候选入口：

```bash
bash experiments/stereo_ddd_backbone_photo_aug_full_finetune.sh
```

该实验会同时更新Backbone、2D/3D检测头和双目融合分支。正式结论必须
在固定400帧验证集与v5比较2D、BEV、3D AP_R40和分距离深度误差。

划分与训练是两个独立步骤。先运行划分命令并检查 `data/kitti/ImageSets_project2000/train.txt`、`val.txt` 和 `summary.json`；确认后再单独执行训练脚本。左右图不会复制到数据集目录，清单中的帧号仍引用原始 `training/image_2` 和 `training/image_3`。

使用 `src/main.py ... --test --load_model <权重>` 完整推理验证集后，会生成KITTI格式预测、PR统计文件和 `kitti_ap_r40.json`，其中包含Car、Pedestrian、Cyclist的2D、AOS、BEV和3D AP_R40。

可视化一帧训练模型的3D预测、真值与BEV对比：

```bash
/root/miniconda3/bin/python src/tools/visualize_stereo_prediction.py \
  --image-id 000008 \
  --load-model exp/stereo_ddd/stereo_sgbm_offset/model_best.pth
```

结果保存在 `exp/stereo_ddd/stereo_single_visual/debug`。重点查看 `0add_pred.png`（预测3D框）、`0bird_pred_gt.png`（预测与真值BEV）和 `0out.png`（综合视图）。帧号必须属于3DOP验证集。

将完整验证结果按距离、遮挡和误差来源拆解：

```bash
conda run -n clip python src/tools/analyze_stereo_errors.py
```

正式`src/main.py ... --test`还会生成`stereo_detection_diagnostics.json`，不改变KITTI预测文件。输出的 `error_analysis.json`（汇总指标）和 `error_analysis_records.csv`（逐目标明细）会同时分解：

- 最终3D框的深度、中心`x/y`、尺寸、朝向、BEV IoU和3D IoU；
- 直接深度、SGBM中心深度、几何Offset、学习残差和最终融合深度；
- SGBM质量分桶、Gate选择正确率、Gate后悔值、不确定性和安全回退原因；
- 分别将深度、`x/y`中心、尺寸或朝向替换成真值的反事实3D IoU。

`src/test.py` 不会构建SGBM输入，不得用于`stereo_ddd`评估；脚本会直接提示改用`src/main.py --test`。

## 数据学习工具

```bash
# 显示 KITTI 2D/3D 真值框
python src/tools/visualize_kitti_gt.py --image-id 000008

# 将某帧 COCO JSON 转成人类可读结构
python src/tools/inspect_kitti_annotation.py --image-id 000008
```

KITTI 标签、坐标系和标定矩阵说明见 [KITTI 数据集中文文档](docs/KITTI_DATASET_CN.md)。

## 目录约定

```text
src/stereo_kitti_demo.py       第一阶段主入口
src/lib/                       原 CenterNet 推理与训练底层
src/tools/                     KITTI 转换、检查和可视化工具
data/kitti/                    本地数据，不提交 Git
models/                        本地权重，不提交 Git
exp/stereo_stage1/             运行结果，不提交 Git
docs/STEREO_STAGE1_CN.md       当前阶段项目记录
```

原始 CenterNet 项目的安装细节保留在 [INSTALL.md](docs/INSTALL.md)，原论文和许可证信息见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)。

## 安全边界

当前代码是教学和技术验证原型。双目在弱纹理、强反光、遮挡、雨雾和远距离场景会产生无效或错误深度，不能未经验证直接作为车辆唯一制动依据。
