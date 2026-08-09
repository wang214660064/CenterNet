# GitHub双目3D改进路线调研

## 1. 当前结论

当前项目不应立即迁移到重型3D体素网络。最合适的演进顺序是：

1. 完成园区距离加权、置信门控与连续帧稳定；
2. 使用轻量深度网络与SGBM做离线A/B，不先改3D检测头；
3. 深度替换有效后，再评估端到端左右图特征关联；
4. 最后才考虑3D体素或LiDAR教师模型。

## 2. 推荐候选

### OpenStereo / LightStereo

- 仓库：https://github.com/XiandaGuo/OpenStereo
- 提供统一训练评估框架、AMP、TensorRT和多种立体模型；LightStereo面向高效2D代价聚合。
- 适合程度：高。建议先作为SGBM的可替换深度后端，在同一project2000验证集比较0～30m MAE、延迟和显存。
- 接入方式：保持当前CenterNet和gate不变，只替换`sgbm_depth/quality`来源。

### RAFT-Stereo realtime

- 仓库：https://github.com/princeton-vl/RAFT-Stereo
- 官方提供`raftstereo-realtime.pth`和减少迭代次数的实时配置。
- 适合程度：中高。深度质量通常强于传统SGBM，但旧环境和可选CUDA相关算子需要单独兼容验证。
- 接入方式：先离线生成视差缓存，避免每轮训练重复运行深度网络。

### ADStereo_fast

- 仓库：https://github.com/cocowy1/ADStereo
- 提供轻量fast版本，重点处理下采样细节损失和视差对齐。
- 适合程度：中高。对远距离小车辆的细节保持思路与当前问题匹配，但需要先核对许可证、权重和真实延迟。

### YOLOStereo3D / visualDet3D

- 仓库：https://github.com/Owen-Liuyuxuan/visualDet3D
- 使用轻量立体匹配特征增强单阶段3D检测，目标是避免重型稠密3D体素。
- 适合程度：中。其“左右特征关联后再做3D头”的设计值得作为当前Gate v3之后的结构参考，但直接迁移会重写数据和检测头。

### StreamDSGN

- 仓库：https://github.com/weiyangdaren/streamDSGN-pytorch
- 面向连续帧双目3D检测和流式感知。
- 适合程度：中。园区车更需要稳定跟踪、速度和TTC，可借鉴时序特征复用；不建议当前直接替换模型。

## 3. 暂不采用

### DSGN / DSGN++

- 仓库：https://github.com/dvlab-research/DSGN
- 使用平面扫描体和3D几何体积，精度路线明确，但官方说明训练显存接近29GB，并使用LiDAR生成深度监督。
- 不适合当前原因：结构重、环境老、训练数据与传感器监督发生变化，不符合2000帧快速迭代路线。

### FoundationStereo / Fast-FoundationStereo

- 仓库：https://github.com/NVlabs/FoundationStereo
- 快速版：https://github.com/NVlabs/Fast-FoundationStereo
- 可作为未来园区数据伪标签教师或零样本深度基线；暂不直接作为车端模块，先核对许可证、PyTorch/CUDA环境和端到端延迟。

## 4. 建议实验顺序

```text
Gate v3距离加权
    ↓
SGBM左右一致性与质量特征
    ↓
LightStereo或RAFT-Stereo离线深度A/B
    ↓
连续帧跟踪、速度和TTC
    ↓
YOLOStereo3D式左右特征关联
```

每一步只改变一个变量，并统一报告0～15m、15～30m、30～50m的MAE、失败率、Car 3D AP、推理延迟和显存。
