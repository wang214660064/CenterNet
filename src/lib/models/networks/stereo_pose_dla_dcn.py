"""带SGBM深度残差分支的DLA-34 CenterNet。"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .pose_dla_dcn import DLASeg, fill_fc_weights
from .stereo_depth_offset import geometric_surface_offset


class ECAAttention(nn.Module):
  """轻量通道注意力，不明显增加模型体量。"""
  def __init__(self, channels, kernel_size=3):
    super(ECAAttention, self).__init__()
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.conv = nn.Conv1d(
        1, 1, kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2, bias=False)

  def forward(self, features):
    weights = self.pool(features).squeeze(-1).transpose(1, 2)
    weights = torch.sigmoid(self.conv(weights))
    weights = weights.transpose(1, 2).unsqueeze(-1)
    return features * weights


class StereoDLASeg(DLASeg):
  def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
               last_level, head_conv, out_channel=0):
    stereo_heads = ('depth_offset', 'depth_log_variance', 'depth_gate')
    regular_heads = {name: channels for name, channels in heads.items()
                     if name not in stereo_heads}
    super(StereoDLASeg, self).__init__(
        base_name, regular_heads, pretrained, down_ratio, final_kernel,
        last_level, head_conv, out_channel)
    # 旧v4权重不包含此头；零初始化使旧权重推理时仍使用原二维中心。
    if hasattr(self, 'proj_center_offset'):
      conv_layers = [module for module in self.proj_center_offset.modules()
                     if isinstance(module, nn.Conv2d)]
      nn.init.zeros_(conv_layers[-1].weight)
      if conv_layers[-1].bias is not None:
        nn.init.zeros_(conv_layers[-1].bias)
    feature_channels = self.base.channels[self.first_level]
    # SGBM质量编码器输入：深度、有效比例、局部离散度、深度梯度。
    self.stereo_quality_encoder = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))
    self.stereo_coarse_fusion = nn.Sequential(
        nn.Conv2d(feature_channels + 64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))
    self.stereo_fusion = nn.Sequential(
        nn.Conv2d(feature_channels + 64 + 64, head_conv,
                  kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))
    self.stereo_attention = ECAAttention(head_conv)
    # 目标中心不仅使用单像素，还按预测框尺寸选择不同邻域进行聚合。
    self.target_context = nn.Sequential(
        nn.Conv2d(head_conv * 3 + 1, head_conv,
                  kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))
    self.depth_offset = nn.Conv2d(head_conv, 1, kernel_size=1)
    # 学习几何先验的可信程度，避免把“车长一半”强行用于遮挡或错误视差。
    self.depth_geometry_gate = nn.Conv2d(head_conv, 1, kernel_size=1)
    self.depth_log_variance = nn.Conv2d(head_conv, 1, kernel_size=1)
    self.depth_gate = nn.Conv2d(head_conv, 1, kernel_size=1)
    fill_fc_weights(self.stereo_quality_encoder)
    fill_fc_weights(self.stereo_coarse_fusion)
    fill_fc_weights(self.stereo_fusion)
    fill_fc_weights(self.target_context)
    nn.init.zeros_(self.depth_offset.weight)
    nn.init.zeros_(self.depth_offset.bias)
    nn.init.zeros_(self.depth_geometry_gate.weight)
    nn.init.zeros_(self.depth_geometry_gate.bias)
    nn.init.zeros_(self.depth_log_variance.weight)
    nn.init.zeros_(self.depth_log_variance.bias)
    nn.init.zeros_(self.depth_gate.weight)
    nn.init.zeros_(self.depth_gate.bias)
    self.heads = dict(regular_heads)
    self.heads.update({
        'depth_offset': 1, 'depth_log_variance': 1, 'depth_gate': 1})
    self.train_stereo_only = False
    self.train_dimension_only = False

  def train(self, mode=True):
    super(StereoDLASeg, self).train(mode)
    if mode and self.train_stereo_only:
      # 冻结分支保持eval，防止小数据集继续改变BatchNorm统计量。
      trainable = {
          'stereo_quality_encoder', 'stereo_coarse_fusion', 'stereo_fusion',
          'stereo_attention', 'target_context', 'depth_offset',
          'depth_geometry_gate', 'depth_log_variance',
          'depth_gate', 'proj_center_offset'}
      if self.train_dimension_only:
        trainable.add('dim')
      for name, module in self.named_children():
        if name not in trainable:
          module.eval()
    return self

  def forward(self, image, sgbm_depth=None, sgbm_quality=None):
    features = self.base(image)
    features = self.dla_up(features)
    fused_levels = []
    for i in range(self.last_level - self.first_level):
      fused_levels.append(features[i].clone())
    self.ida_up(fused_levels, 0, len(fused_levels))
    image_features = fused_levels[-1]

    output = {}
    for head in self.heads:
      if head not in ('depth_offset', 'depth_log_variance', 'depth_gate'):
        output[head] = self.__getattr__(head)(image_features)

    if sgbm_depth is None:
      shape = (image.shape[0], 1, image_features.shape[2], image_features.shape[3])
      sgbm_depth = image_features.new_zeros(shape)
    if sgbm_quality is None:
      sgbm_quality = torch.isfinite(sgbm_depth).to(dtype=image_features.dtype)
    valid = torch.isfinite(sgbm_depth) & (sgbm_depth > 0)
    safe_depth = torch.where(valid, sgbm_depth, torch.zeros_like(sgbm_depth))
    normalized_depth = torch.log1p(torch.clamp(safe_depth, max=80.0)) / np.log1p(80.0)
    quality = torch.where(valid, sgbm_quality, torch.zeros_like(sgbm_quality))
    quality = torch.clamp(quality, 0, 1)
    local_mean = F.avg_pool2d(normalized_depth, 5, stride=1, padding=2)
    local_square_mean = F.avg_pool2d(
        normalized_depth * normalized_depth, 5, stride=1, padding=2)
    local_std = torch.sqrt(torch.clamp(
        local_square_mean - local_mean * local_mean, min=0.0))
    gradient_x = F.pad(torch.abs(
        normalized_depth[:, :, :, 1:] - normalized_depth[:, :, :, :-1]),
        (0, 1, 0, 0))
    gradient_y = F.pad(torch.abs(
        normalized_depth[:, :, 1:, :] - normalized_depth[:, :, :-1, :]),
        (0, 0, 0, 1))
    depth_gradient = 0.5 * (gradient_x + gradient_y)
    quality_features = self.stereo_quality_encoder(torch.cat(
        (normalized_depth, quality, local_std, depth_gradient), dim=1))

    coarse_image = F.avg_pool2d(image_features, 2, stride=2)
    coarse_quality = F.avg_pool2d(quality_features, 2, stride=2)
    coarse_features = self.stereo_coarse_fusion(torch.cat(
        (coarse_image, coarse_quality), dim=1))
    coarse_features = F.interpolate(
        coarse_features, size=image_features.shape[-2:], mode='bilinear',
        align_corners=False)
    stereo_features = self.stereo_fusion(torch.cat(
        (image_features, quality_features, coarse_features), dim=1))
    stereo_features = self.stereo_attention(stereo_features)

    objectness = torch.sigmoid(output['hm']).amax(dim=1, keepdim=True).detach()
    box_size = torch.abs(output['wh']).amax(dim=1, keepdim=True).detach()
    box_size = torch.clamp(box_size, min=1.0)
    anchors = image_features.new_tensor([4.0, 12.0, 32.0]).view(1, 3, 1, 1)
    scale_weights = torch.softmax(
        -torch.abs(torch.log(box_size) - torch.log(anchors)), dim=1)
    pooled_scales = torch.stack((
        F.avg_pool2d(stereo_features, 3, stride=1, padding=1),
        F.avg_pool2d(stereo_features, 7, stride=1, padding=3),
        F.avg_pool2d(stereo_features, 15, stride=1, padding=7)), dim=1)
    target_average = (
        pooled_scales * scale_weights.unsqueeze(2)).sum(dim=1)
    target_features = self.target_context(torch.cat((
        stereo_features,
        target_average,
        F.max_pool2d(stereo_features, 7, stride=1, padding=3),
        objectness), dim=1))
    geometry = geometric_surface_offset(output['dim'], output['rot'])
    # 质量越差，几何先验贡献越小；其余误差由残差头学习。
    geometry_weight = torch.sigmoid(
        self.depth_geometry_gate(target_features))
    geometry_weight = geometry_weight * quality.detach()
    geometry_offset = geometry * geometry_weight
    residual_offset = self.depth_offset(target_features)
    output['depth_geometry_offset'] = geometry_offset
    output['depth_offset_residual'] = residual_offset
    output['depth_offset'] = geometry_offset + residual_offset
    output['depth_log_variance'] = self.depth_log_variance(target_features)
    output['depth_gate'] = self.depth_gate(target_features)
    return [output]


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
  return StereoDLASeg(
      'dla{}'.format(num_layers), heads, pretrained=True,
      down_ratio=down_ratio, final_kernel=1, last_level=5,
      head_conv=head_conv)
