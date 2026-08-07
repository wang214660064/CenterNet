"""带SGBM深度残差分支的DLA-34 CenterNet。"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch import nn

from .pose_dla_dcn import DLASeg, fill_fc_weights


class StereoDLASeg(DLASeg):
  def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
               last_level, head_conv, out_channel=0):
    stereo_heads = ('depth_offset', 'depth_log_variance')
    regular_heads = {name: channels for name, channels in heads.items()
                     if name not in stereo_heads}
    super(StereoDLASeg, self).__init__(
        base_name, regular_heads, pretrained, down_ratio, final_kernel,
        last_level, head_conv, out_channel)
    feature_channels = self.base.channels[self.first_level]
    self.stereo_fusion = nn.Sequential(
        nn.Conv2d(feature_channels + 2, head_conv, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))
    self.depth_offset = nn.Conv2d(head_conv, 1, kernel_size=1)
    self.depth_log_variance = nn.Conv2d(head_conv, 1, kernel_size=1)
    fill_fc_weights(self.stereo_fusion)
    nn.init.zeros_(self.depth_offset.weight)
    nn.init.zeros_(self.depth_offset.bias)
    nn.init.zeros_(self.depth_log_variance.weight)
    nn.init.zeros_(self.depth_log_variance.bias)
    self.heads = dict(regular_heads)
    self.heads.update({'depth_offset': 1, 'depth_log_variance': 1})

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
      if head not in ('depth_offset', 'depth_log_variance'):
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
    stereo_features = self.stereo_fusion(torch.cat(
        (image_features, normalized_depth, torch.clamp(quality, 0, 1)), dim=1))
    output['depth_offset'] = self.depth_offset(stereo_features)
    output['depth_log_variance'] = self.depth_log_variance(stereo_features)
    return [output]


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
  return StereoDLASeg(
      'dla{}'.format(num_layers), heads, pretrained=True,
      down_ratio=down_ratio, final_kernel=1, last_level=5,
      head_conv=head_conv)
