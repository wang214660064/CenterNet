"""SGBM深度残差头：预测几何深度到3D框中心深度的修正量。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class StereoDepthOffsetHead(nn.Module):
  """融合图像特征、SGBM深度和质量，输出offset及其不确定性。"""

  def __init__(self, feature_channels, hidden_channels=64, max_depth=80.0):
    super(StereoDepthOffsetHead, self).__init__()
    self.max_depth = float(max_depth)
    self.features = nn.Sequential(
        nn.Conv2d(feature_channels + 2, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    self.offset = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    self.log_variance = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    nn.init.zeros_(self.offset.weight)
    nn.init.zeros_(self.offset.bias)
    nn.init.zeros_(self.log_variance.weight)
    nn.init.zeros_(self.log_variance.bias)

  def forward(self, image_features, sgbm_depth, sgbm_quality):
    target_size = image_features.shape[-2:]
    valid = torch.isfinite(sgbm_depth) & (sgbm_depth > 0)
    sgbm_depth = torch.where(valid, sgbm_depth, torch.zeros_like(sgbm_depth))
    if sgbm_depth.shape[-2:] != target_size:
      sgbm_depth = F.interpolate(
          sgbm_depth, size=target_size, mode='bilinear', align_corners=False)
      valid = F.interpolate(
          valid.to(dtype=sgbm_depth.dtype), size=target_size, mode='nearest') > 0.5
    if sgbm_quality.shape[-2:] != target_size:
      sgbm_quality = F.interpolate(
          sgbm_quality, size=target_size, mode='bilinear', align_corners=False)

    safe_depth = torch.where(valid, sgbm_depth, torch.zeros_like(sgbm_depth))
    normalized_depth = torch.log1p(
        torch.clamp(safe_depth, max=self.max_depth)) / math.log1p(self.max_depth)
    quality = torch.where(valid, sgbm_quality, torch.zeros_like(sgbm_quality))
    quality = torch.clamp(quality, min=0.0, max=1.0)
    fused = torch.cat((image_features, normalized_depth, quality), dim=1)
    hidden = self.features(fused)
    return {
        'depth_offset': self.offset(hidden),
        'depth_log_variance': self.log_variance(hidden),
        'sgbm_valid_mask': valid,
    }


def stereo_offset_loss(predicted_offset, predicted_log_variance,
                       target_offset, valid_mask):
  """带有效掩码的Laplace不确定性损失。"""
  valid_mask = valid_mask.to(dtype=predicted_offset.dtype)
  valid_count = torch.clamp(valid_mask.sum(), min=1.0)
  log_variance = torch.clamp(predicted_log_variance, min=-5.0, max=5.0)
  loss = (torch.abs(predicted_offset - target_offset) * torch.exp(-log_variance) +
          log_variance)
  return (loss * valid_mask).sum() / valid_count


def fuse_stereo_depth(direct_depth, sgbm_depth, quality, predicted_offset,
                      predicted_log_variance, min_quality=0.5,
                      far_distance=30.0, far_min_quality=0.8,
                      max_uncertainty=10.0, far_max_uncertainty=3.0,
                      max_offset_abs=8.0, max_offset_ratio=0.15,
                      min_offset_limit=2.0):
  """按距离、质量和不确定性安全地融合SGBM offset。"""
  uncertainty = torch.exp(0.5 * torch.clamp(
      predicted_log_variance, min=-5.0, max=5.0))
  offset_limit = torch.clamp(
      sgbm_depth * max_offset_ratio, min=min_offset_limit,
      max=max_offset_abs)
  safe_offset = torch.clamp(predicted_offset, -offset_limit, offset_limit)
  corrected_depth = sgbm_depth + safe_offset
  far = sgbm_depth >= far_distance
  required_quality = torch.where(
      far, torch.full_like(quality, far_min_quality),
      torch.full_like(quality, min_quality))
  allowed_uncertainty = torch.where(
      far, torch.full_like(uncertainty, far_max_uncertainty),
      torch.full_like(uncertainty, max_uncertainty))
  use_stereo = ((sgbm_depth > 0) & (quality >= required_quality) &
                (corrected_depth > 0) & (uncertainty <= allowed_uncertainty))
  return torch.where(use_stereo, corrected_depth, direct_depth), use_stereo, safe_offset
