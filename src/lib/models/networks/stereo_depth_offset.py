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


def stereo_huber_uncertainty_loss(predicted_offset, predicted_log_variance,
                                  target_offset, valid_mask, delta=1.0,
                                  calibration_weight=0.05):
  """Huber回归加小权重不确定性校准，避免方差项主导offset学习。"""
  valid_mask = valid_mask.to(dtype=predicted_offset.dtype)
  valid_count = torch.clamp(valid_mask.sum(), min=1.0)
  huber = F.smooth_l1_loss(
      predicted_offset, target_offset, reduction='none', beta=delta)
  absolute_error = torch.abs(predicted_offset.detach() - target_offset)
  target_log_variance = torch.log(torch.clamp(
      absolute_error * absolute_error, min=1e-2, max=1e2))
  calibration = F.smooth_l1_loss(
      torch.clamp(predicted_log_variance, min=-5.0, max=5.0),
      target_log_variance, reduction='none', beta=1.0)
  loss = huber + calibration_weight * calibration
  return (loss * valid_mask).sum() / valid_count


def campus_distance_weights(depth, near_distance=15.0, core_distance=30.0,
                            warning_distance=50.0, near_weight=2.0,
                            core_weight=1.5, warning_weight=0.5,
                            beyond_weight=0.0):
  """园区距离权重：优先0～30m，保留30～50m，忽略更远3D监督。"""
  return torch.where(
      depth < near_distance, torch.full_like(depth, near_weight),
      torch.where(
          depth < core_distance, torch.full_like(depth, core_weight),
          torch.where(
              depth < warning_distance,
              torch.full_like(depth, warning_weight),
              torch.full_like(depth, beyond_weight))))


def regret_focal_gate_loss(logits, stereo_error, direct_error, valid_mask,
                           distance_weight, gamma=2.0, alpha=0.5,
                           ambiguity_margin=0.2, max_regret=4.0):
  """门控Focal损失：忽略模糊样本，并强化代价高的错误选择。"""
  stereo_error = stereo_error.detach()
  direct_error = direct_error.detach()
  target = (stereo_error < direct_error).to(dtype=logits.dtype)
  error_gap = torch.abs(stereo_error - direct_error)
  clear_choice = (error_gap >= ambiguity_margin).to(dtype=logits.dtype)
  regret_weight = 1.0 + torch.clamp(error_gap, max=max_regret)
  weight = valid_mask.to(dtype=logits.dtype) * distance_weight * clear_choice
  weight = weight * regret_weight

  bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
  probability = torch.sigmoid(logits)
  probability_target = target * probability + (1.0 - target) * (1.0 - probability)
  alpha_target = target * alpha + (1.0 - target) * (1.0 - alpha)
  focal = alpha_target * torch.pow(1.0 - probability_target, gamma) * bce
  return (focal * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def fuse_stereo_depth(direct_depth, sgbm_depth, quality, predicted_offset,
                      predicted_log_variance, learned_gate_logits=None,
                      min_quality=0.5,
                      far_distance=30.0, far_min_quality=0.8,
                      max_uncertainty=10.0, far_max_uncertainty=3.0,
                      max_offset_abs=8.0, max_offset_ratio=0.15,
                      min_offset_limit=2.0, fusion_max_depth=50.0):
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
  safe_to_use = ((sgbm_depth > 0) & (sgbm_depth < fusion_max_depth) &
                 (quality >= required_quality) &
                 (corrected_depth > 0) &
                 (uncertainty <= allowed_uncertainty))
  if learned_gate_logits is None:
    # 兼容旧模型和原有单元测试。
    return (torch.where(safe_to_use, corrected_depth, direct_depth),
            safe_to_use, safe_offset)
  learned_gate = torch.sigmoid(learned_gate_logits)
  effective_gate = learned_gate * safe_to_use.to(dtype=learned_gate.dtype)
  final_depth = (effective_gate * corrected_depth +
                 (1.0 - effective_gate) * direct_depth)
  return final_depth, effective_gate, safe_offset
