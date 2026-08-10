"""双目DDD训练器：原3D损失加SGBM深度offset不确定性损失。"""

from __future__ import absolute_import, division, print_function

import torch
from torch.nn import functional as F

from models.utils import _transpose_and_gather_feat
from models.networks.stereo_depth_offset import (
    campus_distance_weights, fuse_stereo_depth, regret_focal_gate_loss,
    stereo_huber_uncertainty_loss)
from .ddd import DddLoss, DddTrainer


def projected_center_to_camera_xy(center_output, depth, inverse_affine, calib):
  """将输出特征图上的投影中心反投影为相机坐标x/y。"""
  ones = torch.ones_like(center_output[..., :1])
  homogeneous_center = torch.cat([center_output, ones], dim=2)
  center_image = torch.bmm(
      homogeneous_center, inverse_affine.transpose(1, 2))

  p00 = calib[:, None, 0, 0]
  p02 = calib[:, None, 0, 2]
  p03 = calib[:, None, 0, 3]
  p11 = calib[:, None, 1, 1]
  p12 = calib[:, None, 1, 2]
  p13 = calib[:, None, 1, 3]
  p23 = calib[:, None, 2, 3]
  z = depth[..., 0] - p23
  x = (center_image[..., 0] * depth[..., 0] - p03 - p02 * z) / p00
  y = (center_image[..., 1] * depth[..., 0] - p13 - p12 * z) / p11
  return torch.stack([x, y], dim=2)


def projected_center_iou_surrogate_loss(
    predicted_xy, target_xy, extent_xy, weight, beta=0.1):
  """计算只作用于投影中心头的尺度归一化3D重叠代理损失。

  横向范围使用车辆朝向后的相机X方向尺寸，纵向范围使用车辆高度。
  重叠项模拟3D IoU对中心偏移的敏感度；Huber项保证完全错开时仍有梯度。
  """
  extent_xy = torch.clamp(extent_xy, min=0.1)
  normalized_error = torch.abs(predicted_xy - target_xy) / extent_xy
  overlap_xy = torch.clamp(1.0 - normalized_error, min=0.0)
  intersection_ratio = overlap_xy[..., 0] * overlap_xy[..., 1]
  iou_xy = intersection_ratio / torch.clamp(
      2.0 - intersection_ratio, min=1e-6)
  iou_loss = 1.0 - iou_xy

  fallback = F.smooth_l1_loss(
      normalized_error, torch.zeros_like(normalized_error),
      reduction='none', beta=beta).mean(dim=2)
  per_target = iou_loss + 0.1 * fallback
  scalar_weight = weight[..., 0]
  return ((per_target * scalar_weight).sum() /
          torch.clamp(scalar_weight.sum(), min=1.0))


class StereoDddLoss(DddLoss):
  def forward(self, outputs, batch):
    base_loss, stats = super(StereoDddLoss, self).forward(outputs, batch)
    offset_loss = 0
    gate_loss = 0
    fusion_depth_loss = 0
    proj_center_loss = 0
    proj_center_xy_loss = 0
    proj_center_iou_loss = 0
    geometry_offset_mean = 0
    residual_offset_mean = 0
    for output in outputs:
      predicted_offset = _transpose_and_gather_feat(
          output['depth_offset'], batch['ind'])
      predicted_log_variance = _transpose_and_gather_feat(
          output['depth_log_variance'], batch['ind'])
      predicted_gate_logits = _transpose_and_gather_feat(
          output['depth_gate'], batch['ind'])
      target = batch['depth_offset']
      target_depth = batch['dep']
      mask = batch['depth_offset_mask'].unsqueeze(2).to(
          dtype=predicted_offset.dtype)
      predicted_geometry = _transpose_and_gather_feat(
          output['depth_geometry_offset'], batch['ind'])
      predicted_residual = _transpose_and_gather_feat(
          output['depth_offset_residual'], batch['ind'])
      diagnostic_count = torch.clamp(mask.sum(), min=1.0)
      geometry_offset_mean += (
          (predicted_geometry * mask).sum() /
          diagnostic_count / len(outputs))
      residual_offset_mean += (
          (predicted_residual * mask).sum() /
          diagnostic_count / len(outputs))
      distance_weight = campus_distance_weights(
          target_depth,
          near_distance=self.opt.campus_near_distance,
          core_distance=self.opt.campus_core_distance,
          warning_distance=self.opt.campus_warning_distance,
          near_weight=self.opt.campus_near_weight,
          core_weight=self.opt.campus_core_weight,
          warning_weight=self.opt.campus_warning_weight,
          beyond_weight=self.opt.campus_beyond_weight)
      weighted_mask = mask * distance_weight
      valid_count = torch.clamp(weighted_mask.sum(), min=1.0)
      predicted_proj_center = _transpose_and_gather_feat(
          output['proj_center_offset'], batch['ind'])
      proj_center_mask = batch['proj_center_mask'].unsqueeze(2).to(
          dtype=predicted_proj_center.dtype)
      proj_center_weight = proj_center_mask * distance_weight
      proj_center_error = F.smooth_l1_loss(
          predicted_proj_center, batch['proj_center_offset'],
          reduction='none', beta=1.0)
      proj_center_loss += (
          (proj_center_error * proj_center_weight).sum() /
          torch.clamp(proj_center_weight.sum() * 2.0, min=1.0) /
          len(outputs))
      if self.opt.depth_offset_loss == 'huber':
        offset_loss += stereo_huber_uncertainty_loss(
            predicted_offset, predicted_log_variance, target, weighted_mask,
            delta=self.opt.depth_offset_huber_delta,
            calibration_weight=(
                self.opt.depth_uncertainty_calibration_weight)) / len(outputs)
      else:
        log_variance = torch.clamp(
            predicted_log_variance, min=-5.0, max=5.0)
        per_target = (
            torch.abs(predicted_offset - target) * torch.exp(-log_variance) +
            log_variance)
        offset_loss += (
            (per_target * weighted_mask).sum() /
            valid_count / len(outputs))
      output['direct_dep'] = output['dep']
      output['dep'], output['stereo_gate_mask'], output['safe_depth_offset'] = (
          fuse_stereo_depth(
              output['direct_dep'], batch['sgbm_depth'], batch['sgbm_quality'],
              output['depth_offset'], output['depth_log_variance'],
              learned_gate_logits=output['depth_gate'],
              min_quality=self.opt.stereo_min_quality,
              far_distance=self.opt.stereo_far_distance,
              far_min_quality=self.opt.stereo_far_min_quality,
              max_uncertainty=self.opt.depth_offset_max_uncertainty,
              far_max_uncertainty=self.opt.depth_offset_far_max_uncertainty,
              max_offset_abs=self.opt.depth_offset_max_abs,
              max_offset_ratio=self.opt.depth_offset_max_ratio,
              min_offset_limit=self.opt.depth_offset_min_limit,
              fusion_max_depth=self.opt.campus_warning_distance))
      final_depth = _transpose_and_gather_feat(output['dep'], batch['ind'])
      direct_depth = _transpose_and_gather_feat(
          output['direct_dep'], batch['ind'])
      safe_offset = _transpose_and_gather_feat(
          output['safe_depth_offset'], batch['ind'])
      sgbm_depth = _transpose_and_gather_feat(
          batch['sgbm_depth'], batch['ind'])
      sgbm_quality = _transpose_and_gather_feat(
          batch['sgbm_quality'], batch['ind'])
      corrected_depth = sgbm_depth + safe_offset

      # v6a只让梯度更新投影中心头。最终深度显式detach，避免中心损失
      # 反向改变已经验证过的SGBM融合、直接深度和gate分支。
      predicted_center_output = (
          batch['proj_center_base'] + predicted_proj_center)
      predicted_camera_xy = projected_center_to_camera_xy(
          predicted_center_output, final_depth.detach(),
          batch['proj_center_inverse_affine'].to(
              dtype=predicted_proj_center.dtype),
          batch['proj_center_calib'].to(dtype=predicted_proj_center.dtype))
      target_camera_xy = batch['proj_center_camera_xy'].to(
          dtype=predicted_proj_center.dtype)
      finite_xy = (torch.isfinite(predicted_camera_xy).all(dim=2, keepdim=True) &
                   torch.isfinite(target_camera_xy).all(dim=2, keepdim=True) &
                   (final_depth.detach() > 0.1)).to(
                       dtype=predicted_proj_center.dtype)
      xy_weight = proj_center_weight * finite_xy
      xy_error = F.smooth_l1_loss(
          predicted_camera_xy, target_camera_xy, reduction='none',
          beta=self.opt.proj_center_xy_beta)
      proj_center_xy_loss += (
          (xy_error * xy_weight).sum() /
          torch.clamp(xy_weight.sum() * 2.0, min=1.0) /
          len(outputs))

      # v7使用车辆真实尺度归一化中心误差，使同样的米制偏移对小目标
      # 产生更大惩罚。该代理损失只更新proj_center_offset头。
      target_extent_xy = batch['proj_center_extent_xy'].to(
          dtype=predicted_proj_center.dtype)
      finite_extent = (
          torch.isfinite(target_extent_xy).all(dim=2, keepdim=True) &
          (target_extent_xy > 0.1).all(dim=2, keepdim=True)).to(
              dtype=predicted_proj_center.dtype)
      iou_weight = xy_weight * finite_extent
      proj_center_iou_loss += projected_center_iou_surrogate_loss(
          predicted_camera_xy, target_camera_xy, target_extent_xy,
          iou_weight, beta=self.opt.proj_center_iou_beta) / len(outputs)

      far = sgbm_depth >= self.opt.stereo_far_distance
      required_quality = torch.where(
          far, torch.full_like(sgbm_quality, self.opt.stereo_far_min_quality),
          torch.full_like(sgbm_quality, self.opt.stereo_min_quality))
      uncertainty_limit = torch.where(
          far, torch.full_like(
              predicted_log_variance,
              self.opt.depth_offset_far_max_uncertainty),
          torch.full_like(
              predicted_log_variance,
              self.opt.depth_offset_max_uncertainty))
      uncertainty = torch.exp(0.5 * torch.clamp(
          predicted_log_variance, min=-5.0, max=5.0))
      safe_mask = ((sgbm_depth > 0) & (corrected_depth > 0) &
                   (sgbm_depth < self.opt.campus_warning_distance) &
                   (sgbm_quality >= required_quality) &
                   (uncertainty <= uncertainty_limit)).to(
                       dtype=predicted_gate_logits.dtype)
      gate_mask = mask * safe_mask

      # 真值只用于生成训练门控标签；推理阶段不会读取标签。
      stereo_error = torch.abs(corrected_depth - target_depth)
      direct_error = torch.abs(direct_depth - target_depth)
      gate_loss += regret_focal_gate_loss(
          predicted_gate_logits, stereo_error, direct_error, gate_mask,
          distance_weight, gamma=self.opt.depth_gate_focal_gamma,
          alpha=self.opt.depth_gate_focal_alpha,
          ambiguity_margin=self.opt.depth_gate_ambiguity_margin,
          max_regret=self.opt.depth_gate_max_regret) / len(outputs)
      fusion_per_target = F.smooth_l1_loss(
          final_depth, target_depth, reduction='none', beta=1.0)
      fusion_weight = gate_mask * distance_weight
      fusion_depth_loss += (
          (fusion_per_target * fusion_weight).sum() /
          torch.clamp(fusion_weight.sum(), min=1.0) / len(outputs))
    loss = (base_loss + self.opt.depth_offset_weight * offset_loss +
            self.opt.depth_gate_weight * gate_loss +
            self.opt.depth_fusion_weight * fusion_depth_loss +
            self.opt.proj_center_weight * proj_center_loss +
            self.opt.proj_center_xy_weight * proj_center_xy_loss +
            self.opt.proj_center_iou_weight * proj_center_iou_loss)
    stats['loss'] = loss
    stats['depth_offset_loss'] = offset_loss
    stats['depth_gate_loss'] = gate_loss
    stats['depth_fusion_loss'] = fusion_depth_loss
    stats['proj_center_loss'] = proj_center_loss
    stats['proj_center_xy_loss'] = proj_center_xy_loss
    stats['proj_center_iou_loss'] = proj_center_iou_loss
    stats['geometry_offset_mean'] = geometry_offset_mean.detach()
    stats['residual_offset_mean'] = residual_offset_mean.detach()
    return loss, stats


class StereoDddTrainer(DddTrainer):
  def _get_losses(self, opt):
    loss_states = [
        'loss', 'hm_loss', 'dep_loss', 'dim_loss', 'rot_loss',
        'wh_loss', 'off_loss', 'depth_offset_loss', 'depth_gate_loss',
        'depth_fusion_loss', 'proj_center_loss', 'geometry_offset_mean',
        'proj_center_xy_loss', 'proj_center_iou_loss',
        'residual_offset_mean']
    return loss_states, StereoDddLoss(opt)
