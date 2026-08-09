"""双目DDD训练器：原3D损失加SGBM深度offset不确定性损失。"""

from __future__ import absolute_import, division, print_function

import torch
from torch.nn import functional as F

from models.utils import _transpose_and_gather_feat
from models.networks.stereo_depth_offset import (
    campus_distance_weights, fuse_stereo_depth, regret_focal_gate_loss,
    stereo_huber_uncertainty_loss)
from .ddd import DddLoss, DddTrainer


class StereoDddLoss(DddLoss):
  def forward(self, outputs, batch):
    base_loss, stats = super(StereoDddLoss, self).forward(outputs, batch)
    offset_loss = 0
    gate_loss = 0
    fusion_depth_loss = 0
    proj_center_loss = 0
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
            self.opt.proj_center_weight * proj_center_loss)
    stats['loss'] = loss
    stats['depth_offset_loss'] = offset_loss
    stats['depth_gate_loss'] = gate_loss
    stats['depth_fusion_loss'] = fusion_depth_loss
    stats['proj_center_loss'] = proj_center_loss
    stats['geometry_offset_mean'] = geometry_offset_mean.detach()
    stats['residual_offset_mean'] = residual_offset_mean.detach()
    return loss, stats


class StereoDddTrainer(DddTrainer):
  def _get_losses(self, opt):
    loss_states = [
        'loss', 'hm_loss', 'dep_loss', 'dim_loss', 'rot_loss',
        'wh_loss', 'off_loss', 'depth_offset_loss', 'depth_gate_loss',
        'depth_fusion_loss', 'proj_center_loss', 'geometry_offset_mean',
        'residual_offset_mean']
    return loss_states, StereoDddLoss(opt)
