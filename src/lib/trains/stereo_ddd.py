"""双目DDD训练器：原3D损失加SGBM深度offset不确定性损失。"""

from __future__ import absolute_import, division, print_function

import torch

from models.utils import _transpose_and_gather_feat
from .ddd import DddLoss, DddTrainer


class StereoDddLoss(DddLoss):
  def forward(self, outputs, batch):
    base_loss, stats = super(StereoDddLoss, self).forward(outputs, batch)
    offset_loss = 0
    for output in outputs:
      predicted_offset = _transpose_and_gather_feat(
          output['depth_offset'], batch['ind'])
      predicted_log_variance = _transpose_and_gather_feat(
          output['depth_log_variance'], batch['ind'])
      target = batch['depth_offset']
      mask = batch['depth_offset_mask'].unsqueeze(2).to(
          dtype=predicted_offset.dtype)
      valid_count = torch.clamp(mask.sum(), min=1.0)
      log_variance = torch.clamp(predicted_log_variance, min=-5.0, max=5.0)
      per_target = (
          torch.abs(predicted_offset - target) * torch.exp(-log_variance) +
          log_variance)
      offset_loss += (per_target * mask).sum() / valid_count / len(outputs)
      corrected_depth = batch['sgbm_depth'] + output['depth_offset']
      dense_log_variance = torch.clamp(
          output['depth_log_variance'], min=-5.0, max=5.0)
      uncertainty = torch.exp(0.5 * dense_log_variance)
      use_stereo = (
          (batch['sgbm_depth'] > 0) &
          (batch['sgbm_quality'] >= self.opt.stereo_min_quality) &
          (corrected_depth > 0) &
          (uncertainty <= self.opt.depth_offset_max_uncertainty))
      output['direct_dep'] = output['dep']
      output['dep'] = torch.where(use_stereo, corrected_depth, output['dep'])
    loss = base_loss + self.opt.depth_offset_weight * offset_loss
    stats['loss'] = loss
    stats['depth_offset_loss'] = offset_loss
    return loss, stats


class StereoDddTrainer(DddTrainer):
  def _get_losses(self, opt):
    loss_states = [
        'loss', 'hm_loss', 'dep_loss', 'dim_loss', 'rot_loss',
        'wh_loss', 'off_loss', 'depth_offset_loss']
    return loss_states, StereoDddLoss(opt)
