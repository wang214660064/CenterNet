import sys
from pathlib import Path

import torch


LIB = Path(__file__).parents[1] / 'src' / 'lib'
sys.path.insert(0, str(LIB))

from models.networks.stereo_depth_offset import (
    campus_distance_weights, fuse_stereo_depth, regret_focal_gate_loss)


def test_far_gate_and_offset_limit():
  direct = torch.tensor([[[[12.0, 42.0, 45.0]]]])
  stereo = torch.tensor([[[[10.0, 40.0, 40.0]]]])
  quality = torch.tensor([[[[0.6, 0.6, 0.9]]]])
  offset = torch.tensor([[[[20.0, 20.0, 20.0]]]])
  log_variance = torch.zeros_like(offset)

  final, gate, safe_offset = fuse_stereo_depth(
      direct, stereo, quality, offset, log_variance)

  # 近距离offset限制为2m；远距离低质量时退回直接深度。
  assert safe_offset[0, 0, 0, 0].item() == 2.0
  assert final[0, 0, 0, 0].item() == 12.0
  assert not gate[0, 0, 0, 1].item()
  # 40m处最多修正6m，且高质量时允许使用。
  assert gate[0, 0, 0, 2].item()
  assert final[0, 0, 0, 2].item() == 46.0


def test_learned_gate_blends_two_depth_candidates():
  direct = torch.full((1, 1, 1, 1), 20.0)
  stereo = torch.full_like(direct, 10.0)
  quality = torch.ones_like(direct)
  offset = torch.full_like(direct, 2.0)
  log_variance = torch.zeros_like(direct)
  gate_logits = torch.zeros_like(direct)  # sigmoid后为0.5

  final, gate, _ = fuse_stereo_depth(
      direct, stereo, quality, offset, log_variance,
      learned_gate_logits=gate_logits)

  assert torch.allclose(gate, torch.full_like(gate, 0.5))
  assert torch.allclose(final, torch.full_like(final, 16.0))


def test_campus_weights_and_beyond_range_gate():
  depth = torch.tensor([[[[10.0, 20.0, 40.0, 55.0]]]])
  weights = campus_distance_weights(depth)
  assert torch.allclose(
      weights, torch.tensor([[[[2.0, 1.5, 0.5, 0.0]]]]))

  direct = torch.full_like(depth, 45.0)
  quality = torch.ones_like(depth)
  offset = torch.zeros_like(depth)
  variance = torch.zeros_like(depth)
  _, gate, _ = fuse_stereo_depth(
      direct, depth, quality, offset, variance,
      learned_gate_logits=torch.full_like(depth, 10.0))
  assert gate[0, 0, 0, 2] > 0.99
  assert gate[0, 0, 0, 3] == 0


def test_regret_focal_gate_loss_ignores_ambiguous_sample():
  logits = torch.tensor([[[0.0], [0.0]]], requires_grad=True)
  stereo_error = torch.tensor([[[0.9], [0.0]]])
  direct_error = torch.tensor([[[1.0], [2.0]]])
  mask = torch.ones_like(logits)
  distance_weight = torch.ones_like(logits)
  loss = regret_focal_gate_loss(
      logits, stereo_error, direct_error, mask, distance_weight,
      ambiguity_margin=0.2)
  self_only_clear = regret_focal_gate_loss(
      logits[:, 1:], stereo_error[:, 1:], direct_error[:, 1:],
      mask[:, 1:], distance_weight[:, 1:], ambiguity_margin=0.2)
  assert torch.allclose(loss, self_only_clear)
  loss.backward()
  assert logits.grad is not None
