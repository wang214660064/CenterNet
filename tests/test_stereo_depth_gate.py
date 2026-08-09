import sys
from pathlib import Path

import torch


LIB = Path(__file__).parents[1] / 'src' / 'lib'
sys.path.insert(0, str(LIB))

from models.networks.stereo_depth_offset import fuse_stereo_depth


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
