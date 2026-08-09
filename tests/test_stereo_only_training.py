import sys
from pathlib import Path

import torch


SRC = Path(__file__).parents[1] / 'src'
sys.path.insert(0, str(SRC))
import main


class TinyStereoModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.base = torch.nn.Linear(2, 2)
    self.hm = torch.nn.Linear(2, 1)
    self.wh = torch.nn.Linear(2, 2)
    self.reg = torch.nn.Linear(2, 2)
    self.dep = torch.nn.Linear(2, 1)
    self.dim = torch.nn.Linear(2, 3)
    self.rot = torch.nn.Linear(2, 8)
    self.stereo_fusion = torch.nn.Linear(2, 2)
    self.stereo_quality_encoder = torch.nn.Linear(2, 2)
    self.stereo_coarse_fusion = torch.nn.Linear(2, 2)
    self.stereo_attention = torch.nn.Linear(2, 2)
    self.target_context = torch.nn.Linear(2, 2)
    self.depth_offset = torch.nn.Linear(2, 1)
    self.depth_geometry_gate = torch.nn.Linear(2, 1)
    self.depth_log_variance = torch.nn.Linear(2, 1)
    self.depth_gate = torch.nn.Linear(2, 1)
    self.train_stereo_only = False
    self.train_stereo_3d_heads = False


def test_configure_stereo_only_training():
  model = TinyStereoModel()
  parameters = main.configure_stereo_only_training(model)
  assert model.train_stereo_only
  assert not model.base.weight.requires_grad
  assert model.stereo_fusion.weight.requires_grad
  assert model.depth_offset.weight.requires_grad
  assert model.depth_geometry_gate.weight.requires_grad
  assert model.depth_log_variance.weight.requires_grad
  assert model.depth_gate.weight.requires_grad
  assert sum(parameter.numel() for parameter in parameters) == 42


def test_configure_stereo_3d_head_training():
  model = TinyStereoModel()
  groups = main.configure_stereo_3d_head_training(model, 2e-6, 5e-7)
  assert model.train_stereo_only
  assert model.train_stereo_3d_heads
  assert not model.base.weight.requires_grad
  assert not model.hm.weight.requires_grad
  assert model.dep.weight.requires_grad
  assert model.dim.weight.requires_grad
  assert model.rot.weight.requires_grad
  assert model.depth_offset.weight.requires_grad
  assert groups[0]['lr'] == 2e-6
  assert groups[1]['lr'] == 5e-7
  assert groups[0]['name'] == 'stereo'
  assert groups[1]['name'] == 'ddd_heads'
