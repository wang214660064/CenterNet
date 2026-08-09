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
