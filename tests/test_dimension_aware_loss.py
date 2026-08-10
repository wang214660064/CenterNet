import sys
from pathlib import Path

import torch


LIB = Path(__file__).parents[1] / 'src' / 'lib'
sys.path.insert(0, str(LIB))

from trains.stereo_ddd import dimension_aware_l1_loss


def test_dimension_aware_loss_is_zero_at_target():
  target = torch.tensor([[[1.5, 1.8, 4.0]]])
  loss = dimension_aware_l1_loss(
      target, target, torch.ones((1, 1, 1)))
  torch.testing.assert_close(loss, torch.tensor(0.0))


def test_dimension_aware_loss_penalizes_equal_meter_error_by_relative_size():
  target = torch.tensor([[[1.0, 1.0, 4.0]]])
  height_error = target.clone()
  length_error = target.clone()
  height_error[..., 0] += 0.2
  length_error[..., 2] += 0.2
  weight = torch.ones((1, 1, 1))

  height_loss = dimension_aware_l1_loss(height_error, target, weight)
  length_loss = dimension_aware_l1_loss(length_error, target, weight)
  assert height_loss > length_loss


def test_dimension_aware_loss_has_gradient():
  predicted = torch.tensor([[[2.0, 2.0, 6.0]]], requires_grad=True)
  target = torch.tensor([[[1.0, 1.0, 4.0]]])
  loss = dimension_aware_l1_loss(
      predicted, target, torch.ones((1, 1, 1)))

  loss.backward()
  assert predicted.grad is not None
  assert torch.isfinite(predicted.grad).all()
  assert predicted.grad.abs().sum() > 0
