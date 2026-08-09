import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "lib"))

from models.networks.stereo_depth_offset import (  # noqa: E402
    StereoDepthOffsetHead,
    stereo_huber_uncertainty_loss,
    stereo_offset_loss,
    geometric_surface_offset,
)


class StereoDepthOffsetHeadTest(unittest.TestCase):
  def test_output_shape_and_invalid_depth_mask(self):
    head = StereoDepthOffsetHead(feature_channels=16, hidden_channels=8)
    features = torch.randn(2, 16, 24, 80)
    depth = torch.full((2, 1, 96, 320), 12.0)
    quality = torch.full((2, 1, 96, 320), 0.8)
    depth[0, :, :8, :8] = float('nan')
    output = head(features, depth, quality)
    self.assertEqual(tuple(output['depth_offset'].shape), (2, 1, 24, 80))
    self.assertEqual(tuple(output['depth_log_variance'].shape), (2, 1, 24, 80))
    self.assertEqual(tuple(output['sgbm_valid_mask'].shape), (2, 1, 24, 80))

  def test_masked_loss_ignores_invalid_target(self):
    prediction = torch.tensor([[[[2.0, 100.0]]]])
    log_variance = torch.zeros_like(prediction)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[True, False]]]])
    loss = stereo_offset_loss(prediction, log_variance, target, mask)
    self.assertAlmostEqual(loss.item(), 2.0)

  def test_huber_uncertainty_loss_is_non_negative_and_has_gradient(self):
    prediction = torch.tensor([[[[0.0, 10.0]]]], requires_grad=True)
    log_variance = torch.zeros_like(prediction, requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]]])
    mask = torch.tensor([[[[True, False]]]])
    loss = stereo_huber_uncertainty_loss(
        prediction, log_variance, target, mask, delta=1.0,
        calibration_weight=0.05)
    self.assertGreaterEqual(loss.item(), 0.0)
    loss.backward()
    self.assertIsNotNone(prediction.grad)
    self.assertIsNotNone(log_variance.grad)

  def test_geometry_offset_uses_dimensions_and_rotation_without_gradient(self):
    dimensions = torch.tensor(
        [[[[1.5]], [[2.0]], [[4.0]]]], requires_grad=True)
    rotation = torch.zeros((1, 8, 1, 1), requires_grad=True)
    rotation.data[:, 1] = 2.0
    rotation.data[:, 2] = 1.0
    # 第一旋转bin还原alpha=0，此时表面到中心约为length/2=2m。
    offset = geometric_surface_offset(dimensions, rotation)
    self.assertTrue(torch.allclose(offset, torch.tensor([[[[2.0]]]]), atol=1e-5))
    self.assertFalse(offset.requires_grad)

  def test_geometry_offset_rotated_side_uses_width(self):
    dimensions = torch.tensor([[[[1.5]], [[2.0]], [[4.0]]]])
    rotation = torch.zeros((1, 8, 1, 1))
    rotation[:, 1] = 2.0
    rotation[:, 3] = 1.0
    # 第一旋转bin还原alpha=-pi/2，此时约为width/2=1m。
    offset = geometric_surface_offset(dimensions, rotation)
    self.assertTrue(torch.allclose(offset, torch.tensor([[[[1.0]]]]), atol=1e-5))


if __name__ == '__main__':
  unittest.main()
