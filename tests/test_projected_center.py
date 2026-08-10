import sys
from pathlib import Path
import numpy as np
import torch


LIB = Path(__file__).parents[1] / 'src' / 'lib'
sys.path.insert(0, str(LIB))

from models.decode import ddd_decode
from utils.ddd_utils import ddd2locrot, project_3d_center_to_image
from utils.post_process import ddd_post_process_3d
from trains.stereo_ddd import projected_center_to_camera_xy


def test_projected_center_round_trip_recovers_bottom_center_location():
  calib = np.array([
      [700.0, 0.0, 600.0, 0.0],
      [0.0, 700.0, 180.0, 0.0],
      [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
  location = np.array([2.0, 1.7, 20.0], dtype=np.float32)
  dimensions = np.array([1.6, 1.8, 4.0], dtype=np.float32)

  center_2d = project_3d_center_to_image(location, dimensions, calib)
  recovered, _ = ddd2locrot(
      center_2d, 0.0, dimensions, location[2], calib)

  np.testing.assert_allclose(recovered, location, atol=1e-5)


def test_projected_center_camera_xy_uses_calibration_and_detached_depth():
  center = torch.tensor([[[60.0, 35.0]]], requires_grad=True)
  depth = torch.tensor([[[20.0]]], requires_grad=True)
  inverse_affine = torch.tensor([
      [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
  calib = torch.tensor([[
      [100.0, 0.0, 50.0, 0.0],
      [0.0, 100.0, 40.0, 0.0],
      [0.0, 0.0, 1.0, 0.0]]])

  camera_xy = projected_center_to_camera_xy(
      center, depth.detach(), inverse_affine, calib)
  torch.testing.assert_close(camera_xy, torch.tensor([[[2.0, -1.0]]]))
  camera_xy.sum().backward()
  assert center.grad is not None
  assert depth.grad is None


def test_ddd_decode_keeps_2d_center_and_applies_projected_center_offset():
  heat = torch.zeros((1, 1, 4, 4))
  heat[0, 0, 1, 2] = 1.0
  rot = torch.zeros((1, 8, 4, 4))
  depth = torch.full((1, 1, 4, 4), 10.0)
  dim = torch.ones((1, 3, 4, 4))
  wh = torch.zeros((1, 2, 4, 4))
  wh[0, :, 1, 2] = torch.tensor([4.0, 2.0])
  reg = torch.zeros((1, 2, 4, 4))
  reg[0, :, 1, 2] = torch.tensor([0.25, 0.5])
  projected_offset = torch.zeros((1, 2, 4, 4))
  projected_offset[0, :, 1, 2] = torch.tensor([1.5, -0.5])

  dets = ddd_decode(
      heat, rot, depth, dim, wh=wh, reg=reg,
      proj_center_offset=projected_offset, K=1)

  assert dets.shape == (1, 1, 20)
  torch.testing.assert_close(dets[0, 0, :2], torch.tensor([3.75, 1.0]))
  torch.testing.assert_close(dets[0, 0, 17:19], torch.tensor([2.25, 1.5]))


def test_projected_center_offset_is_limited_by_vector_length():
  heat = torch.zeros((1, 1, 2, 2))
  heat[0, 0, 0, 0] = 1.0
  rot = torch.zeros((1, 8, 2, 2))
  depth = torch.ones((1, 1, 2, 2))
  dim = torch.ones((1, 3, 2, 2))
  projected_offset = torch.zeros((1, 2, 2, 2))
  projected_offset[0, :, 0, 0] = torch.tensor([60.0, 80.0])

  dets = ddd_decode(
      heat, rot, depth, dim, proj_center_offset=projected_offset,
      proj_center_max_offset=50.0, K=1)

  # (60,80)的长度为100，限幅后为(30,40)；无reg时原中心为(0.5,0.5)。
  torch.testing.assert_close(dets[0, 0, :2], torch.tensor([30.5, 40.5]))


def test_post_process_uses_projected_center_for_3d_and_2d_center_for_bbox():
  calib = np.array([
      [100.0, 0.0, 50.0, 0.0],
      [0.0, 100.0, 40.0, 0.0],
      [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
  # 投影中心(60, 35)，2D框中心(55, 38)，框宽高20x10。
  row = np.array([
      60.0, 35.0, 0.9, 0.0, 20.0,
      1.6, 1.8, 4.0, 20.0, 10.0, 55.0, 38.0], dtype=np.float32)
  result = ddd_post_process_3d(
      [{1: row.reshape(1, -1)}], [calib], include_wh=True,
      include_projected_center=True)[0][1][0]

  np.testing.assert_allclose(result[1:5], [45.0, 33.0, 65.0, 43.0])
  # x=(60-50)*20/100=2；y恢复后加回半个车高，底面中心y为-0.2。
  np.testing.assert_allclose(result[8:11], [2.0, -0.2, 20.0], atol=1e-5)
