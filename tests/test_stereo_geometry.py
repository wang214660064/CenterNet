import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / 'src' / 'stereo_kitti_demo.py'
SPEC = importlib.util.spec_from_file_location('stereo_kitti_demo', MODULE_PATH)
STEREO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEREO)


class StereoGeometryTest(unittest.TestCase):
  def test_kitti_projection_to_baseline_and_depth(self):
    fx = 700.0
    p2 = np.array([[fx, 0, 600, 35], [0, fx, 180, 0], [0, 0, 1, 0]], dtype=float)
    p3 = np.array([[fx, 0, 600, -343], [0, fx, 180, 0], [0, 0, 1, 0]], dtype=float)
    params = STEREO.stereo_parameters(p2, p3)
    self.assertTrue(np.isclose(params['baseline'], 0.54))
    disparity = np.full((2, 3), 37.8, dtype=np.float32)
    depth = STEREO.disparity_to_depth(disparity, params, max_depth=80)
    self.assertTrue(np.allclose(depth, 10.0))

  def test_measure_detection_rejects_invalid_depth(self):
    disparity = np.zeros((100, 100), dtype=np.float32)
    depth = np.full((100, 100), np.nan, dtype=np.float32)
    params = {'fx': 700.0, 'fy': 700.0, 'cx': 50.0, 'cy': 50.0,
              'baseline': 0.54, 'disparity_offset': 0.0}
    result = STEREO.measure_detection([10, 10, 90, 90], disparity, depth, params)
    self.assertIsNone(result)


if __name__ == '__main__':
  unittest.main()
