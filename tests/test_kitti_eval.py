"""KITTI AP_R40结果解析测试。"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (Path(__file__).parents[1] / 'src' / 'lib' / 'datasets' /
               'dataset' / 'kitti.py')
SPEC = importlib.util.spec_from_file_location('kitti_dataset', MODULE_PATH)
KITTI_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KITTI_MODULE)


class KittiEvalTest(unittest.TestCase):
  def test_ap_r40_skips_first_recall_point(self):
    with tempfile.TemporaryDirectory() as directory:
      stats = Path(directory) / 'stats_car_detection_3d.txt'
      # 三行分别对应Easy、Moderate、Hard；第一个点不计入AP_R40。
      stats.write_text('\n'.join([
          ' '.join(['0'] + ['0.8'] * 40),
          ' '.join(['0'] + ['0.6'] * 40),
          ' '.join(['0'] + ['0.4'] * 40),
      ]))
      KITTI_MODULE.KITTI._save_ap_r40(directory)
      result = json.loads(
          (Path(directory) / 'kitti_ap_r40.json').read_text())
      metric = result['car']['3d']
      self.assertAlmostEqual(metric['easy'], 80.0)
      self.assertAlmostEqual(metric['moderate'], 60.0)
      self.assertAlmostEqual(metric['hard'], 40.0)


if __name__ == '__main__':
  unittest.main()
