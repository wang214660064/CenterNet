import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / 'src' / 'tools' / 'analyze_stereo_errors.py'
SPEC = importlib.util.spec_from_file_location('analyze_stereo_errors', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bbox_iou_and_distance_bucket():
  assert MODULE.bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
  assert MODULE.bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
  assert MODULE.distance_bucket(14.99) == '0-15m'
  assert MODULE.distance_bucket(15.0) == '15-30m'
  assert MODULE.distance_bucket(50.0) == '50m以上'
