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


def test_summarize_reports_center_xy_oracle_gain():
  record = {
      'matched': True, 'iou_3d': 0.4,
      'iou_3d_fix_depth': 0.5, 'iou_3d_fix_center_xy': 0.7,
      'iou_3d_fix_dimensions': 0.45, 'iou_3d_fix_yaw': 0.42,
      'depth_abs_error_m': 1.0, 'location_xy_error_m': 0.3}
  summary = MODULE.summarize([record])
  assert abs(summary['iou_3d_gain_fix_center_xy'] - 0.3) < 1e-6
  assert summary['location_xy_error_m_mean'] == 0.3
