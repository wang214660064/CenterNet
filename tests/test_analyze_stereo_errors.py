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


def test_add_depth_diagnostics_measures_gate_choice_and_regret():
  record = {'matched': True}
  diagnostic = {
      'z_direct_m': 19.0, 'z_stereo_m': 20.2, 'z_final_m': 20.1,
      'z_sgbm_m': 18.0, 'sgbm_quality': 0.9,
      'geometry_offset_m': 2.0, 'residual_offset_m': 0.2,
      'predicted_offset_m': 2.2, 'safe_offset_m': 2.2,
      'uncertainty_m': 0.3, 'learned_gate': 0.9, 'effective_gate': 0.9,
      'stereo_safety_allowed': True, 'fallback_reason': None}
  MODULE.add_depth_diagnostics(record, diagnostic, gt_depth=20.0)
  assert record['diagnostics_available'] is True
  assert abs(record['direct_depth_abs_error_m'] - 1.0) < 1e-6
  assert abs(record['stereo_center_abs_error_m'] - 0.2) < 1e-6
  assert abs(record['sgbm_center_abs_error_m'] - 2.0) < 1e-6
  assert abs(record['offset_depth_gain_m'] - 1.8) < 1e-6
  assert record['offset_improved'] is True
  assert record['preferred_depth_candidate'] == 'stereo'
  assert record['gate_choice_correct'] is True
  assert abs(record['gate_blend_gain_m'] - 0.1) < 1e-6
  assert abs(record['soft_vs_hard_gate_gain_m'] - 0.1) < 1e-6


def test_quality_bucket_boundaries():
  assert MODULE.quality_bucket(None) == '无诊断数据'
  assert MODULE.quality_bucket(0.0, sgbm_depth=0.0) == '无效SGBM'
  assert MODULE.quality_bucket(0.49) == '低质量<0.5'
  assert MODULE.quality_bucket(0.5) == '中质量0.5-0.8'
  assert MODULE.quality_bucket(0.8) == '高质量>=0.8'


def test_invalid_raw_sgbm_is_not_counted_as_corrected_candidate():
  record = {'matched': True}
  diagnostic = {
      'z_direct_m': 10.0, 'z_sgbm_m': 0.0, 'z_stereo_m': 2.0,
      'z_final_m': 10.0, 'sgbm_quality': 0.0, 'effective_gate': 0.0}
  MODULE.add_depth_diagnostics(record, diagnostic, gt_depth=9.0)
  assert record['direct_depth_abs_error_m'] == 1.0
  assert 'stereo_center_abs_error_m' not in record
  assert 'candidate_oracle_abs_error_m' not in record
