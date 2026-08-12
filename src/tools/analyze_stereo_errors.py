#!/usr/bin/env python
"""分解KITTI Stereo DDD的二维、3D中心、深度、尺寸和朝向误差。"""

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np


DISTANCE_BUCKETS = ((0, 15, '0-15m'), (15, 30, '15-30m'),
                    (30, 50, '30-50m'), (50, float('inf'), '50m以上'))
CLASS_IDS = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}


def parse_args():
  parser = argparse.ArgumentParser(description='分解Stereo DDD验证误差')
  parser.add_argument('--eval-dir', default='exp/stereo_ddd/stereo_sgbm_offset_eval')
  parser.add_argument('--sgbm-json', default='exp/stereo_stage2/sgbm_depth_metrics.json')
  parser.add_argument('--diagnostics-json', default=None,
                      help='默认读取eval-dir下的目标级双目诊断JSON')
  parser.add_argument('--annotations', default='data/kitti/annotations/kitti_3dop_val.json')
  parser.add_argument('--score-thresh', type=float, default=0.25)
  parser.add_argument('--match-iou', type=float, default=0.5)
  return parser.parse_args()


def read_kitti(path, prediction=False):
  objects = []
  if not os.path.exists(path):
    return objects
  class_ranks = {}
  with open(path, 'r') as stream:
    for line in stream:
      values = line.split()
      if len(values) < 15:
        continue
      class_name = values[0]
      class_rank = class_ranks.get(class_name, 0)
      class_ranks[class_name] = class_rank + 1
      objects.append({
          'class': class_name, 'class_rank': class_rank,
          'truncation': float(values[1]),
          'occlusion': int(values[2]), 'alpha': float(values[3]),
          'bbox': np.asarray(values[4:8], dtype=np.float32),
          'dimensions': np.asarray(values[8:11], dtype=np.float32),
          'location': np.asarray(values[11:14], dtype=np.float32),
          'rotation_y': float(values[14]),
          'score': float(values[15]) if prediction and len(values) > 15 else 1.0})
  return objects


def bbox_iou(a, b):
  x1, y1 = max(a[0], b[0]), max(a[1], b[1])
  x2, y2 = min(a[2], b[2]), min(a[3], b[3])
  intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
  area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
  area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
  return intersection / max(area_a + area_b - intersection, 1e-6)


def angle_error(a, b):
  difference = (a - b + math.pi) % (2 * math.pi) - math.pi
  return abs(difference)


def bev_corners(obj):
  h, w, length = obj['dimensions']
  x, _, z = obj['location']
  corners = np.asarray([[length / 2, w / 2], [length / 2, -w / 2],
                        [-length / 2, -w / 2], [-length / 2, w / 2]],
                       dtype=np.float32)
  c, s = math.cos(obj['rotation_y']), math.sin(obj['rotation_y'])
  rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
  return corners.dot(rotation.T) + np.asarray([x, z], dtype=np.float32)


def spatial_iou(gt, pred):
  gt_bev, pred_bev = bev_corners(gt), bev_corners(pred)
  intersection, _ = cv2.intersectConvexConvex(gt_bev, pred_bev)
  gt_area = float(gt['dimensions'][1] * gt['dimensions'][2])
  pred_area = float(pred['dimensions'][1] * pred['dimensions'][2])
  bev_union = max(gt_area + pred_area - intersection, 1e-6)
  bev_iou = float(intersection / bev_union)

  gt_top = gt['location'][1] - gt['dimensions'][0]
  pred_top = pred['location'][1] - pred['dimensions'][0]
  vertical = max(0.0, min(gt['location'][1], pred['location'][1]) -
                 max(gt_top, pred_top))
  intersection_3d = float(intersection * vertical)
  gt_volume = float(np.prod(gt['dimensions']))
  pred_volume = float(np.prod(pred['dimensions']))
  return bev_iou, intersection_3d / max(gt_volume + pred_volume - intersection_3d, 1e-6)


def match_objects(gt_objects, pred_objects, score_thresh, match_iou):
  predictions = [p for p in pred_objects if p['score'] >= score_thresh]
  candidates = []
  for gt_id, gt in enumerate(gt_objects):
    for pred_id, pred in enumerate(predictions):
      if gt['class'] == pred['class']:
        candidates.append((bbox_iou(gt['bbox'], pred['bbox']), gt_id, pred_id))
  matches, used_gt, used_pred = {}, set(), set()
  for iou, gt_id, pred_id in sorted(candidates, reverse=True):
    if iou < match_iou or gt_id in used_gt or pred_id in used_pred:
      continue
    matches[gt_id] = (predictions[pred_id], iou)
    used_gt.add(gt_id)
    used_pred.add(pred_id)
  return matches


def distance_bucket(depth):
  for lower, upper, name in DISTANCE_BUCKETS:
    if lower <= depth < upper:
      return name


def quality_bucket(quality, sgbm_depth=None):
  if quality is None:
    return '无诊断数据'
  if sgbm_depth is not None and sgbm_depth <= 0:
    return '无效SGBM'
  if quality < 0.5:
    return '低质量<0.5'
  if quality < 0.8:
    return '中质量0.5-0.8'
  return '高质量>=0.8'


def load_diagnostics(path):
  """按原始帧号、类别和该类候选顺序建立索引。"""
  if not path or not os.path.exists(path):
    return {}
  with open(path, 'r', encoding='utf-8') as stream:
    payload = json.load(stream)
  indexed = {}
  for image_id, detections in payload.get('images', {}).items():
    for detection in detections:
      key = (str(image_id).zfill(6), int(detection['class_id']),
             int(detection['class_rank']))
      indexed[key] = detection
  return indexed


def add_depth_diagnostics(record, diagnostic, gt_depth):
  """将深度融合中间量展开为可统计字段。"""
  if diagnostic is None:
    record['diagnostics_available'] = False
    return
  record['diagnostics_available'] = True
  copied_fields = (
      'z_direct_m', 'z_stereo_m', 'z_final_m', 'z_sgbm_m', 'sgbm_quality',
      'geometry_offset_m', 'residual_offset_m', 'predicted_offset_m',
      'safe_offset_m', 'uncertainty_m', 'learned_gate', 'effective_gate',
      'stereo_safety_allowed', 'fallback_reason')
  for key in copied_fields:
    record[key] = diagnostic.get(key)

  direct = diagnostic.get('z_direct_m')
  raw_stereo = diagnostic.get('z_sgbm_m')
  stereo = diagnostic.get('z_stereo_m')
  final = diagnostic.get('z_final_m')
  if direct is not None:
    record['direct_depth_abs_error_m'] = abs(direct - gt_depth)
  raw_stereo_valid = raw_stereo is not None and raw_stereo > 0
  if raw_stereo_valid:
    record['sgbm_center_abs_error_m'] = abs(raw_stereo - gt_depth)
  stereo_valid = raw_stereo_valid and stereo is not None and stereo > 0
  if stereo_valid:
    record['stereo_center_abs_error_m'] = abs(stereo - gt_depth)
  if final is not None:
    record['diagnostic_final_depth_abs_error_m'] = abs(final - gt_depth)

  if direct is None or not stereo_valid or final is None:
    return
  direct_error = record['direct_depth_abs_error_m']
  stereo_error = record['stereo_center_abs_error_m']
  raw_stereo_error = record.get('sgbm_center_abs_error_m')
  if raw_stereo_error is not None:
    record['offset_depth_gain_m'] = raw_stereo_error - stereo_error
    record['offset_improved'] = stereo_error < raw_stereo_error
  oracle_error = min(direct_error, stereo_error)
  final_error = record['diagnostic_final_depth_abs_error_m']
  record['candidate_oracle_abs_error_m'] = oracle_error
  record['gate_regret_m'] = max(0.0, final_error - oracle_error)
  record['gate_blend_gain_m'] = oracle_error - final_error
  record['preferred_depth_candidate'] = (
      'stereo' if stereo_error < direct_error else 'direct')
  hard_depth = (
      stereo if diagnostic.get('effective_gate', 0.0) >= 0.5 else direct)
  hard_error = abs(hard_depth - gt_depth)
  record['hard_gate_depth_abs_error_m'] = hard_error
  record['soft_vs_hard_gate_gain_m'] = hard_error - final_error
  # 两个候选误差过于接近时，不强行判定Gate选择对错。
  if abs(direct_error - stereo_error) >= 0.2:
    selected = 'stereo' if diagnostic.get('effective_gate', 0.0) >= 0.5 else 'direct'
    record['gate_choice_correct'] = selected == record['preferred_depth_candidate']


def mean(records, key):
  values = [record[key] for record in records if record.get(key) is not None]
  return float(np.mean(values)) if values else None


def summarize(records):
  matched = [record for record in records if record['matched']]
  summary = {
      'gt_count': len(records), 'matched_count': len(matched),
      'recall_at_2d_iou_0_5': len(matched) / max(len(records), 1),
  }
  for key in ('bbox_iou', 'depth_abs_error_m', 'depth_relative_error',
              'location_x_abs_error_m', 'location_y_abs_error_m',
              'location_xy_error_m',
              'sgbm_abs_error_m', 'dimension_mae_m', 'dimension_relative_error',
              'yaw_error_deg', 'bev_iou', 'iou_3d', 'iou_3d_fix_depth',
              'iou_3d_fix_center_xy', 'iou_3d_fix_dimensions',
              'iou_3d_fix_yaw', 'direct_depth_abs_error_m',
              'sgbm_center_abs_error_m', 'stereo_center_abs_error_m',
              'diagnostic_final_depth_abs_error_m',
              'candidate_oracle_abs_error_m', 'gate_regret_m',
              'gate_blend_gain_m', 'hard_gate_depth_abs_error_m',
              'soft_vs_hard_gate_gain_m', 'offset_depth_gain_m',
              'sgbm_quality', 'effective_gate',
              'uncertainty_m', 'geometry_offset_m', 'residual_offset_m'):
    summary[key + '_mean'] = mean(matched, key)
  for component in ('depth', 'center_xy', 'dimensions', 'yaw'):
    fixed_key = 'iou_3d_fix_' + component
    gains = [r[fixed_key] - r['iou_3d'] for r in matched]
    summary['iou_3d_gain_fix_' + component] = (
        float(np.mean(gains)) if gains else None)
  valid_stereo = [r for r in matched if r.get('sgbm_abs_error_m') is not None]
  summary['final_depth_better_than_sgbm_ratio'] = (
      sum(r['depth_abs_error_m'] < r['sgbm_abs_error_m'] for r in valid_stereo) /
      max(len(valid_stereo), 1))
  summary['iou_3d_at_0_7_ratio'] = (
      sum(r['iou_3d'] >= 0.7 for r in matched) / max(len(matched), 1))
  with_diagnostics = [r for r in matched if r.get('diagnostics_available')]
  summary['diagnostics_coverage_ratio'] = (
      len(with_diagnostics) / max(len(matched), 1))
  safety_records = [
      r for r in with_diagnostics if r.get('stereo_safety_allowed') is not None]
  summary['stereo_safety_allowed_ratio'] = (
      sum(bool(r['stereo_safety_allowed']) for r in safety_records) /
      max(len(safety_records), 1))
  gate_choices = [r for r in with_diagnostics if r.get('gate_choice_correct') is not None]
  summary['gate_choice_accuracy'] = (
      sum(bool(r['gate_choice_correct']) for r in gate_choices) /
      len(gate_choices) if gate_choices else None)
  offset_records = [r for r in with_diagnostics if r.get('offset_improved') is not None]
  summary['offset_improves_ratio'] = (
      sum(bool(r['offset_improved']) for r in offset_records) /
      len(offset_records) if offset_records else None)
  return summary


def main():
  args = parse_args()
  diagnostics_path = args.diagnostics_json or os.path.join(
      args.eval_dir, 'stereo_detection_diagnostics.json')
  diagnostics = load_diagnostics(diagnostics_path)
  # 报告中记录实际解析的默认路径，方便复现。
  args.diagnostics_json = diagnostics_path
  with open(args.annotations, 'r') as stream:
    images = sorted(json.load(stream)['images'], key=lambda item: int(item['id']))
  sgbm_records = []
  if args.sgbm_json and os.path.exists(args.sgbm_json):
    with open(args.sgbm_json, 'r') as stream:
      sgbm_records = json.load(stream).get('records', [])
  sgbm_by_target = {}
  for record in sgbm_records:
    key = (record['image_id'], record['class_name'],
           tuple(round(value, 2) for value in record['bbox']))
    sgbm_by_target[key] = record

  records = []
  for eval_id, image in enumerate(images):
    image_id = '{:06d}'.format(int(image['id']))
    gt_path = os.path.join(args.eval_dir, 'ground_truth', '{:06d}.txt'.format(eval_id))
    pred_path = os.path.join(args.eval_dir, 'eval_results', '{:06d}.txt'.format(eval_id))
    gt_objects = read_kitti(gt_path)
    pred_objects = read_kitti(pred_path, prediction=True)
    matches = match_objects(gt_objects, pred_objects, args.score_thresh, args.match_iou)
    for gt_id, gt in enumerate(gt_objects):
      if gt['class'] != 'Car':
        continue
      height = float(gt['bbox'][3] - gt['bbox'][1])
      moderate = height >= 25 and gt['occlusion'] <= 1 and gt['truncation'] <= 0.3
      record = {
          'image_id': image_id, 'distance_bucket': distance_bucket(gt['location'][2]),
          'gt_depth_m': float(gt['location'][2]), 'occlusion': gt['occlusion'],
          'truncation': gt['truncation'], 'moderate': moderate,
          'matched': gt_id in matches}
      if gt_id in matches:
        pred, iou = matches[gt_id]
        bev_iou, iou_3d = spatial_iou(gt, pred)
        fixed_depth = dict(pred)
        fixed_depth['location'] = pred['location'].copy()
        fixed_depth['location'][2] = gt['location'][2]
        fixed_center_xy = dict(pred)
        fixed_center_xy['location'] = pred['location'].copy()
        fixed_center_xy['location'][:2] = gt['location'][:2]
        fixed_dimensions = dict(pred)
        fixed_dimensions['dimensions'] = gt['dimensions'].copy()
        fixed_yaw = dict(pred)
        fixed_yaw['rotation_y'] = gt['rotation_y']
        _, iou_fix_depth = spatial_iou(gt, fixed_depth)
        _, iou_fix_center_xy = spatial_iou(gt, fixed_center_xy)
        _, iou_fix_dimensions = spatial_iou(gt, fixed_dimensions)
        _, iou_fix_yaw = spatial_iou(gt, fixed_yaw)
        depth_error = float(pred['location'][2] - gt['location'][2])
        location_xy_error = pred['location'][:2] - gt['location'][:2]
        dim_error = np.abs(pred['dimensions'] - gt['dimensions'])
        key = (image_id, gt['class'], tuple(round(float(v), 2) for v in gt['bbox']))
        stereo = sgbm_by_target.get(key)
        record.update({
            'score': pred['score'], 'bbox_iou': iou,
            'pred_depth_m': float(pred['location'][2]),
            'depth_signed_error_m': depth_error,
            'depth_abs_error_m': abs(depth_error),
            'depth_relative_error': abs(depth_error) / max(gt['location'][2], 1e-6),
            'location_x_abs_error_m': abs(float(location_xy_error[0])),
            'location_y_abs_error_m': abs(float(location_xy_error[1])),
            'location_xy_error_m': float(np.linalg.norm(location_xy_error)),
            'sgbm_depth_m': stereo.get('stereo_depth_m') if stereo else None,
            'sgbm_abs_error_m': stereo.get('absolute_error_m') if stereo else None,
            'effective_depth_correction_m': (
                float(pred['location'][2]) - stereo['stereo_depth_m'])
                if stereo and stereo.get('stereo_depth_m') is not None else None,
            'dimension_mae_m': float(np.mean(dim_error)),
            'dimension_relative_error': float(np.mean(
                dim_error / np.maximum(gt['dimensions'], 1e-6))),
            'yaw_error_deg': math.degrees(angle_error(
                pred['rotation_y'], gt['rotation_y'])),
            'bev_iou': bev_iou, 'iou_3d': iou_3d,
            'iou_3d_fix_depth': iou_fix_depth,
            'iou_3d_fix_center_xy': iou_fix_center_xy,
            'iou_3d_fix_dimensions': iou_fix_dimensions,
            'iou_3d_fix_yaw': iou_fix_yaw})
        diagnostic_key = (
            image_id, CLASS_IDS.get(pred['class'], -1), pred['class_rank'])
        add_depth_diagnostics(
            record, diagnostics.get(diagnostic_key), float(gt['location'][2]))
      records.append(record)

  moderate_records = [record for record in records if record['moderate']]
  report = {
      'settings': vars(args), 'scope': 'Car Moderate条件目标',
      'overall': summarize(moderate_records),
      'by_distance': {name: summarize([
          record for record in moderate_records if record['distance_bucket'] == name])
          for _, _, name in DISTANCE_BUCKETS},
      'by_occlusion': {str(level): summarize([
          record for record in records if record['occlusion'] == level])
          for level in range(4)},
  }
  report['by_stereo_quality'] = {
      name: summarize([
          record for record in moderate_records
          if quality_bucket(record.get('sgbm_quality'), record.get('z_sgbm_m')) == name])
      for name in ('无诊断数据', '无效SGBM', '低质量<0.5',
                   '中质量0.5-0.8', '高质量>=0.8')}
  fallback_reasons = set()
  for record in moderate_records:
    fallback_reasons.update(record.get('fallback_reason') or [])
  report['by_fallback_reason'] = {
      reason: summarize([
          record for record in moderate_records
          if reason in (record.get('fallback_reason') or [])])
      for reason in sorted(fallback_reasons)}
  matched = [record for record in moderate_records if record['matched']]
  report['representative_frames'] = {
      'largest_depth_errors': [r['image_id'] for r in sorted(
          matched, key=lambda x: x['depth_abs_error_m'], reverse=True)[:12]],
      'largest_yaw_errors': [r['image_id'] for r in sorted(
          matched, key=lambda x: x['yaw_error_deg'], reverse=True)[:12]],
      'largest_center_xy_errors': [r['image_id'] for r in sorted(
          matched, key=lambda x: x['location_xy_error_m'], reverse=True)[:12]],
      'lowest_3d_iou': [r['image_id'] for r in sorted(
          matched, key=lambda x: x['iou_3d'])[:12]],
  }

  os.makedirs(args.eval_dir, exist_ok=True)
  json_path = os.path.join(args.eval_dir, 'error_analysis.json')
  csv_path = os.path.join(args.eval_dir, 'error_analysis_records.csv')
  with open(json_path, 'w', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, ensure_ascii=False)
  fieldnames = sorted({key for record in records for key in record})
  with open(csv_path, 'w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
  print(json.dumps(report, indent=2, ensure_ascii=False))
  print('明细：{}'.format(csv_path))


if __name__ == '__main__':
  main()
