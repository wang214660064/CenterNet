#!/usr/bin/env python
"""分解KITTI Stereo DDD的二维、深度、尺寸和朝向误差。"""

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np


DISTANCE_BUCKETS = ((0, 15, '0-15m'), (15, 30, '15-30m'),
                    (30, 50, '30-50m'), (50, float('inf'), '50m以上'))


def parse_args():
  parser = argparse.ArgumentParser(description='分解Stereo DDD验证误差')
  parser.add_argument('--eval-dir', default='exp/stereo_ddd/stereo_sgbm_offset_eval')
  parser.add_argument('--sgbm-json', default='exp/stereo_stage2/sgbm_depth_metrics.json')
  parser.add_argument('--annotations', default='data/kitti/annotations/kitti_3dop_val.json')
  parser.add_argument('--score-thresh', type=float, default=0.25)
  parser.add_argument('--match-iou', type=float, default=0.5)
  return parser.parse_args()


def read_kitti(path, prediction=False):
  objects = []
  if not os.path.exists(path):
    return objects
  with open(path, 'r') as stream:
    for line in stream:
      values = line.split()
      if len(values) < 15:
        continue
      objects.append({
          'class': values[0], 'truncation': float(values[1]),
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
              'sgbm_abs_error_m', 'dimension_mae_m', 'dimension_relative_error',
              'yaw_error_deg', 'bev_iou', 'iou_3d', 'iou_3d_fix_depth',
              'iou_3d_fix_dimensions', 'iou_3d_fix_yaw'):
    summary[key + '_mean'] = mean(matched, key)
  for component in ('depth', 'dimensions', 'yaw'):
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
  return summary


def main():
  args = parse_args()
  with open(args.annotations, 'r') as stream:
    images = sorted(json.load(stream)['images'], key=lambda item: int(item['id']))
  with open(args.sgbm_json, 'r') as stream:
    sgbm_records = json.load(stream)['records']
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
        fixed_dimensions = dict(pred)
        fixed_dimensions['dimensions'] = gt['dimensions'].copy()
        fixed_yaw = dict(pred)
        fixed_yaw['rotation_y'] = gt['rotation_y']
        _, iou_fix_depth = spatial_iou(gt, fixed_depth)
        _, iou_fix_dimensions = spatial_iou(gt, fixed_dimensions)
        _, iou_fix_yaw = spatial_iou(gt, fixed_yaw)
        depth_error = float(pred['location'][2] - gt['location'][2])
        dim_error = np.abs(pred['dimensions'] - gt['dimensions'])
        key = (image_id, gt['class'], tuple(round(float(v), 2) for v in gt['bbox']))
        stereo = sgbm_by_target.get(key)
        record.update({
            'score': pred['score'], 'bbox_iou': iou,
            'pred_depth_m': float(pred['location'][2]),
            'depth_signed_error_m': depth_error,
            'depth_abs_error_m': abs(depth_error),
            'depth_relative_error': abs(depth_error) / max(gt['location'][2], 1e-6),
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
            'iou_3d_fix_dimensions': iou_fix_dimensions,
            'iou_3d_fix_yaw': iou_fix_yaw})
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
  matched = [record for record in moderate_records if record['matched']]
  report['representative_frames'] = {
      'largest_depth_errors': [r['image_id'] for r in sorted(
          matched, key=lambda x: x['depth_abs_error_m'], reverse=True)[:12]],
      'largest_yaw_errors': [r['image_id'] for r in sorted(
          matched, key=lambda x: x['yaw_error_deg'], reverse=True)[:12]],
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
