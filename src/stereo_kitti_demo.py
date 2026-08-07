#!/usr/bin/env python3
"""KITTI 双目测距第一阶段：CenterNet 2D 检测 + SGBM 深度。"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys

import numpy as np

try:
  import cv2
except ImportError:
  cv2 = None


COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
DEFAULT_CLASSES = {1, 2, 3, 4, 6, 8, 10, 13}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def require_opencv():
  if cv2 is None:
    raise RuntimeError('缺少 OpenCV，请先安装 requirements.txt 中的 opencv-python。')


def parse_args():
  parser = argparse.ArgumentParser(
      description='使用 CenterNet 检测 KITTI 左图，并通过双目视差估计目标距离。')
  parser.add_argument('--data-dir', default=os.path.join(PROJECT_ROOT, 'data', 'kitti'))
  parser.add_argument('--split', choices=['training', 'testing'], default='training')
  parser.add_argument('--image-id', default='000008', help='不带扩展名的 KITTI 帧编号')
  parser.add_argument('--model', default=os.path.join(
      PROJECT_ROOT, 'models', 'ctdet_coco_dla_2x.pth'))
  parser.add_argument('--output-dir', default=os.path.join(
      PROJECT_ROOT, 'exp', 'stereo_stage1'))
  parser.add_argument('--score-thresh', type=float, default=0.35)
  parser.add_argument('--max-depth', type=float, default=80.0)
  parser.add_argument('--warning-depth', type=float, default=8.0)
  parser.add_argument('--num-disparities', type=int, default=128,
                      help='最大搜索视差，必须是 16 的倍数')
  parser.add_argument('--block-size', type=int, default=5,
                      help='SGBM 匹配窗口，必须为正奇数')
  parser.add_argument('--gpus', default='-1', help='例如 0；-1 表示 CPU')
  return parser.parse_args()


def read_projection_matrices(calib_path):
  values = {}
  with open(calib_path, 'r') as stream:
    for line in stream:
      if ':' not in line:
        continue
      key, raw = line.split(':', 1)
      values[key] = np.asarray([float(item) for item in raw.split()], dtype=np.float64)
  if 'P2' not in values or 'P3' not in values:
    raise ValueError('标定文件缺少 P2/P3: {}'.format(calib_path))
  return values['P2'].reshape(3, 4), values['P3'].reshape(3, 4)


def stereo_parameters(p2, p3):
  fx = float(p2[0, 0])
  if fx <= 0 or not np.isclose(fx, p3[0, 0], rtol=1e-3):
    raise ValueError('P2/P3 的水平焦距无效或不一致')
  baseline = abs(float(p2[0, 3] / fx - p3[0, 3] / p3[0, 0]))
  disparity_offset = float(p2[0, 2] - p3[0, 2])
  if baseline <= 0:
    raise ValueError('由 P2/P3 算出的双目基线无效')
  return {
      'fx': fx, 'fy': float(p2[1, 1]),
      'cx': float(p2[0, 2]), 'cy': float(p2[1, 2]),
      'tx': float(p2[0, 3]), 'ty': float(p2[1, 3]),
      'baseline': baseline, 'disparity_offset': disparity_offset,
  }


def compute_disparity(left, right, num_disparities, block_size):
  require_opencv()
  if left.shape[:2] != right.shape[:2]:
    raise ValueError('左右图尺寸不一致: {} != {}'.format(left.shape[:2], right.shape[:2]))
  if num_disparities <= 0 or num_disparities % 16:
    raise ValueError('--num-disparities 必须是正数且为 16 的倍数')
  if block_size <= 0 or block_size % 2 == 0:
    raise ValueError('--block-size 必须是正奇数')
  matcher = cv2.StereoSGBM_create(
      minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
      P1=8 * block_size * block_size, P2=32 * block_size * block_size,
      disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
      speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
  left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
  right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
  return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0


def disparity_to_depth(disparity, params, max_depth):
  effective = disparity - params['disparity_offset']
  depth = np.full(disparity.shape, np.nan, dtype=np.float32)
  valid = effective > 0.5
  depth[valid] = params['fx'] * params['baseline'] / effective[valid]
  depth[(depth <= 0) | (depth > max_depth)] = np.nan
  return depth


def measure_detection(bbox, disparity, depth, params):
  height, width = depth.shape
  x1, y1, x2, y2 = bbox
  x1, x2 = sorted((max(0, int(x1)), min(width, int(x2))))
  y1, y2 = sorted((max(0, int(y1)), min(height, int(y2))))
  if x2 - x1 < 4 or y2 - y1 < 4:
    return None

  # 避开背景和地面，采样目标框中下部的中心区域。
  roi_x1 = x1 + int((x2 - x1) * 0.2)
  roi_x2 = x2 - int((x2 - x1) * 0.2)
  roi_y1 = y1 + int((y2 - y1) * 0.35)
  roi_y2 = y1 + int((y2 - y1) * 0.85)
  roi_depth = depth[roi_y1:roi_y2, roi_x1:roi_x2]
  valid_depth = roi_depth[np.isfinite(roi_depth)]
  sample_count = int(valid_depth.size)
  roi_count = max(1, int(roi_depth.size))
  if sample_count < max(12, int(roi_count * 0.05)):
    return None

  median_depth = float(np.median(valid_depth))
  mad = float(np.median(np.abs(valid_depth - median_depth)))
  if mad > 0:
    valid_depth = valid_depth[np.abs(valid_depth - median_depth) <= 3.0 * mad]
  if valid_depth.size == 0:
    return None
  z = float(np.median(valid_depth))
  depth_mad = float(np.median(np.abs(valid_depth - z)))
  depth_iqr = float(np.percentile(valid_depth, 75) - np.percentile(valid_depth, 25))
  u = (x1 + x2) / 2.0
  v = (y1 + y2) / 2.0
  x = ((u - params['cx']) * z - params.get('tx', 0.0)) / params['fx']
  y = ((v - params['cy']) * z - params.get('ty', 0.0)) / params['fy']
  effective_disparity = params['fx'] * params['baseline'] / z
  return {
      'distance_m': z, 'camera_xyz_m': [float(x), float(y), z],
      'median_disparity_px': float(effective_disparity + params['disparity_offset']),
      'valid_depth_ratio': float(sample_count / roi_count),
      'depth_mad_m': depth_mad,
      'depth_iqr_m': depth_iqr,
      'filtered_sample_count': int(valid_depth.size),
      'sample_roi': [roi_x1, roi_y1, roi_x2, roi_y2],
  }


def load_centernet(model_path, gpus):
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
  from opts import opts
  from detectors.ctdet import CtdetDetector
  option = opts().init(['ctdet', '--load_model', model_path, '--gpus', gpus])
  return CtdetDetector(option)


def collect_detections(raw_results, score_thresh):
  detections = []
  for class_id in DEFAULT_CLASSES:
    for item in raw_results.get(class_id, []):
      if float(item[4]) >= score_thresh:
        detections.append({
            'class_id': class_id, 'class_name': COCO_NAMES[class_id],
            'score': float(item[4]), 'bbox': [float(value) for value in item[:4]],
        })
  return sorted(detections, key=lambda item: item['score'], reverse=True)


def draw_results(image, detections, warning_depth):
  require_opencv()
  canvas = image.copy()
  for detection in detections:
    x1, y1, x2, y2 = [int(value) for value in detection['bbox']]
    measured = detection.get('stereo')
    warning = measured is not None and measured['distance_m'] <= warning_depth
    color = (0, 0, 255) if warning else (40, 210, 40)
    if measured:
      label = '{} {:.2f}  {:.1f}m'.format(
          detection['class_name'], detection['score'], measured['distance_m'])
    else:
      label = '{} {:.2f}  depth:N/A'.format(detection['class_name'], detection['score'])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)
  return canvas


def colorize_disparity(disparity):
  require_opencv()
  valid = disparity > 0
  normalized = np.zeros(disparity.shape, dtype=np.uint8)
  if np.any(valid):
    upper = max(1.0, float(np.percentile(disparity[valid], 99)))
    normalized[valid] = np.clip(disparity[valid] * 255.0 / upper, 0, 255).astype(np.uint8)
  return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def require_file(path, hint):
  if not os.path.isfile(path):
    raise FileNotFoundError('{}\n{}'.format(path, hint))


def main():
  args = parse_args()
  require_opencv()
  base = os.path.abspath(args.data_dir)
  image_id = str(args.image_id).zfill(6)
  left_path = os.path.join(base, args.split, 'image_2', image_id + '.png')
  right_path = os.path.join(base, args.split, 'image_3', image_id + '.png')
  calib_path = os.path.join(base, args.split, 'calib', image_id + '.txt')
  require_file(left_path, '请检查 --data-dir、--split 和 --image-id。')
  require_file(right_path, '请将 KITTI 右彩色图解压到 image_3 目录。')
  require_file(calib_path, '双目测距需要该帧的 P2/P3 标定矩阵。')
  require_file(os.path.abspath(args.model), '请下载 CenterNet COCO 2D 检测模型。')

  left, right = cv2.imread(left_path), cv2.imread(right_path)
  if left is None or right is None:
    raise RuntimeError('OpenCV 无法读取左右图像')
  p2, p3 = read_projection_matrices(calib_path)
  params = stereo_parameters(p2, p3)
  disparity = compute_disparity(left, right, args.num_disparities, args.block_size)
  depth = disparity_to_depth(disparity, params, args.max_depth)

  detector = load_centernet(os.path.abspath(args.model), args.gpus)
  raw_results = detector.run(left)['results']
  detections = collect_detections(raw_results, args.score_thresh)
  for detection in detections:
    detection['stereo'] = measure_detection(
        detection['bbox'], disparity, depth, params)

  output_dir = os.path.abspath(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)
  stem = '{}_{}'.format(args.split, image_id)
  overlay_path = os.path.join(output_dir, stem + '_detections.jpg')
  disparity_path = os.path.join(output_dir, stem + '_disparity.jpg')
  json_path = os.path.join(output_dir, stem + '_results.json')
  cv2.imwrite(overlay_path, draw_results(left, detections, args.warning_depth))
  cv2.imwrite(disparity_path, colorize_disparity(disparity))
  payload = {
      'image_id': image_id, 'split': args.split,
      'left_image': left_path, 'right_image': right_path,
      'calibration': params, 'detections': detections,
      'notes': 'distance_m 是检测框中心区域的双目深度中位数，不是安全认证测距值。',
  }
  with open(json_path, 'w') as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
  print('完成：检测 {} 个目标'.format(len(detections)))
  print('叠加图：{}'.format(overlay_path))
  print('视差图：{}'.format(disparity_path))
  print('结构化结果：{}'.format(json_path))


if __name__ == '__main__':
  main()
