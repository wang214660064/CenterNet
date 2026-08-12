#!/usr/bin/env python
"""可视化双目在线增强前后的左右图和SGBM视差。"""

import argparse
import os
import sys

import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'lib'))

from utils.stereo_augmentation import augment_stereo_pair  # noqa: E402


def parse_args():
  parser = argparse.ArgumentParser(description='检查双目在线增强效果')
  parser.add_argument('--image-id', default='000008')
  parser.add_argument('--seed', type=int, default=317)
  parser.add_argument('--output', default='exp/stereo_augmentation/preview.jpg')
  return parser.parse_args()


def disparity_color(left, right):
  matcher = cv2.StereoSGBM_create(
      minDisparity=0, numDisparities=128, blockSize=5,
      P1=8 * 5 * 5, P2=32 * 5 * 5, disp12MaxDiff=1,
      uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
      preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
  disparity = matcher.compute(
      cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
      cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
  valid = disparity > 0
  normalized = np.zeros_like(disparity, dtype=np.uint8)
  if np.any(valid):
    upper = max(float(np.percentile(disparity[valid], 99)), 1.0)
    normalized[valid] = np.clip(disparity[valid] / upper * 255, 0, 255)
  return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def label(image, text):
  result = image.copy()
  cv2.rectangle(result, (0, 0), (result.shape[1], 38), (20, 20, 20), -1)
  cv2.putText(result, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
              0.72, (255, 255, 255), 2, cv2.LINE_AA)
  return result


def main():
  args = parse_args()
  image_name = '{}.png'.format(args.image_id)
  left_path = os.path.join(PROJECT_ROOT, 'data', 'kitti', 'training',
                           'image_2', image_name)
  right_path = os.path.join(PROJECT_ROOT, 'data', 'kitti', 'training',
                            'image_3', image_name)
  left, right = cv2.imread(left_path), cv2.imread(right_path)
  if left is None or right is None:
    raise RuntimeError('无法读取双目图像: {} / {}'.format(left_path, right_path))

  left_aug, right_aug, metadata = augment_stereo_pair(
      left, right, rng=np.random.RandomState(args.seed),
      shared_prob=1.0, mismatch_prob=1.0, strength=0.15)
  original_disp = disparity_color(left, right)
  augmented_disp = disparity_color(left_aug, right_aug)
  # OpenCV内置字体不支持中文，图上使用简短英文避免乱码。
  top = np.hstack([label(left, 'Original Left'), label(right, 'Original Right'),
                   label(original_disp, 'Original SGBM Disparity')])
  bottom = np.hstack([
      label(left_aug, 'Augmented Left'), label(right_aug, 'Augmented Right'),
      label(augmented_disp, 'Augmented SGBM Disparity')])
  output = np.vstack([top, bottom])
  output_path = os.path.join(PROJECT_ROOT, args.output)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  cv2.imwrite(output_path, output)
  print('增强信息: {}'.format(metadata))
  print('输出: {}'.format(output_path))


if __name__ == '__main__':
  main()
