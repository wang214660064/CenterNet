#!/usr/bin/env python
"""可视化原始左图与只进入Backbone的增强图像。"""

import argparse
import os
import sys

import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'lib'))

from utils.backbone_augmentation import augment_backbone_image  # noqa: E402


def parse_args():
  parser = argparse.ArgumentParser(description='检查Backbone左图外观增强')
  parser.add_argument('--image-id', default='000008')
  parser.add_argument('--seed', type=int, default=317)
  parser.add_argument(
      '--output', default='exp/backbone_augmentation/preview.jpg')
  return parser.parse_args()


def label(image, text):
  result = image.copy()
  cv2.rectangle(result, (0, 0), (result.shape[1], 38), (20, 20, 20), -1)
  cv2.putText(result, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
              0.72, (255, 255, 255), 2, cv2.LINE_AA)
  return result


def main():
  args = parse_args()
  image_path = os.path.join(
      PROJECT_ROOT, 'data', 'kitti', 'training', 'image_2',
      '{}.png'.format(args.image_id))
  original = cv2.imread(image_path)
  if original is None:
    raise RuntimeError('无法读取左目图像: {}'.format(image_path))

  augmented, _ = augment_backbone_image(
      original, rng=np.random.RandomState(args.seed),
      probability=1.0, strength=0.15)
  difference = cv2.absdiff(original, augmented)
  difference = cv2.convertScaleAbs(difference, alpha=4.0)
  output = np.hstack((
      label(original, 'Original Left for SGBM'),
      label(augmented, 'Augmented Left for Backbone'),
      label(difference, 'Appearance Difference x4')))

  output_path = os.path.join(PROJECT_ROOT, args.output)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  cv2.imwrite(output_path, output)
  mean_difference = float(np.abs(
      original.astype(np.float32) - augmented.astype(np.float32)).mean())
  print('平均像素差: {:.3f}'.format(mean_difference))
  print('SGBM仍使用未增强的原始左右图。')
  print('输出: {}'.format(output_path))


if __name__ == '__main__':
  main()
