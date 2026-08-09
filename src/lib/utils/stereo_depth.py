"""SGBM视差的深度转换和质量估计工具。"""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np


def disparity_to_depth(disparity, p2, p3, max_depth):
  """使用KITTI逐帧标定把视差转换为左相机坐标系深度（米）。"""
  fx = float(p2[0, 0])
  baseline = abs(float(p2[0, 3] / fx - p3[0, 3] / p3[0, 0]))
  principal_offset = float(p2[0, 2] - p3[0, 2])
  effective = disparity.astype(np.float32) - principal_offset
  depth = np.full(disparity.shape, np.nan, dtype=np.float32)
  valid = np.isfinite(effective) & (effective > 0.5)
  depth[valid] = fx * baseline / effective[valid]
  depth[(depth <= 0) | (depth > max_depth)] = np.nan
  return depth


def local_valid_quality(disparity, window):
  """用局部有效视差比例生成0～1质量图。"""
  if window <= 0 or window % 2 == 0:
    raise ValueError('stereo_quality_window必须是正奇数')
  valid = (np.isfinite(disparity) & (disparity > 0.5)).astype(np.float32)
  return cv2.boxFilter(
      valid, ddepth=-1, ksize=(window, window), normalize=True,
      borderType=cv2.BORDER_REPLICATE)
