"""KITTI双目3D样本：在原DDD监督上增加SGBM深度残差标签。"""

from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np

from .ddd import DddDataset


class StereoDddDataset(DddDataset):
  def _read_p3(self, image_id):
    calib_path = os.path.join(
        self.data_dir, 'training', 'calib', '{:06d}.txt'.format(image_id))
    with open(calib_path, 'r') as stream:
      for line in stream:
        if line.startswith('P3:'):
          return np.asarray(
              [float(value) for value in line.split()[1:]],
              dtype=np.float32).reshape(3, 4)
    raise ValueError('标定文件缺少P3: {}'.format(calib_path))

  def _sgbm(self, left, right):
    block_size = self.opt.stereo_block_size
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=self.opt.stereo_num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

  @staticmethod
  def _stereo_params(p2, p3):
    fx = float(p2[0, 0])
    baseline = abs(float(p2[0, 3] / fx - p3[0, 3] / p3[0, 0]))
    return fx, baseline, float(p2[0, 2] - p3[0, 2])

  def _measure_box_depth(self, depth, bbox):
    height, width = depth.shape
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, int(x1)), min(width, int(x2))))
    y1, y2 = sorted((max(0, int(y1)), min(height, int(y2))))
    if x2 - x1 < 4 or y2 - y1 < 4:
      return None
    roi = depth[
        y1 + int((y2 - y1) * 0.35):y1 + int((y2 - y1) * 0.85),
        x1 + int((x2 - x1) * 0.2):x2 - int((x2 - x1) * 0.2)]
    values = roi[np.isfinite(roi)]
    if values.size < max(12, int(roi.size * 0.05)):
      return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
      values = values[np.abs(values - median) <= 3.0 * mad]
    if values.size == 0:
      return None
    return float(np.median(values)), float(values.size) / max(1, roi.size)

  def __getitem__(self, index):
    ret = super(StereoDddDataset, self).__getitem__(index)
    transform = ret.pop('stereo_trans_output')
    image_id = self.images[index]
    image_info = self.coco.loadImgs(ids=[image_id])[0]
    file_name = image_info['file_name']
    left_path = os.path.join(self.data_dir, 'training', 'image_2', file_name)
    right_path = os.path.join(self.data_dir, 'training', 'image_3', file_name)
    left, right = cv2.imread(left_path), cv2.imread(right_path)
    if left is None or right is None:
      raise RuntimeError('无法读取双目图像: {} / {}'.format(left_path, right_path))

    p2 = np.asarray(image_info['calib'], dtype=np.float32)
    p3 = self._read_p3(image_id)
    fx, baseline, principal_offset = self._stereo_params(p2, p3)
    disparity = self._sgbm(left, right)
    effective = disparity - principal_offset
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = effective > 0.5
    depth[valid] = fx * baseline / effective[valid]
    depth[(depth <= 0) | (depth > self.opt.stereo_max_depth)] = np.nan

    safe_depth = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
    valid_map = np.isfinite(depth).astype(np.float32)
    output_size = (self.opt.output_w, self.opt.output_h)
    ret['sgbm_depth'] = cv2.warpAffine(
        safe_depth, transform, output_size, flags=cv2.INTER_LINEAR)[None]
    ret['sgbm_quality'] = cv2.warpAffine(
        valid_map, transform, output_size, flags=cv2.INTER_NEAREST)[None]

    offset_target = np.zeros((self.max_objs, 1), dtype=np.float32)
    offset_mask = np.zeros((self.max_objs), dtype=np.uint8)
    ann_ids = self.coco.getAnnIds(imgIds=[image_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    for k, ann in enumerate(anns[:self.max_objs]):
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id < 0 or ret['rot_mask'][k] == 0:
        continue
      measured = self._measure_box_depth(depth, self._coco_box_to_bbox(ann['bbox']))
      if measured is None:
        continue
      sgbm_depth, quality = measured
      offset_target[k, 0] = float(ann['depth']) - sgbm_depth
      offset_mask[k] = 1
      # 中心位置的质量值用于offset头的门控输入。
      center_index = int(ret['ind'][k])
      center_y, center_x = divmod(center_index, self.opt.output_w)
      ret['sgbm_quality'][0, center_y, center_x] = quality

    ret['depth_offset'] = offset_target
    ret['depth_offset_mask'] = offset_mask
    return ret
