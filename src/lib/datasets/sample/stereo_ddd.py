"""KITTI双目3D样本：在原DDD监督上增加SGBM深度残差标签。"""

from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np

from .ddd import DddDataset
from lib.utils.stereo_augmentation import augment_stereo_pair
from lib.utils.stereo_depth import disparity_to_depth, local_valid_quality


class StereoDddDataset(DddDataset):
  def _read_image(self, img_path):
    """复用已增强的左图，避免网络输入和SGBM看到不同图像。"""
    cached_path = getattr(self, '_stereo_left_path', None)
    # DddDataset从images/trainval读图，SGBM从training/image_2读图；
    # 两个路径不同，但同一帧的文件名相同。
    if cached_path is not None and \
        os.path.basename(cached_path) == os.path.basename(img_path):
      return self._stereo_left_image
    return cv2.imread(img_path)

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
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    q1, q3 = np.percentile(values, [25, 75])
    return (median, float(values.size) / max(1, roi.size),
            mad, float(q3 - q1))

  def __getitem__(self, index):
    image_id = self.images[index]
    image_info = self.coco.loadImgs(ids=[image_id])[0]
    file_name = image_info['file_name']
    left_path = os.path.join(self.data_dir, 'training', 'image_2', file_name)
    right_path = os.path.join(self.data_dir, 'training', 'image_3', file_name)
    left, right = cv2.imread(left_path), cv2.imread(right_path)
    if left is None or right is None:
      raise RuntimeError('无法读取双目图像: {} / {}'.format(left_path, right_path))
    if left.shape != right.shape:
      raise RuntimeError('左右目图像尺寸不一致: {} / {}'.format(
          left.shape, right.shape))

    if self.split == 'train' and self.opt.stereo_photo_aug:
      left, right, _ = augment_stereo_pair(
          left, right,
          shared_prob=self.opt.stereo_photo_aug_prob,
          mismatch_prob=self.opt.stereo_camera_mismatch_prob,
          strength=self.opt.stereo_photo_aug_strength)

    # DddDataset通过_read_image复用这张左图，标注、左图和
    # 后续SGBM深度图再经过同一个几何仿射变换。
    self._stereo_left_path = left_path
    self._stereo_left_image = left
    try:
      ret = super(StereoDddDataset, self).__getitem__(index)
    finally:
      del self._stereo_left_path
      del self._stereo_left_image
    transform = ret.pop('stereo_trans_output')

    p2 = np.asarray(image_info['calib'], dtype=np.float32)
    # v6a训练专用：把输出特征坐标还原到原始图像，再通过P2反投影。
    ret['proj_center_inverse_affine'] = cv2.invertAffineTransform(
        transform).astype(np.float32)
    ret['proj_center_calib'] = p2
    p3 = self._read_p3(image_id)
    disparity = self._sgbm(left, right)
    quality_map = local_valid_quality(
        disparity, self.opt.stereo_quality_window)
    depth = disparity_to_depth(disparity, p2, p3, self.opt.stereo_max_depth)

    safe_depth = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
    output_size = (self.opt.output_w, self.opt.output_h)
    ret['sgbm_depth'] = cv2.warpAffine(
        safe_depth, transform, output_size, flags=cv2.INTER_LINEAR)[None]
    ret['sgbm_quality'] = cv2.warpAffine(
        quality_map, transform, output_size, flags=cv2.INTER_LINEAR)[None]

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
      sgbm_depth, _, _, _ = measured
      raw_offset = float(ann['depth']) - sgbm_depth
      # 监督目标与推理阶段使用相同限幅，避免学习推理时不会采用的大修正。
      offset_limit = max(
          self.opt.depth_offset_min_limit,
          min(self.opt.depth_offset_max_abs,
              sgbm_depth * self.opt.depth_offset_max_ratio))
      offset_target[k, 0] = np.clip(raw_offset, -offset_limit, offset_limit)
      offset_mask[k] = 1

    ret['depth_offset'] = offset_target
    ret['depth_offset_mask'] = offset_mask
    return ret
