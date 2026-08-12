from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.ddd_utils import project_3d_center_to_image
from utils.backbone_augmentation import augment_backbone_image
import pycocotools.coco as coco

class DddDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _convert_alpha(self, alpha):
    return math.radians(alpha + 45) if self.alpha_in_degree else alpha

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    if img is None:
      raise RuntimeError('无法读取图像: {}'.format(img_path))
    if self.split == 'train' and self.opt.backbone_photo_aug:
      # 这个图像副本只进入Backbone。StereoDddDataset会另行读取
      # 原始左右图计算SGBM，因此视差和质量监督不受影响。
      img, _ = augment_backbone_image(
          img, probability=self.opt.backbone_photo_aug_prob,
          strength=self.opt.backbone_photo_aug_strength)
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = self.calib

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
    if self.opt.keep_res:
      s = np.array([self.opt.input_w, self.opt.input_h], dtype=np.int32)
    else:
      s = np.array([width, height], dtype=np.int32)
    
    aug = False
    if self.split == 'train' and np.random.random() < self.opt.aug_ddd:
      aug = True
      sf = self.opt.scale
      cf = self.opt.shift
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)

    trans_input = get_affine_transform(
      c, s, 0, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    # if self.split == 'train' and not self.opt.no_color_aug:
    #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    num_classes = self.opt.num_classes
    trans_output = get_affine_transform(
      c, s, 0, [self.opt.output_w, self.opt.output_h])

    hm = np.zeros(
      (num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    dep = np.zeros((self.max_objs, 1), dtype=np.float32)
    rotbin = np.zeros((self.max_objs, 2), dtype=np.int64)
    rotres = np.zeros((self.max_objs, 2), dtype=np.float32)
    dim = np.zeros((self.max_objs, 3), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    rot_mask = np.zeros((self.max_objs), dtype=np.uint8)
    proj_center_offset = np.zeros((self.max_objs, 2), dtype=np.float32)
    proj_center_mask = np.zeros((self.max_objs), dtype=np.uint8)
    proj_center_base = np.zeros((self.max_objs, 2), dtype=np.float32)
    proj_center_camera_xy = np.zeros((self.max_objs, 2), dtype=np.float32)
    proj_center_extent_xy = np.zeros((self.max_objs, 2), dtype=np.float32)

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id <= -99:
        continue
      # if flipped:
      #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((h, w))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if cls_id < 0:
          ignore_id = [_ for _ in range(num_classes)] \
                      if cls_id == - 1 else  [- cls_id - 2]
          if self.opt.rect_mask:
            hm[ignore_id, int(bbox[1]): int(bbox[3]) + 1, 
              int(bbox[0]): int(bbox[2]) + 1] = 0.9999
          else:
            for cc in ignore_id:
              draw_gaussian(hm[cc], ct, radius)
            hm[ignore_id, ct_int[1], ct_int[0]] = 0.9999
          continue
        draw_gaussian(hm[cls_id], ct, radius)

        projected_center = ct.copy()
        proj_center_base[k] = ct
        if self.opt.task == 'stereo_ddd' and \
            'location' in ann and 'dim' in ann:
          try:
            # KITTI location是底面中心；这里保存3D框几何中心的相机坐标。
            proj_center_camera_xy[k] = [
                ann['location'][0], ann['location'][1] - ann['dim'][0] / 2.0]
            # KITTI尺寸顺序为[h,w,l]。车辆绕相机Y轴旋转后，其相机X方向
            # 占用范围约为|cos(ry)|*w + |sin(ry)|*l。
            rotation_y = float(ann.get('rotation_y', 0.0))
            height, width_3d, length_3d = ann['dim']
            proj_center_extent_xy[k] = [
                abs(np.cos(rotation_y)) * width_3d +
                abs(np.sin(rotation_y)) * length_3d,
                height]
            projected_center_image = project_3d_center_to_image(
                ann['location'], ann['dim'], calib)
            projected_center = affine_transform(
                projected_center_image, trans_output)
            if np.all(np.isfinite(projected_center)):
              proj_center_offset[k] = projected_center - ct
              offset_norm = np.linalg.norm(proj_center_offset[k])
              # 极端边缘目标的几何中心可能远在画外，需要Edge Fusion单独处理。
              within_limit = offset_norm <= self.opt.proj_center_max_offset
              proj_center_mask[k] = 1 if not aug and within_limit else 0
            else:
              projected_center = ct.copy()
          except ValueError:
            projected_center = ct.copy()

        wh[k] = 1. * w, 1. * h
        gt_center = projected_center if self.opt.task == 'stereo_ddd' else ct
        gt_entry = [gt_center[0], gt_center[1], 1] + \
                   self._alpha_to_8(self._convert_alpha(ann['alpha'])) + \
                   [ann['depth']] + (np.array(ann['dim']) / 1).tolist()
        if self.opt.reg_bbox:
          gt_entry += [w, h]
        if self.opt.task == 'stereo_ddd':
          # 3D恢复使用投影中心，2D框继续使用原二维中心。
          gt_entry += [ct[0], ct[1]]
        gt_entry += [cls_id]
        gt_det.append(gt_entry)
        # if (not self.opt.car_only) or cls_id == 1: # Only estimate ADD for cars !!!
        if 1:
          alpha = self._convert_alpha(ann['alpha'])
          # print('img_id cls_id alpha rot_y', img_path, cls_id, alpha, ann['rotation_y'])
          if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            rotbin[k, 0] = 1
            rotres[k, 0] = alpha - (-0.5 * np.pi)    
          if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            rotbin[k, 1] = 1
            rotres[k, 1] = alpha - (0.5 * np.pi)
          dep[k] = ann['depth']
          dim[k] = ann['dim']
          # print('        cat dim', cls_id, dim[k])
          ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
          reg[k] = ct - ct_int
          reg_mask[k] = 1 if not aug else 0
          rot_mask[k] = 1
    # print('gt_det', gt_det)
    # print('')
    ret = {'input': inp, 'hm': hm, 'dep': dep, 'dim': dim, 'ind': ind, 
           'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
           'rot_mask': rot_mask}
    if self.opt.reg_bbox:
      ret.update({'wh': wh})
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.task == 'stereo_ddd':
      ret['stereo_trans_output'] = trans_output.astype(np.float32)
      ret['proj_center_offset'] = proj_center_offset
      ret['proj_center_mask'] = proj_center_mask
      ret['proj_center_base'] = proj_center_base
      ret['proj_center_camera_xy'] = proj_center_camera_xy
      ret['proj_center_extent_xy'] = proj_center_extent_xy
    if self.opt.debug > 0 or not ('train' in self.split):
      empty_det_size = 16 + (2 if self.opt.reg_bbox else 0) + \
                       (2 if self.opt.task == 'stereo_ddd' else 0)
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, empty_det_size), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': calib,
              'image_path': img_path, 'img_id': img_id}
      ret['meta'] = meta
    
    return ret

  def _alpha_to_8(self, alpha):
    # return [alpha, 0, 0, 0, 0, 0, 0, 0]
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
