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
import shutil
import subprocess

import torch.utils.data as data


class KITTI(data.Dataset):
  num_classes = 3
  default_resolution = [384, 1280]
  mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(KITTI, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'kitti')
    self.img_dir = os.path.join(self.data_dir, 'images', 'trainval')
    if opt.trainval:
      split = 'trainval' if split == 'train' else 'test'
      self.img_dir = os.path.join(self.data_dir, 'images', split)
      self.annot_path = os.path.join(
        self.data_dir, 'annotations', 'kitti_{}.json').format(split)
    else:
      self.annot_path = os.path.join(self.data_dir, 
        'annotations', 'kitti_{}_{}.json').format(opt.kitti_split, split)
    self.max_objs = 50
    self.class_name = [
      '__background__', 'Pedestrian', 'Car', 'Cyclist']
    self.cat_ids = {1:0, 2:1, 3:2, 4:-3, 5:-3, 6:-2, 7:-99, 8:-99, 9:-1}
    
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt
    self.alpha_in_degree = False

    print('==> initializing kitti {}, {} data.'.format(opt.kitti_split, split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    pass

  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results')
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    for img_id in results.keys():
      out_path = os.path.join(results_dir, '{:06d}.txt'.format(img_id))
      f = open(out_path, 'w')
      for cls_ind in results[img_id]:
        for j in range(len(results[img_id][cls_ind])):
          class_name = self.class_name[cls_ind]
          f.write('{} 0.0 0'.format(class_name))
          for i in range(len(results[img_id][cls_ind][j])):
            f.write(' {:.2f}'.format(results[img_id][cls_ind][j][i]))
          f.write('\n')
      f.close()

  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    gt_dir = os.path.join(save_dir, 'ground_truth')
    eval_results_dir = os.path.join(save_dir, 'eval_results')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)
    source_dir = os.path.join(self.data_dir, 'training', 'label_2')
    # 官方评估器要求输入从000000开始连续编号，3dop划分的原始帧号并不连续。
    for eval_id, img_id in enumerate(sorted(results, key=int)):
      source_name = '{:06d}.txt'.format(int(img_id))
      eval_name = '{:06d}.txt'.format(eval_id)
      shutil.copy2(os.path.join(source_dir, source_name),
                   os.path.join(gt_dir, eval_name))
      shutil.copy2(os.path.join(save_dir, 'results', source_name),
                   os.path.join(eval_results_dir, eval_name))

    eval_binary = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'tools', 'kitti_eval',
        'evaluate_object_3d_offline'))
    # KITTI 官方评估程序内部直接拼接目录和文件名，因此预测目录末尾必须有分隔符。
    results_dir = eval_results_dir + os.sep
    completed = subprocess.run(
        [eval_binary, gt_dir, results_dir],
        check=True, capture_output=True, text=True)
    eval_log = os.path.join(save_dir, 'kitti_eval.log')
    with open(eval_log, 'w', encoding='utf-8') as stream:
      stream.write(completed.stdout)
      stream.write(completed.stderr)
    self._save_ap_r40(save_dir)

  @staticmethod
  def _save_ap_r40(save_dir):
    """从41点PR曲线计算KITTI AP_R40，跳过recall=0的第一个点。"""
    difficulties = ['easy', 'moderate', 'hard']
    suffixes = {
        '2d': 'detection',
        'aos': 'orientation',
        'bev': 'detection_ground',
        '3d': 'detection_3d',
    }
    metrics = {}
    for class_name in ('car', 'pedestrian', 'cyclist'):
      metrics[class_name] = {}
      for metric_name, suffix in suffixes.items():
        path = os.path.join(
            save_dir, 'stats_{}_{}.txt'.format(class_name, suffix))
        if not os.path.exists(path):
          continue
        rows = []
        with open(path, 'r') as stream:
          for line in stream:
            values = [float(value) for value in line.split()]
            rows.append(100.0 * sum(values[1:41]) / 40.0)
        metrics[class_name][metric_name] = {
            name: value for name, value in zip(difficulties, rows)}
    with open(os.path.join(save_dir, 'kitti_ap_r40.json'),
              'w', encoding='utf-8') as stream:
      json.dump(metrics, stream, indent=2, ensure_ascii=False)
    
