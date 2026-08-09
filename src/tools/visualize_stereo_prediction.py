#!/usr/bin/env python
"""用训练好的Stereo DDD模型可视化一帧KITTI预测。"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_DIR = os.path.join(SRC_DIR, 'lib')
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)
if LIB_DIR not in sys.path:
  sys.path.insert(0, LIB_DIR)

from opts import opts  # noqa: E402
from datasets.dataset_factory import get_dataset  # noqa: E402
from models.model import create_model, load_model  # noqa: E402
from trains.train_factory import train_factory  # noqa: E402


def parse_args():
  parser = argparse.ArgumentParser(description='可视化单帧Stereo DDD预测')
  parser.add_argument('--image-id', default='000008', help='KITTI六位帧号')
  parser.add_argument('--load-model', required=True, help='训练权重路径')
  parser.add_argument('--exp-id', default='stereo_single_visual')
  parser.add_argument('--center-thresh', type=float, default=0.25)
  return parser.parse_args()


def find_image_index(images, image_id):
  target = int(image_id)
  for index, current_id in enumerate(images):
    if int(current_id) == target:
      return index
  raise ValueError('帧 {} 不在KITTI 3dop验证集中'.format(image_id))


def main():
  args = parse_args()
  option = opts().init([
      'stereo_ddd', '--dataset', 'kitti', '--kitti_split', '3dop',
      '--arch', 'stereo_dla_34', '--exp_id', args.exp_id,
      '--load_model', args.load_model,
      '--debug', '4', '--num_iters', '1',
      '--center_thresh', str(args.center_thresh)])

  Dataset = get_dataset(option.dataset, option.task)
  option = opts().update_dataset_info_and_set_heads(option, Dataset)
  os.environ['CUDA_VISIBLE_DEVICES'] = option.gpus_str
  use_cuda = option.gpus[0] >= 0 and torch.cuda.is_available()
  if not use_cuda:
    option.gpus = [-1]
  option.device = torch.device('cuda' if use_cuda else 'cpu')
  print('运行设备：{}'.format(
      torch.cuda.get_device_name(0) if use_cuda else 'CPU'))

  dataset = Dataset(option, 'val')
  index = find_image_index(dataset.images, args.image_id)
  loader = DataLoader(Subset(dataset, [index]), batch_size=1, shuffle=False,
                      num_workers=0, pin_memory=option.device.type == 'cuda')

  model = create_model(option.arch, option.heads, option.head_conv)
  model = load_model(model, option.load_model)
  optimizer = torch.optim.Adam(model.parameters(), option.lr)
  Trainer = train_factory[option.task]
  trainer = Trainer(option, model, optimizer)
  trainer.set_device(option.gpus, option.chunk_sizes, option.device)

  os.makedirs(option.debug_dir, exist_ok=True)
  with torch.no_grad():
    trainer.val(0, loader)

  print('可视化完成，输出目录：{}'.format(option.debug_dir))
  print('重点查看：0add_pred.png、0bird_pred_gt.png、0out.png')


if __name__ == '__main__':
  main()
