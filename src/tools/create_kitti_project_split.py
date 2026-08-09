#!/usr/bin/env python
"""生成固定、无交叉的project2000正式训练与验证数据集。"""

import argparse
import json
import os
import random


def parse_args():
  parser = argparse.ArgumentParser(description='生成KITTI快速训练划分')
  parser.add_argument('--annotations-dir', default='data/kitti/annotations')
  parser.add_argument('--kitti-dir', default='data/kitti')
  parser.add_argument('--output-name', default='project2000')
  parser.add_argument('--train-count', type=int, default=1600)
  parser.add_argument('--val-count', type=int, default=400)
  parser.add_argument('--seed', type=int, default=317)
  return parser.parse_args()


def select_subset(dataset, count, rng):
  image_ids = [int(image['id']) for image in dataset['images']]
  if count > len(image_ids):
    raise ValueError('抽样数{}超过数据量{}'.format(count, len(image_ids)))
  selected_ids = set(rng.sample(image_ids, count))
  images = [image for image in dataset['images']
            if int(image['id']) in selected_ids]
  annotations = [ann for ann in dataset['annotations']
                 if int(ann['image_id']) in selected_ids]
  for ann_id, annotation in enumerate(annotations, start=1):
    annotation['id'] = ann_id
  return {
      'images': sorted(images, key=lambda item: int(item['id'])),
      'annotations': annotations,
      'categories': dataset['categories'],
  }


def main():
  args = parse_args()
  paths = {
      'train': os.path.join(args.annotations_dir, 'kitti_3dop_train.json'),
      'val': os.path.join(args.annotations_dir, 'kitti_3dop_val.json'),
  }
  rng = random.Random(args.seed)
  outputs = {}
  split_dir = os.path.join(args.kitti_dir, 'ImageSets_{}'.format(args.output_name))
  os.makedirs(split_dir, exist_ok=True)
  for split, count in (('train', args.train_count), ('val', args.val_count)):
    with open(paths[split], 'r', encoding='utf-8') as stream:
      source = json.load(stream)
    outputs[split] = select_subset(source, count, rng)
    output_path = os.path.join(
        args.annotations_dir,
        'kitti_{}_{}.json'.format(args.output_name, split))
    with open(output_path, 'w', encoding='utf-8') as stream:
      json.dump(outputs[split], stream, ensure_ascii=False)
    list_path = os.path.join(split_dir, '{}.txt'.format(split))
    with open(list_path, 'w', encoding='utf-8') as stream:
      for image in outputs[split]['images']:
        stream.write('{:06d}\n'.format(int(image['id'])))
    print('{}: {}帧，{}个标注 -> {}'.format(
        split, len(outputs[split]['images']),
        len(outputs[split]['annotations']), output_path))

  train_ids = {int(item['id']) for item in outputs['train']['images']}
  val_ids = {int(item['id']) for item in outputs['val']['images']}
  if train_ids & val_ids:
    raise RuntimeError('训练集和验证集存在重复帧')
  summary = {
      'name': args.output_name,
      'seed': args.seed,
      'train_frames': len(train_ids),
      'val_frames': len(val_ids),
      'overlap_frames': 0,
      'image_storage': 'training/image_2与training/image_3（不重复复制）',
  }
  with open(os.path.join(split_dir, 'summary.json'), 'w', encoding='utf-8') as stream:
    json.dump(summary, stream, indent=2, ensure_ascii=False)
  with open(os.path.join(split_dir, 'README_CN.md'), 'w', encoding='utf-8') as stream:
    stream.write(
        '# 本项目KITTI project2000正式数据集\n\n'
        '- `train.txt`：正式训练集1600帧。\n'
        '- `val.txt`：正式验证集400帧。\n'
        '- 左右图仍位于`training/image_2`和`training/image_3`，不重复复制。\n'
        '- COCO训练标注位于`annotations/kitti_quick2000_train.json`和'
        '`kitti_quick2000_val.json`。\n'
        '- 本项目后续训练、验证、AP和分桶分析均以该划分为准。\n')
  print('可观察子集目录：{}'.format(split_dir))


if __name__ == '__main__':
  main()
