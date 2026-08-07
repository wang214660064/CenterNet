#!/usr/bin/env bash
set -euo pipefail

python src/main.py stereo_ddd \
  --dataset kitti \
  --kitti_split 3dop \
  --arch stereo_dla_34 \
  --exp_id stereo_sgbm_offset \
  --batch_size 4 \
  --num_workers 2 \
  --lr 1.25e-4 \
  --num_epochs 70 \
  --lr_step 45,60 \
  --val_intervals 5 \
  --depth_offset_weight 1.0 \
  --stereo_num_disparities 128 \
  --stereo_block_size 5 \
  --stereo_max_depth 80 \
  --gpus -1
