#!/usr/bin/env bash
set -euo pipefail

python src/main.py stereo_ddd \
  --dataset kitti \
  --kitti_split 3dop \
  --arch stereo_dla_34 \
  --exp_id stereo_sgbm_distance_gate \
  --load_model exp/stereo_ddd/stereo_sgbm_offset/model_best.pth \
  --batch_size 4 \
  --num_workers 8 \
  --val_num_workers 8 \
  --lr 5e-5 \
  --num_epochs 70 \
  --lr_step 45,60 \
  --val_intervals 5 \
  --early_stopping_patience 2 \
  --early_stopping_min_delta 0.01 \
  --depth_offset_weight 1.0 \
  --stereo_num_disparities 128 \
  --stereo_block_size 5 \
  --stereo_max_depth 80 \
  --stereo_quality_window 31 \
  --stereo_far_distance 30 \
  --stereo_far_min_quality 0.8 \
  --depth_offset_far_max_uncertainty 3 \
  --depth_offset_max_abs 8 \
  --depth_offset_max_ratio 0.15 \
  --depth_offset_min_limit 2
