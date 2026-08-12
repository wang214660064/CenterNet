#!/usr/bin/env bash
set -euo pipefail

# 单变量A/B：从v5最佳权重出发，只校准最终深度Gate头。
python src/main.py stereo_ddd \
  --dataset kitti \
  --kitti_split project2000 \
  --arch stereo_dla_34 \
  --exp_id stereo_project2000_gate_calibration_v8 \
  --load_model exp/stereo_ddd/stereo_project2000_projected_center_v5_continue_e11/model_best.pth \
  --train_gate_only \
  --batch_size 16 \
  --num_workers 8 \
  --val_num_workers 8 \
  --lr 5e-5 \
  --num_epochs 20 \
  --lr_step 12,16 \
  --val_intervals 1 \
  --metric depth_fusion_loss \
  --early_stopping_patience 5 \
  --early_stopping_min_delta 0.0005 \
  --depth_gate_weight 0.1 \
  --depth_fusion_weight 1.0 \
  --depth_gate_core_range_weight 1.5 \
  --depth_gate_mid_quality_weight 2.0 \
  --depth_gate_focal_gamma 2.0 \
  --depth_gate_focal_alpha 0.5 \
  --depth_gate_ambiguity_margin 0.2 \
  --depth_gate_max_regret 4.0 \
  --proj_center_max_offset 64 \
  --campus_near_distance 15 \
  --campus_core_distance 30 \
  --campus_warning_distance 50 \
  --campus_near_weight 2.0 \
  --campus_core_weight 1.5 \
  --campus_warning_weight 0.5 \
  --campus_beyond_weight 0.0 \
  --stereo_num_disparities 128 \
  --stereo_block_size 5 \
  --stereo_max_depth 80 \
  --stereo_min_quality 0.5 \
  --stereo_quality_window 31 \
  --stereo_far_distance 30 \
  --stereo_far_min_quality 0.8 \
  --depth_offset_max_uncertainty 10 \
  --depth_offset_far_max_uncertainty 3 \
  --depth_offset_max_abs 8 \
  --depth_offset_max_ratio 0.15 \
  --depth_offset_min_limit 2
