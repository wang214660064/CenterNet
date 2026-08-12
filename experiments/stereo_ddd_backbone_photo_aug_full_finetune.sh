#!/usr/bin/env bash
set -euo pipefail

# 从v5最佳权重出发，使用全新优化器极小学习率微调全模型。
# v5检查点轮次为26，因此本实验记录为27～41，最多新增15轮。
python src/main.py stereo_ddd \
  --dataset kitti \
  --kitti_split project2000 \
  --arch stereo_dla_34 \
  --exp_id stereo_project2000_backbone_photo_aug_full_finetune \
  --load_model exp/stereo_ddd/stereo_project2000_projected_center_v5_continue_e11/model_best.pth \
  --backbone_photo_aug \
  --backbone_photo_aug_prob 0.5 \
  --backbone_photo_aug_strength 0.15 \
  --batch_size 4 \
  --num_workers 8 \
  --val_num_workers 8 \
  --lr 1e-6 \
  --num_epochs 41 \
  --lr_step 36,39 \
  --val_intervals 1 \
  --metric loss \
  --early_stopping_patience 5 \
  --early_stopping_min_delta 0.002 \
  --depth_offset_weight 1.0 \
  --depth_offset_loss huber \
  --depth_offset_huber_delta 1.0 \
  --depth_uncertainty_calibration_weight 0.05 \
  --depth_gate_weight 0.2 \
  --depth_fusion_weight 0.5 \
  --proj_center_weight 1.0 \
  --depth_gate_focal_gamma 2.0 \
  --depth_gate_focal_alpha 0.5 \
  --depth_gate_ambiguity_margin 0.2 \
  --depth_gate_max_regret 4.0 \
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
