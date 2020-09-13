#!/usr/bin/env bash
python train.py --dataroot ./database/horse2zebra \
  --model cyclegan \
  --mask \
  --mask_weight_decay 0.001 \
  --lambda_update_coeff 1.0 \
  --upconv_bound \
  --upconv_coeff 100.0 \
  --gpu_ids 0 \
  --name mask_horse2zebra \
  --AtoB_macs_threshold 4.0  \
  --BtoA_macs_threshold 4.0 \
