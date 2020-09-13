#!/usr/bin/env bash
python train.py --dataroot ./database/horse2zebra \
  --model mobilecyclegan \
  --mask \
  --mask_weight_decay 0.002 \
  --lambda_update_coeff 5.0 \
  --upconv_bound \
  --upconv_coeff 200.0 \
  --gpu_ids 0 \
  --ngf 32 \
  --name mobile_mask_horse2zebra \
  --AtoB_macs_threshold 2.5 \
  --BtoA_macs_threshold 4.0