#!/usr/bin/env bash
python train.py --dataroot ./database/edges2shoes-r \
  --model mobilepix2pix \
  --mask \
  --mask_weight_decay 0.006 \
  --lambda_update_coeff 2.0 \
  --upconv_bound \
  --upconv_coeff 100.0 \
  --gpu_ids 0 \
  --ngf 48 \
  --ndf 128 \
  --gan_mode hinge \
  --load_size 256 \
  --no_flip \
  --name mobile_mask_edges2shoes \
  --AtoB_macs_threshold 4.5