#!/usr/bin/env bash
python train.py --dataroot ./database/cityscapes \
  --model mobilepix2pix \
  --mask \
  --mask_weight_decay 0.002 \
  --lambda_update_coeff 2.0 \
  --upconv_bound \
  --upconv_coeff 200.0 \
  --gpu_ids 0 \
  --ngf 48 \
  --ndf 128 \
  --gan_mode hinge \
  --direction BtoA \
  --load_size 256 \
  --no_flip \
  --save_epoch_freq 5 \
  --name mobile_mask_cityscapes \
  --AtoB_macs_threshold 4.5