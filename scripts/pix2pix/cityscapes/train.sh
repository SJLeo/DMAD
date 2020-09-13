#!/usr/bin/env bash
python train.py --dataroot ./database/cityscapes \
  --model pix2pix \
  --mask \
  --mask_weight_decay 0.01 \
  --gpu_ids 0 \
  --ngf 48 \
  --ndf 128 \
  --gan_mode hinge \
  --direction BtoA \
  --load_size 256 \
  --no_flip \
  --save_epoch_freq 5 \
  --name mask_cityscapes \
  --AtoB_macs_threshold 4.0 \
  --frozen_threshold 0.7

