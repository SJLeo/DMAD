#!/usr/bin/env bash
python train.py --dataroot ../datasets/edges2shoes \
  --model pix2pix \
  --mask \
  --mask_weight_decay 0.01 \
  --gpu_ids 0 \
  --ngf 48 \
  --ndf 128 \
  --gan_mode hinge \
  --load_size 256 \
  --no_flip \
  --batch_size 4 \
  --save_epoch_freq 5 \
  --name mask_edges2shoes \
  --AtoB_macs_threshold 3.0 \
  --frozen_threshold 0.8