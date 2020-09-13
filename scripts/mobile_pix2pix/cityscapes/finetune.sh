#!/usr/bin/env bash
python prune.py --dataroot ./database/cityscapes \
  --model mobilepix2pix \
  --mask \
  --checkpoints_dir ./experiments/mobile_mask_cityscapes \
  --name mobile_pruned_cityscapes \
  --load_path ./experiments/mobile_mask_cityscapes/model_best.pth \
  --gpu_ids 0 \
  --ngf 48 \
  --ndf 128 \
  --gan_mode hinge \
  --direction BtoA \
  --load_size 256 \
  --no_flip \
  --save_epoch_freq 5 \
  --finetune \
  --lambda_attention_distill 10.0 \
  --lambda_discriminator_distill 0.0001 \
  --pretrain_path ../pretrain/cityscapes_pretrain.pth