#!/usr/bin/env bash
python prune.py --dataroot ./database/cityscapes \
  --model pix2pix \
  --mask \
  --checkpoints_dir ./experiments/mask_cityscapes \
  --name pruned_cityscapes \
  --load_path ./experiments/mask_cityscapes/model_best.pth \
  --gpu_ids 0 \
  --ngf 48 \
  --ndf 128 \
  --gan_mode hinge \
  --direction BtoA \
  --load_size 256 \
  --no_flip \
  --finetune \
  --save_epoch_freq 5 \
  --lambda_attention_distill 10.0 \
  --lambda_discriminator_distill 0.0001 \
  --pretrain_path ../pretrain/cityscapes_pretrain.pth