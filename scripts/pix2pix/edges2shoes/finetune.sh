#!/usr/bin/env bash
python prune.py --dataroot ./database/edges2shoes-r \
  --model pix2pix \
  --mask \
  --checkpoints_dir ./experiments/mask_edges2shoes \
  --name pruned_edges2shoes \
  --load_path ./experiments/mask_edges2shoes/model_best.pth \
  --gpu_ids 0 \
  --ngf 48 \
  --ndf 128 \
  --gan_mode hinge \
  --load_size 256 \
  --no_flip \
  --batch_size 4 \
  --finetune \
  --save_epoch_freq 5 \
  --lambda_attention_distill 10.0 \
  --lambda_discriminator_distill 0.0001 \
  --pretrain_path ../pretrain/edges2shoes_pretrain.pth