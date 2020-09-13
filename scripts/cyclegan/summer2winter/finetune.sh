#!/usr/bin/env bash
python prune.py --dataroot ./database/summer2winter \
  --model cyclegan \
  --mask \
  --checkpoints_dir ./experiments/mask_summer2winter \
  --name pruned_summer2winter \
  --load_path ./experiments/mask_summer2winter/model_best.pth \
  --gpu_ids 0 \
  --finetune \
  --lambda_attention_distill 100.0 \
  --lambda_discriminator_distill 0.0001 \
  --pretrain_path ../pretrain/summer2winter_pretrain.pth