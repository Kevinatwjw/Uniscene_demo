#!/bin/bash

# 适配 Mini 数据集的评估脚本 (路径已修正适配你的截图)

# 1. 显卡数设为 1
# 2. imageset 指向 mini dict pkl
# 3. bev_path 指向之前生成的 ckpt/VAE_Mini/val/bevmap_4
# 4. gts_path 指向 data/gts
# 5. [关键修改] 权重路径适配你的 ckpt 目录结构

torchrun --nnodes=1 --nproc_per_node=1 \
    eval_OccDiT.py \
    --imageset="./data/nuscenes_mmdet3d-12Hz/nuscenes_infos_val_mini_dict.pkl" \
    --bev_path="./ckpt/VAE_Mini/val/bevmap_4" \
    --gts_path="../data/gts" \
    --ckpt="./ckpt/DiT/0600000.pt" \
    --vae_ckpt="./ckpt/VAE/epoch_296.pth" \
    --vae_config="./config/train_vae_4_DwT_L_me.py" \
    --ae_ckpt="./ckpt/AE_eval/epoch_196.pth" \
    --lambda_noise_prior=0.05 \
    --cfg-scale=1.0 \
    --vis \
    --global-batch-size=1 \
    --genocc_save_path="./outputs/mini_eval_results"