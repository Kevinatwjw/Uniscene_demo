#!/bin/bash

# 适配 Mini 数据集的训练冒烟测试脚本
# 注意：路径已根据你的目录结构进行修正

# 1. nproc_per_node=1 (单卡训练)
# 2. imageset -> 指向 val_mini_dict.pkl (因为我们只生成了val的BEV,指向了能跑通的地方)
# 3. bev_path -> 指向 ckpt/VAE_Mini/val/bevmap_4 (同上，指向实际存在的BEV数据)
# 4. gts_path -> 指向 ../data/gts (你重命名后的文件夹)
# 5. vae_ckpt -> 指向 ckpt/VAE/ 下的权重

torchrun --nnodes=1 --nproc_per_node=1 \
    train_OccDiT.py \
    --imageset="./data/nuscenes_mmdet3d-12Hz/nuscenes_infos_val_mini_dict.pkl" \
    --bev_path="./ckpt/VAE_Mini/val/bevmap_4" \
    --gts_path="../data/gts" \
    --vae_ckpt="./ckpt/VAE/epoch_296.pth" \
    --vae_config="./config/train_vae_4_DwT_L_me.py" \
    --lambda_noise_prior=0.03 \
    --global-batch-size=1 \
    --results-dir="./results/mini_train_test"