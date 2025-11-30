torchrun --nnodes=1 --nproc_per_node=8 \
    train_OccDiT.py \
    --imageset="./data/nuscenes_infos_train_temporal_v3_scene.pkl" \
    --bev_path="./data/step2/train/bevmap_4" \
    --gts_path="./data/gts" \
    --vae_ckpt="/ckpt/VAE/epoch_296.pth" \
    --vae_config="./config/train_vae_4_DwT_L_me.py" \
    --lambda_noise_prior=0.03 \
