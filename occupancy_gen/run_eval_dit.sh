torchrun --nnodes=1 --nproc_per_node=1 \
    eval_OccDiT.py \
    --imageset="./data/nuscenes_infos_val_temporal_v3_scene.pkl" \
    --bev_path="./data/step2/val/bevmap_4" \
    --gts_path="./data/gts" \
    --ckpt="/ckpt/DiT/0600000.pt" \
    --vae_ckpt="/ckpt/VAE/epoch_296.pth" \
    --vae_config="./config/train_vae_4_DwT_L_me.py"\
    --ae_ckpt="/ckpt/AE_eval/epoch_196.pth"\
    --lambda_noise_prior=0.05 \
    --cfg-scale=1.0 \
    --vis \
    --genocc_save_path=$gen_occ_path
