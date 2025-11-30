# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A minimal training script for DiT using PyTorch DDP."""
import argparse
import os
from copy import deepcopy
from time import time

import imageio
import numpy as np
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from dataload_util import CustomDataset_Tframe_continuous
from dataset import get_nuScenes_label_name
from diffusion import create_diffusion
from diffusion.models import OccDiT
from download import find_model
from mmengine import Config
from mmengine.registry import MODELS
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.metric_util import multi_step_fid_mmd, multi_step_MeanIou, multi_step_TemporalConsistency

from occupancy_gen.visualize_nuscenes_occupancy import draw_nusc_occupancy_Bev_Front
import model_vae  # noqa
from model_vae.VAE.AE_eval import Autoencoder_2D  # FID MMD

# logger
from mmengine.logging import MMLogger

#################################################################################
#                                  Eval Loop                                    #
#################################################################################


def main(args):
    """Trains a new DiT model."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    torch.set_grad_enabled(False)
    # Variables for monitoring/logging purposes:
    start_epoch = 0
    time()

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    # rank=args.local_rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    Tframe = 5
    use_bev_concat = True
    use_occ_meta = True
    use_vq = False
    use_noise_prior = True

    meta_num_mode = 3
    if meta_num_mode == 1:
        meta_num = 4 * Tframe + 12 * (Tframe - 1)
    elif meta_num_mode == 2:
        meta_num = 4 + 12 * (Tframe - 1)
    else:
        meta_num = 4
    in_ch = args.in_ch
    vis = args.vis
    ref_idx = 0
    scale_factor = 70

    bev_ch_use = [0, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    lambda_np = args.lambda_noise_prior
    DiT_cfg = {
        "depth": 12,
        "in_channels": in_ch,
        "hidden_size": 512,
        "use_label": False,
        "use_bev_concat": use_bev_concat,
        "bev_in_ch": 1,
        "bev_out_ch": 1,
        "use_meta": use_occ_meta,
        "meta_num": meta_num,
        "direct_concat": True,
        "Tframe": Tframe,
    }

    Dit_model = OccDiT(**DiT_cfg).to(device)

    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    Dit_model.load_state_dict(state_dict, strict=True)
    Dit_model.eval()  # important!

    ckpt_path_list = ckpt_path.split("/")
    log_path = "/".join(ckpt_path_list[:-2])
    vis_root = "/".join(ckpt_path_list[:-2]) + f"/vis_demo/video/"
    zvis_root = "/".join(ckpt_path_list[:-2]) + "/vis_demo/zvis/"

    diffusion = create_diffusion("ddim50")
    iter_num = ckpt_path_list[-1].split(".")[0]
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    Dit_model = DDP(Dit_model.to(device), device_ids=[rank])

    gen_occ_save_path = args.genocc_save_path
    os.makedirs(gen_occ_save_path, exist_ok=True)
    # logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    imageset = args.imageset
    bev_path = args.bev_path
    gts_path = args.gts_path

    dataset = CustomDataset_Tframe_continuous(
        imageset, gts_path, bev_path, bev_ch_use, meta_num=4, Tframe=Tframe, return_token=True
    )

    p_time = Tframe

    bz = int(args.global_batch_size // dist.get_world_size())
    # bz=1
    if vis:
        bz = 1
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=bz,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare models for training:
    # update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    Dit_model.eval()  # important! This enables embedding dropout for classifier-free guidance
    # ema.eval()  # EMA model should always be in eval mode

    cfg1 = Config.fromfile(args.vae_config)

    OccVAE = MODELS.build(cfg1.model)
    OccVAE = OccVAE.to(device)

    resume_from = args.vae_ckpt
    vae_ckpt = torch.load(resume_from, map_location="cpu")
    OccVAE.load_state_dict(vae_ckpt["state_dict"], strict=True)

    OccVAE.eval()

    ae_eval = Autoencoder_2D(num_classes=18, expansion=4)
    ae_ckpt_path = args.ae_ckpt
    ae_ckpt = torch.load(ae_ckpt_path, map_location="cpu")
    ae_eval.load_state_dict(ae_ckpt["state_dict"], strict=True)
    ae_eval = ae_eval.to(device)
    ae_eval.eval()

    if not vis:
        log_file = os.path.join(log_path, f"Dit_eval_ddim_continuous_mVAE_{iter_num}_{args.cfg_scale}_{lambda_np}.log")
        logger = MMLogger("genocc", log_file=log_file)
        MMLogger._instance_dict["genocc"] = logger
        logger.info(f"Cfg scale:{args.cfg_scale}")

    label_name = get_nuScenes_label_name(cfg1.label_mapping)
    unique_label = np.asarray(cfg1.unique_label)
    unique_label_str = [label_name[l] for l in unique_label]
    CalMeanIou_sem = multi_step_MeanIou(
        unique_label, cfg1.get("ignore_label", -100), unique_label_str, "sem", times=p_time
    )
    CalMeanIou_sem.reset()

    CalMeanIou_vox = multi_step_MeanIou([1], cfg1.get("ignore_label", -100), ["occupied"], "vox", times=p_time)
    CalMeanIou_vox.reset()

    Cal_fid_mmd = multi_step_fid_mmd()
    Cal_TC = multi_step_TemporalConsistency("TC", times=p_time)
    Cal_TC.reset()

    with torch.no_grad():
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)

            for i_iter_val, (occ_ori, y, occ_meta, pose_meta, tokens, bev_ori) in enumerate(tqdm(loader)):

                occ_ori = occ_ori.to(device)
                y = y.to(device)
                images = []

                with torch.no_grad():
                    z_ori = OccVAE.encode(occ_ori) * scale_factor  # x: B C T H W

                if use_noise_prior == True:
                    z_ori_s = z_ori[:, :, 0].unsqueeze(2)  #
                    z_ori_s = z_ori[:, :, ref_idx].unsqueeze(2)  #
                    z_ori_s = z_ori_s.repeat(1, 1, Tframe, 1, 1)
                    z_ori_s = z_ori_s.to(device)
                    z = torch.randn((bz, in_ch, Tframe, 50, 50), device=device) + lambda_np * z_ori_s  # z: [N,C,T,H,W]
                else:
                    z = torch.randn((bz, in_ch, Tframe, 50, 50), device=device)

                z = torch.cat([z, z], 0)
                y_null = -torch.ones_like(y)
                y = torch.cat([y, y_null], 0)

                if use_occ_meta:
                    if meta_num_mode == 1:
                        new_meta = torch.cat(
                            (occ_meta.reshape(-1, 4 * Tframe), pose_meta.reshape(-1, 12 * (Tframe - 1))), dim=1
                        )
                    elif meta_num_mode == 2:
                        new_meta = torch.cat((occ_meta[:, ref_idx], pose_meta.reshape(-1, 12 * (Tframe - 1))), dim=1)
                    else:
                        new_meta = occ_meta[:, ref_idx]
                    new_meta = new_meta.to(device)
                    new_meta = torch.cat([new_meta, new_meta], 0)

                    model_kwargs = dict(y=y, meta=new_meta, cfg_scale=args.cfg_scale)

                else:
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

                samples = diffusion.ddim_sample_loop(
                    Dit_model.module.forward_with_cfg,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                )

                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                samples = samples / scale_factor

                rec_shape = [bz, Tframe, 200, 200, 16]
                if use_vq == False:
                    result = OccVAE.generate(samples, rec_shape)
                else:
                    result = OccVAE.generate_vq(samples, rec_shape)
                logit = result["logits"]
                pred = logit.argmax(dim=-1)  #  200, 200, 16

                CalMeanIou_sem._after_step(pred, occ_ori)

                target_occs_iou = deepcopy(occ_ori)
                target_occs_iou[target_occs_iou != 17] = 1
                target_occs_iou[target_occs_iou == 17] = 0
                pred_iou = deepcopy(pred)
                pred_iou[pred_iou != 17] = 1
                pred_iou[pred_iou == 17] = 0

                CalMeanIou_vox._after_step(pred_iou, target_occs_iou)

                # occ_ori_noT = occ_ori.reshape(-1,200,200,16)
                # pred_noT = pred.reshape(-1,200,200,16)

                ae_feature_ori = ae_eval.forward_eval(occ_ori)  # B*T,2048
                ae_feature_gen = ae_eval.forward_eval(pred)

                Cal_fid_mmd._after_step(ae_feature_ori, ae_feature_gen)
                Cal_TC._after_step(ae_feature_gen.reshape(bz, p_time, -1))

                dist.barrier()

                # Save Generated npz
                for i_t in range(Tframe):
                    pred_bz0 = pred.squeeze().cpu().numpy()
                    print(pred_bz0.shape, tokens[i_t])
                    np.savez_compressed(f"{gen_occ_save_path}/{tokens[i_t][0]}.npz", occ=pred_bz0[i_t])

                if vis:
                    # bz = 1
                    # dst_dir = os.path.join(vis_root,str(i_iter_val))
                    os.makedirs(vis_root, exist_ok=True)

                    y[0].squeeze().cpu().numpy()
                    pred = pred.squeeze().cpu().numpy()
                    occ_ori = occ_ori.squeeze().cpu().numpy()

                    save_path = f"{vis_root}/{i_iter_val}"
                    video_filepath = os.path.join(save_path, "output.mp4")
                    images = []
                    for i in range(Tframe):

                        pred_i = pred[i]
                        token = tokens[i][0]
                        bev_layout = bev_ori[0][i].cpu().numpy()
                        save_folder = os.path.join(save_path, "{}_assets".format(token))
                        cat_save_file = os.path.join(save_path, "{}_cat_vis.png".format(token))

                        cat_image = draw_nusc_occupancy_Bev_Front(
                            voxels=pred_i,
                            vox_origin=np.array([-40, -40, -1]),
                            voxel_size=np.array([0.4, 0.4, 0.4]),
                            grid=np.array([200, 200, 16]),
                            pred_lidarseg=None,
                            target_lidarseg=None,
                            save_folder=save_folder,
                            cat_save_file=cat_save_file,
                            cam_positions=None,
                            focal_positions=None,
                            bev_layout=bev_layout,
                        )

                        images.append(cat_image)

                    with imageio.get_writer(video_filepath, fps=2) as video:
                        for image in images:
                            image = np.array(image)
                            video.append_data(image)

            val_miou, avg_val_miou = CalMeanIou_sem._after_epoch()
            val_iou, avg_val_iou = CalMeanIou_vox._after_epoch()

            logger.info(f"Avg mIoU: %.2f%%" % (avg_val_miou))
            logger.info(f"Avg IoU: %.2f%%" % (avg_val_iou))

            val_TC = Cal_TC._after_epoch()
            logger.info(f"Avg TC: %.4f" % (val_TC))
            if rank == 0:
                fid, mmd = Cal_fid_mmd._after_epoch()
                logger.info(f"FID: %.4f" % (fid))
                logger.info(f"MMD: %.6f" % (mmd))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--imageset", type=str)
    parser.add_argument("--bev_path", type=str)
    parser.add_argument("--gts_path", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--in_ch", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default="ckpt/DiT/0600000.pt")
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--vae_ckpt", type=str, default="ckpt/VAE/epoch_296.pth")
    parser.add_argument("--vae_config", type=str, default="config/train_vae_4_DwT_L_me.py")
    parser.add_argument("--ae_ckpt", type=str, default="ckpt/AE_eval/epoch_196.pth")
    parser.add_argument("--lambda_noise_prior", type=float, default=0.05)
    parser.add_argument("--num-sampling-steps", type=int, default=50)  # 1000
    parser.add_argument("--genocc_save_path", type=str, default="gen_occ_save")
    args = parser.parse_args()
    main(args)
