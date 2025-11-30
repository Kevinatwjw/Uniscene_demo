import copy
import math
import os
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange, repeat
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from pyquaternion import Quaternion
from safetensors.torch import load_file as load_safetensors
from torch import autocast
from torch.nn.functional import adaptive_avg_pool2d, grid_sample
from tqdm import tqdm
from train import save_img_seq_to_video
from vwm.modules.diffusionmodules.sampling import EulerEDMSampler
from vwm.util import default, instantiate_from_config


def rt_mat_from_quaternion(r, t):
    r_mat = Quaternion(r).rotation_matrix
    rt_mat = np.eye(4)
    rt_mat[:3, :3] = r_mat
    rt_mat[:3, 3] = t
    return rt_mat


def get_rt_mat_from_pose_pickle(pose):
    ego_pose_r = pose["ego_pose"]["rotation"]
    ego_pose_t = np.array(pose["ego_pose"]["translation"])
    sensor2ego_r = pose["calibrated_sensor"]["rotation"]
    sensor2ego_t = np.array(pose["calibrated_sensor"]["translation"])
    e2g_mat = rt_mat_from_quaternion(ego_pose_r, ego_pose_t)
    cam2e_mat = rt_mat_from_quaternion(sensor2ego_r, sensor2ego_t)
    return e2g_mat, cam2e_mat


def get_intrinsic(pose):
    return np.array(pose["calibrated_sensor"]["camera_intrinsic"])


def warp_img(img, depth_map, K, img2_to_img1_mat):
    """
    Args:
        img: (C, h, w), tensor, color of img1
        depth: (H, W), tensor, depth of img2
        K: (3, 3), tensor, intrinsic
        img1_to_img2_mat: (3, 3), tensor, pose from img1 to img2
    Returns:
        new_img: (C, h, w)
    """
    H, W = depth_map.shape
    h, w = img.shape[1:]
    grid_h, grid_w = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack((grid_w, grid_h), 2).float().cuda()  # (H, W, 2), (u, v)
    depth = depth_map[..., None]
    udvdd1 = torch.cat([grid * depth, depth, torch.ones_like(depth)], dim=-1)
    K_4 = torch.eye(4, device=depth_map.device, dtype=K.dtype)
    K_4[:3, :3] = K
    K_4_inv = torch.linalg.inv(K_4)
    udvdd1_img1 = torch.einsum("mn,hwn->hwm", (K_4 @ img2_to_img1_mat.cuda() @ K_4_inv).float(), udvdd1)
    uv_img1 = udvdd1_img1[..., :2] / udvdd1_img1[..., 2:3]
    grid_uv = torch.cat([uv_img1[..., 0:1] / (W - 1) * 2 - 1, uv_img1[..., 1:2] / (H - 1) * 2 - 1], dim=-1)
    img2 = grid_sample(img[None], grid_uv[None], align_corners=True, padding_mode="zeros")
    if h != H:
        img2 = adaptive_avg_pool2d(img2, (h, w))
    img2 = img2[0]
    return img2


def process_warp(d, pose, pose2, img1):
    e2g_mat, c2e_mat = get_rt_mat_from_pose_pickle(pose)
    c2g_mat = e2g_mat @ c2e_mat
    e2g_mat, c2e_mat = get_rt_mat_from_pose_pickle(pose2)
    c2g_mat2 = e2g_mat @ c2e_mat
    c2_to_c1 = np.linalg.inv(c2g_mat) @ c2g_mat2
    img1 = torch.tensor(img1).float()
    img1 = img1.permute((2, 0, 1))
    depth = torch.tensor(d)
    intrinsic = get_intrinsic(pose2)
    intrinsic = torch.tensor(intrinsic)
    c2_to_c1 = torch.tensor(c2_to_c1)
    img2 = warp_img(img1, depth, K=intrinsic, img2_to_img1_mat=c2_to_c1)
    img2 = img2.permute(1, 2, 0)
    return img2


def apply_semantic_colormap(input_tensor):
    color_map = {
        0: [255, 120, 50],
        1: [255, 192, 203],
        2: [255, 255, 0],
        3: [0, 150, 245],
        4: [0, 255, 255],
        5: [255, 127, 0],
        6: [255, 0, 0],
        7: [255, 240, 150],
        8: [135, 60, 0],
        9: [160, 32, 240],
        10: [255, 0, 255],
        11: [139, 137, 137],
        12: [75, 0, 75],
        13: [150, 240, 80],
        14: [230, 230, 250],
        15: [0, 175, 0],
        16: [0, 255, 127],
        17: [222, 155, 161],
        18: [140, 62, 69],
        19: [227, 164, 30],
        20: [0, 128, 0],
    }
    lut = torch.tensor([color_map[i] for i in range(20)], dtype=torch.uint8)
    output_tensor = lut[input_tensor]
    return output_tensor


def init_model(version_dict, load_ckpt=True):
    config = OmegaConf.load(version_dict["config"])
    model = load_model_from_config(config, version_dict["ckpt"] if load_ckpt else None)
    return model


lowvram_mode = False


def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    return model


def load_model(model):
    model.cuda()


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()


def load_model_from_config(config, ckpt=None):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_svd = torch.load(ckpt, map_location="cpu")
            # dict contains:
            # "epoch", "global_step", "pytorch-lightning_version",
            # "state_dict", "loops", "callbacks", "optimizer_states", "lr_schedulers"
            if "global_step" in pl_svd:
                print(f"Global step: {pl_svd['global_step']}")
            svd = pl_svd["state_dict"]
        elif ckpt.endswith("safetensors"):
            svd = load_safetensors(ckpt)
        else:
            raise NotImplementedError("Please convert the checkpoint to safetensors first")

        missing, unexpected = model.load_state_dict(svd, strict=False)

    model = initial_model_load(model)
    model.eval()
    return model


def init_embedder_options(keys):
    # hardcoded demo settings, might undergo some changes in the future
    value_dict = dict()
    for key in keys:
        if key in ["fps_id", "fps"]:
            fps = 10
            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1
        elif key == "motion_bucket_id":
            value_dict["motion_bucket_id"] = 127  # [0, 511]
    return value_dict


def perform_save_locally(
    save_path,
    samples,
    mode,
    dataset_name,
    sample_index,
    n_frames,
    n_rounds=5,
    n_camera=6,
    save_names=None,
    to_image=False,
    to_grid=False,
):
    # assert mode in ["images", "grids", "videos"]
    merged_path = os.path.join(save_path, mode)
    os.makedirs(merged_path, exist_ok=True)
    samples = samples.cpu()

    if mode == "images":
        samples_flatten = rearrange(
            samples, "(n r t) c h w -> n (r t) c h w", n=n_camera, t=n_frames, r=n_rounds
        )  ## modify
        for cam_idx, cam_sample in enumerate(samples_flatten):
            frame_count = 0
            for sample in cam_sample:
                sample = rearrange(sample.numpy(), "c h w -> h w c")
                if "real" in save_path:
                    sample = 255.0 * (sample + 1.0) / 2.0
                else:
                    sample = 255.0 * sample
                image_save_path = os.path.join(merged_path, *save_names[frame_count][cam_idx].split("/")[-3:])
                os.makedirs(os.path.dirname(image_save_path), exist_ok=True)

                Image.fromarray(sample.astype(np.uint8)).save(image_save_path)
                frame_count += 1

    elif mode == "occ_semantic":
        img_seq = samples
        # video_save_path = os.path.join(merged_path, f"{dataset_name}_{sample_index:06}.mp4")
        video_save_path = os.path.join(
            merged_path, save_names[0][1].split("/")[-3] + "_" + save_names[0][1].split("/")[-1]
        )
        video_save_path = os.path.splitext(video_save_path)[0] + ".mp4"

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        img_seq = rearrange(img_seq, "(r v) t c h w  -> r v t h w c", r=n_rounds, v=n_camera, t=n_frames)  ## modify
        cam_view, T = img_seq.shape[1:3]
        ori_img_shape = list(img_seq.shape[3:])
        img_shape = [n_rounds * T, ori_img_shape[0] * 2, ori_img_shape[1] * 3, 3]
        cur_img_seq = np.zeros(img_shape).astype(np.uint8)
        for round_idx in range(n_rounds):
            for view_idx in range(cam_view):
                current_view = img_seq[round_idx, view_idx]

                current_view_frames = []
                for frame_idx in range(n_frames):
                    current_view_frame = apply_semantic_colormap(current_view[frame_idx][:, :, 0].long())
                    current_view_frames.append(current_view_frame)
                current_view = torch.stack(current_view_frames)

                if view_idx < 3:
                    cur_img_seq[
                        (round_idx) * T : (round_idx + 1) * T,
                        : ori_img_shape[0],
                        view_idx * ori_img_shape[1] : (view_idx * ori_img_shape[1]) + ori_img_shape[1],
                    ] = current_view
                else:
                    cur_img_seq[
                        (round_idx) * T : (round_idx + 1) * T,
                        ori_img_shape[0] :,
                        (view_idx - 3) * ori_img_shape[1] : ((view_idx - 3) * ori_img_shape[1]) + ori_img_shape[1],
                    ] = current_view
        out_img = []
        cur_img_seq = rearrange(cur_img_seq, "(r  t ) h w c -> r t h w c", r=n_rounds, t=n_frames)
        for round_idx in range(n_rounds):
            if round_idx < 1:
                out_img.append(cur_img_seq[round_idx, ...])
            else:
                out_img.append(cur_img_seq[round_idx, :-3, ...])
        out_img = np.concatenate(out_img, axis=0)
        save_img_seq_to_video(video_save_path, out_img.astype(np.uint8), 2)

        if to_image == True:
            samples_flatten = rearrange(
                img_seq, " r v t h w c -> v (r t) h w c", v=n_camera, t=n_frames, r=n_rounds
            )  ## modify
            for cam_idx, cam_sample in enumerate(samples_flatten):
                frame_count = 0
                for sample in cam_sample:
                    sample = apply_semantic_colormap(sample[:, :, 0].long())
                    sample = sample.numpy()
                    image_save_path = os.path.join(merged_path, *save_names[frame_count][cam_idx].split("/")[-3:])
                    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                    Image.fromarray(sample.astype(np.uint8)).save(image_save_path)
                    frame_count += 1

    elif mode == "occ_depth":
        img_seq = samples
        video_save_path = os.path.join(
            merged_path, save_names[0][1].split("/")[-3] + "_" + save_names[0][1].split("/")[-1]
        )
        video_save_path = os.path.splitext(video_save_path)[0] + ".mp4"

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        img_seq = rearrange(img_seq, "(r v) t c h w  -> r v t h w c", r=n_rounds, v=n_camera, t=n_frames)  ## modify
        cam_view, T = img_seq.shape[1:3]
        ori_img_shape = list(img_seq.shape[3:])
        img_shape = [n_rounds * T, ori_img_shape[0] * 2, ori_img_shape[1] * 3, 3]
        cur_img_seq = np.zeros(img_shape).astype(np.uint8)
        for round_idx in range(n_rounds):
            for view_idx in range(cam_view):
                current_view = img_seq[round_idx, view_idx]

                current_view_frames = []
                for frame_idx in range(n_frames):

                    x = current_view[frame_idx][:, :, 0].numpy()
                    mi = np.min(x)  # get minimum positive value
                    ma = np.max(x)
                    x = (x - mi) / (ma - mi + 1e-8)
                    # x = 1 - x  # reverse the colormap
                    x = (255 * x).astype(np.uint8)
                    current_view_frame = cv2.applyColorMap(x, cv2.COLORMAP_JET)
                    current_view_frame = cv2.convertScaleAbs(current_view_frame, alpha=1.1, beta=10)

                    current_view_frames.append(torch.tensor(current_view_frame))
                current_view = torch.stack(current_view_frames)

                if view_idx < 3:
                    cur_img_seq[
                        (round_idx) * T : (round_idx + 1) * T,
                        : ori_img_shape[0],
                        view_idx * ori_img_shape[1] : (view_idx * ori_img_shape[1]) + ori_img_shape[1],
                    ] = current_view
                else:
                    cur_img_seq[
                        (round_idx) * T : (round_idx + 1) * T,
                        ori_img_shape[0] :,
                        (view_idx - 3) * ori_img_shape[1] : ((view_idx - 3) * ori_img_shape[1]) + ori_img_shape[1],
                    ] = current_view
        out_img = []
        cur_img_seq = rearrange(cur_img_seq, "(r  t ) h w c -> r t h w c", r=n_rounds, t=n_frames)
        for round_idx in range(n_rounds):
            if round_idx < 1:
                out_img.append(cur_img_seq[round_idx, ...])
            else:
                out_img.append(cur_img_seq[round_idx, :-3, ...])
        out_img = np.concatenate(out_img, axis=0)
        save_img_seq_to_video(video_save_path, out_img.astype(np.uint8), 2)

        if to_image == True:
            samples_flatten = rearrange(
                img_seq, " r v t h w c -> v (r t) h w c", v=n_camera, t=n_frames, r=n_rounds
            )  ## modify
            for cam_idx, cam_sample in enumerate(samples_flatten):
                frame_count = 0
                for sample in cam_sample:

                    x = sample[:, :, 0].numpy()
                    mi = np.min(x)
                    ma = np.max(x)
                    x = (x - mi) / (ma - mi + 1e-8)
                    # x = 1 - x  # reverse the colormap
                    x = (255 * x).astype(np.uint8)
                    sample = cv2.applyColorMap(x, cv2.COLORMAP_JET)
                    sample = cv2.convertScaleAbs(sample, alpha=1.1, beta=10)

                    image_save_path = os.path.join(merged_path, *save_names[frame_count][cam_idx].split("/")[-3:])
                    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                    Image.fromarray(sample.astype(np.uint8)).save(image_save_path)
                    frame_count += 1

    elif mode == "grids":
        grid = torchvision.utils.make_grid(samples, nrow=n_frames * n_rounds)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
        if "real" in save_path:
            grid = 255.0 * (grid + 1.0) / 2.0
        else:
            grid = 255.0 * grid
        grid_save_path = os.path.join(
            merged_path, save_names[0][1].split("/")[-3] + "_" + save_names[0][1].split("/")[-1]
        )
        grid_save_path = os.path.splitext(grid_save_path)[0] + ".png"
        Image.fromarray(grid.astype(np.uint8)).save(grid_save_path)
    elif mode == "videos":
        video_save_path = os.path.join(
            merged_path, save_names[0][1].split("/")[-3] + "_" + save_names[0][1].split("/")[-1]
        )
        video_save_path = os.path.splitext(video_save_path)[0] + ".mp4"

        img_seq = rearrange(samples.numpy(), "t c h w -> t h w c")
        if "real" in save_path:
            img_seq = 255.0 * (img_seq + 1.0) / 2.0
        else:
            img_seq = 255.0 * img_seq

        img_seq = rearrange(img_seq, "(r  v  t ) h w c -> r v t h w c", r=n_rounds, v=n_camera, t=n_frames)  ## modify
        cam_view, T = img_seq.shape[1:3]
        ori_img_shape = list(img_seq.shape[3:])
        img_shape = [n_rounds * T, ori_img_shape[0] * 2, ori_img_shape[1] * 3, 3]
        cur_img_seq = np.zeros(img_shape).astype(np.uint8)
        for round_idx in range(n_rounds):
            for view_idx in range(cam_view):
                current_view = img_seq[round_idx, view_idx]
                if view_idx < 3:
                    cur_img_seq[
                        (round_idx) * T : (round_idx + 1) * T,
                        : ori_img_shape[0],
                        view_idx * ori_img_shape[1] : (view_idx * ori_img_shape[1]) + ori_img_shape[1],
                    ] = current_view
                else:
                    cur_img_seq[
                        (round_idx) * T : (round_idx + 1) * T,
                        ori_img_shape[0] :,
                        (view_idx - 3) * ori_img_shape[1] : ((view_idx - 3) * ori_img_shape[1]) + ori_img_shape[1],
                    ] = current_view
        out_img = []
        cur_img_seq = rearrange(cur_img_seq, "(r  t ) h w c -> r t h w c", r=n_rounds, t=n_frames)
        for round_idx in range(n_rounds):
            # if round_idx<1:
            out_img.append(cur_img_seq[round_idx, ...])
            # else:
            # out_img.append(cur_img_seq[round_idx, :-3, ... ] )
        out_img = np.concatenate(out_img, axis=0)
        save_img_seq_to_video(video_save_path, out_img.astype(np.uint8), 2)
    else:
        raise NotImplementedError


def init_sampling(
    sampler="EulerEDMSampler",
    guider="VanillaCFG",
    discretization="EDMDiscretization",
    steps=50,
    cfg_scale=2.5,
    num_frames=25,
):
    discretization_config = get_discretization(discretization)
    guider_config = get_guider(guider, cfg_scale, num_frames)
    sampler = get_sampler(sampler, steps, discretization_config, guider_config)
    return sampler


def get_discretization(discretization):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {"target": "vwm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"}
    elif discretization == "EDMDiscretization":
        discretization_config = {
            "target": "vwm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {"sigma_min": 0.002, "sigma_max": 700.0, "rho": 7.0},
        }
    else:
        raise NotImplementedError
    return discretization_config


def get_guider(guider="LinearPredictionGuider", cfg_scale=2.5, num_frames=25):
    if guider == "IdentityGuider":
        guider_config = {"target": "vwm.modules.diffusionmodules.guiders.IdentityGuider"}
    elif guider == "VanillaCFG":
        scale = cfg_scale

        guider_config = {"target": "vwm.modules.diffusionmodules.guiders.VanillaCFG", "params": {"scale": scale}}
    elif guider == "LinearPredictionGuider":
        max_scale = cfg_scale
        min_scale = 1.0

        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {"max_scale": max_scale, "min_scale": min_scale, "num_frames": num_frames},
        }
    elif guider == "TrianglePredictionGuider":
        max_scale = cfg_scale
        min_scale = 1.0

        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.TrianglePredictionGuider",
            "params": {"max_scale": max_scale, "min_scale": min_scale, "num_frames": num_frames},
        }
    else:
        raise NotImplementedError
    return guider_config


def get_sampler(sampler, steps, discretization_config, guider_config):
    if sampler == "EulerEDMSampler":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        sampler = EulerEDMSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown sampler {sampler}")
    return sampler


def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda", views=6):
    batch = dict()
    batch_uc = dict()

    for key in keys:
        if key in value_dict:
            if key in ["fps", "fps_id", "motion_bucket_id", "cond_aug"]:
                batch[key] = repeat(torch.tensor([value_dict[key]]).to(device), "1 -> (b f)", b=math.prod(N), f=views)
            elif key in ["command", "trajectory", "speed", "angle", "goal"]:
                batch[key] = repeat(value_dict[key][None].to(device), "1 ... -> (b f) ...", b=N[0], f=views)

            elif key in ["cond_frames", "cond_frames_without_noise"]:
                tmp = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
                batch[key] = tmp.permute(1, 0, 2, 3, 4)
                # batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0]  )
            elif key in ["occ_semantic", "occ_depth"]:
                batch[key] = rearrange(value_dict[key], "v t c h w -> (v t) c h w")

            else:
                # batch[key] = value_dict[key]
                raise NotImplementedError

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_condition(model, value_dict, num_samples, force_uc_zero_embeddings, device):
    load_model(model.conditioner)
    batch, batch_uc = get_batch(
        list(set([x.input_key for x in model.conditioner.embedders])), value_dict, [num_samples]
    )

    c, uc = model.conditioner.get_unconditional_conditioning(
        batch, batch_uc=batch_uc, force_uc_zero_embeddings=force_uc_zero_embeddings
    )

    unload_model(model.conditioner)

    return c, uc


def fill_latent(cond, length, cond_indices, device):
    latent = torch.zeros(length, *cond.shape[1:]).to(device)
    latent[:cond_indices] = cond
    return latent


@torch.no_grad()
def do_sample_occ(
    images,
    model,
    sampler,
    value_dict,
    num_rounds,
    num_frames,
    num_cameras,
    force_uc_zero_embeddings: Optional[List] = None,
    initial_cond_indices: Optional[List] = None,
    device="cuda",
    pose_seq=None,
):

    force_uc_zero_embeddings = default(force_uc_zero_embeddings, list())
    precision_scope = autocast

    with torch.no_grad(), precision_scope(device), model.ema_scope("Sampling"):
        ini_value_dict = copy.deepcopy(value_dict)
        value_dict["occ_depth"] = ini_value_dict["occ_depth"][:, :num_frames, ...]
        value_dict["occ_semantic"] = ini_value_dict["occ_semantic"][:, :num_frames, ...]
        c, uc = get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)
        load_model(model.first_stage_model)

        z = model.encode_first_stage(images[:, :num_frames, ...])

        samples_z = torch.zeros(
            (num_rounds * ((num_frames - 3) * num_cameras) + 3 * num_cameras * num_rounds, *z.shape[1:])
        ).to(device)

        sampling_progress = tqdm(total=num_rounds, desc="Dreaming")

        def denoiser(x, sigma, cond, cond_mask):
            return model.denoiser(model.model, x, sigma, cond, cond_mask)

        load_model(model.denoiser)
        load_model(model.model)

        initial_cond_mask = torch.zeros((num_cameras, num_frames)).to(device)
        prediction_cond_mask = torch.zeros((num_cameras, num_frames)).to(device)

        initial_cond_mask[:, [0, 1]] = 1
        prediction_cond_mask[:, [0, 1]] = 1

        initial_cond_mask = rearrange(initial_cond_mask, "n f -> (n f)")
        prediction_cond_mask = rearrange(prediction_cond_mask, "n f -> (n f)")

        noise = torch.randn_like(z)

        noise_tmp = rearrange(noise, "(v  t ) h w c -> v t h w c", v=num_cameras, t=num_frames)
        z_tmp = rearrange(z, "(v  t ) h w c -> v t h w c", v=num_cameras, t=num_frames)
        for num_frame in range(1, num_frames):
            for num_camera in range(num_cameras):
                ori_img = z_tmp[0][num_camera]
                ori_pose = pose_seq[0][num_camera]
                tgt_pose = pose_seq[num_frame][num_camera]
                tgt_depth = value_dict["occ_depth"][num_camera][num_frame][0]
                tgt_img = process_warp(tgt_depth, ori_pose, tgt_pose, ori_img)
                noise_tmp[num_camera][num_frame] += 0.03 * tgt_img
        noise = rearrange(noise_tmp, "v  t  h w c -> (v t) h w c", v=num_cameras, t=num_frames)

        sample = sampler(denoiser, noise, cond=c, uc=uc, cond_frame=z, cond_mask=initial_cond_mask)

        sampling_progress.update(1)

        sample_flatten = rearrange(sample, "(v t) c h w -> v t c h w", v=num_cameras, t=num_frames)
        z_flatten = rearrange(z, "(v t) c h w -> v t c h w", v=num_cameras, t=num_frames)
        samples_z_flatten = rearrange(
            samples_z, "(r v t) c h w -> r v t c h w", r=num_rounds, v=num_cameras, t=num_frames
        )
        sample_flatten[:, 0] = z_flatten[:, 0]
        sample = rearrange(sample_flatten, "v t c h w -> (v t) c h w")

        samples_z_flatten[0, :, :num_frames] = sample_flatten
        samples_z = rearrange(samples_z_flatten, "r v t c h w -> (r v t) c h w")

        for n in range(num_rounds - 1):
            print("starting round----->", n + 1)

            value_dict["occ_depth"] = ini_value_dict["occ_depth"][:, (n + 1) * num_frames : (n + 2) * num_frames, ...]
            value_dict["occ_semantic"] = ini_value_dict["occ_semantic"][
                :, (n + 1) * num_frames : (n + 2) * num_frames, ...
            ]

            z = model.encode_first_stage(images[:, (n + 1) * num_frames : (n + 2) * num_frames, ...])
            samples_x_for_guidance = model.decode_first_stage(z)

            samples_x_for_guidance = rearrange(
                samples_x_for_guidance, "(v t) c h w -> v t c h w", v=num_cameras, t=(num_frames)
            )

            value_dict["cond_frames_without_noise"] = samples_x_for_guidance[:, 0, ...].unsqueeze(0)

            value_dict["cond_frames"] = samples_x_for_guidance[:, 0, ...].unsqueeze(0) / model.scale_factor

            c, uc = get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)

            noise = torch.randn_like(z)
            sample = sampler(denoiser, noise, cond=c, uc=uc, cond_frame=z, cond_mask=prediction_cond_mask)
            sampling_progress.update(1)

            samples_z, sample = (
                rearrange(samples_z, "(r v t) c h w -> r v t c h w", r=num_rounds, v=num_cameras, t=num_frames),
                rearrange(sample, "(v t) c h w -> v t c h w", v=num_cameras, t=num_frames),
            )

            samples_z[n + 1, :, :, ...] = sample[:, :, ...]
            samples_z, sample = rearrange(samples_z, "r v t c h w -> (r v t) c h w"), rearrange(
                sample, "v t c h w -> (v t) c h w"
            )

        samples_x = model.decode_first_stage(samples_z)

        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        out_images = images

        return samples, samples_z, out_images
