import argparse
import io
import os
import pickle
import random

import numpy as np
import torch
from einops import rearrange
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pytorch_lightning import seed_everything
from sample_utils import (
    do_sample_occ,
    init_embedder_options,
    init_model,
    init_sampling,
    perform_save_locally,
    set_lowvram_mode,
)
from torchvision import transforms

VERSION2SPECS = {
    "vwm": {
        "config": "configs/example/nusc_mv.yaml",
        "ckpt": "ckpts/video_pretrained.safetensors",
    }
}


DATASET2SOURCES = {
    "NUSCENES_OCC": {
        "data_root": "./data/nuscenes/",
    },
}

cam_names = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]
data_root = "./data/nuscenes/"
version = "v1.0-trainval"
nusc = NuScenes(version=version, dataroot=data_root, verbose=True)


with open("./data/video_pickle_data.pkl", "rb") as f:
    anno_data = pickle.load(f)
    occ_anno_file = anno_data["occ_samples"]
    anno_file = anno_data["anno_file"]
    occ_samples = anno_data["anno_file_occ"]
    nusc_all_cam_data = anno_data["nusc_all_cam_data"]
    nusc_all_cam_path_data = anno_data["nusc_all_cam_path_data"]


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--dataset", type=str, default="NUSCENES_OCC", help="dataset name")
    parser.add_argument(
        "--occ_data_root",
        type=str,
        default="None",
    )

    parser.add_argument("--annos", type=str, default="video_pickle_file.pkl", help="number of frames for each round")
    parser.add_argument("--n_frames", type=int, default=8, help="number of frames for each round")
    parser.add_argument("--n_rounds", type=int, default=1, help="number of sampling rounds")
    parser.add_argument(
        "--action",
        type=str,
        default="free",  # "trajectory",
        help="action mode for control, such as trajectory, cmd, steer, goal",
    )
    parser.add_argument("--height", type=int, default=256, help="target height of the generated video")
    parser.add_argument("--width", type=int, default=512, help="target width of the generated video")
    parser.add_argument(
        "--rand_gen", default=False, action="store_false", help="whether to generate samples randomly or sequentially"
    )

    parser.add_argument("--low_vram", action="store_true", help="whether to save memory or not")
    parser.add_argument("--version", type=str, default="vwm", help="model version")
    parser.add_argument("--save", type=str, default="None", help="directory to save samples")
    parser.add_argument("--n_conds", type=int, default=1, help="number of initial condition frames for the first round")
    parser.add_argument("--seed", type=int, default=23, help="random seed for seed_everything")
    parser.add_argument("--cfg_scale", type=float, default=2.5, help="scale of the classifier-free guidance")
    parser.add_argument("--cond_aug", type=float, default=0.0, help="strength of the noise augmentation")
    parser.add_argument("--n_steps", type=int, default=50, help="number of sampling steps")

    return parser


def get_pose(frame_path="samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151523912404.jpg"):
    index = nusc_all_cam_path_data.index(frame_path)
    ego_pose_token = nusc_all_cam_data[index]["ego_pose_token"]
    calibrated_sensor_token = nusc_all_cam_data[index]["calibrated_sensor_token"]
    ego_pose = nusc.get("ego_pose", ego_pose_token)
    calibrated_sensor = nusc.get("calibrated_sensor", calibrated_sensor_token)
    return {"ego_pose": ego_pose, "calibrated_sensor": calibrated_sensor}


def get_sample(
    selected_index=0,
    dataset_name="NUSCENES",
    num_frames=20,
    action_mode="free",
    depth_path=["depth_cor", "depth_data"],
    semantic_path=["semantic_color", "semantic"],
    occ_samples=None,
    n_rounds=None,
    occ_data_root=None,
):
    dataset_dict = DATASET2SOURCES[dataset_name]
    action_dict = None
    occ_semantic_path_list = None
    occ_depth_path_list = None

    if dataset_name == "IMG":
        image_list = os.listdir(dataset_dict["data_root"])
        total_length = len(image_list)
        while selected_index >= total_length:
            selected_index -= total_length
        image_file = image_list[selected_index]

        path_list = [os.path.join(dataset_dict["data_root"], image_file)] * num_frames
    else:
        all_samples = anno_file
        total_length = len(all_samples)

        while selected_index >= total_length:
            selected_index -= total_length
        sample_dict = all_samples[selected_index]

        pose_list = list()
        path_list = list()
        occ_semantic_path_list = list()
        occ_depth_path_list = list()

        if dataset_name == "NUSCENES_OCC":
            for index in range(num_frames * n_rounds):
                ring_pose_path = []
                ring_image_path = []
                ring_occ_semantic_paths = []
                ring_occ_depth_paths = []
                for i in range(len(cam_names)):
                    image_path = os.path.join(
                        dataset_dict["data_root"], sample_dict["frames"][cam_names[i]][index]
                    )  ## modify

                    occ_semantic_paths = os.path.join(
                        occ_data_root,
                        occ_samples[sample_dict["frames"][cam_names[i]][index]],
                        semantic_path[1] + ".npz",
                    )
                    occ_depth_paths = os.path.join(
                        occ_data_root, occ_samples[sample_dict["frames"][cam_names[i]][index]], depth_path[1] + ".npz"
                    )
                    ring_pose_path.append(get_pose(frame_path=os.path.join(*image_path.split("/")[-3:])))
                    ring_image_path.append(image_path)
                    ring_occ_semantic_paths.append(occ_semantic_paths)
                    ring_occ_depth_paths.append(occ_depth_paths)

                ring_image_path = ring_image_path[-1:] + ring_image_path[:-1]

                pose_list.append(ring_pose_path)
                path_list.append(ring_image_path)
                occ_semantic_path_list.append(ring_occ_semantic_paths)
                occ_depth_path_list.append(ring_occ_depth_paths)

            if action_mode != "free":
                action_dict = dict()
                if action_mode == "traj" or action_mode == "trajectory":
                    action_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
                elif action_mode == "cmd" or action_mode == "command":
                    action_dict["command"] = torch.tensor(sample_dict["cmd"])
                elif action_mode == "steer":
                    # scene might be empty
                    if sample_dict["speed"]:
                        action_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
                    # scene might be empty
                    if sample_dict["angle"]:
                        action_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
                elif action_mode == "goal":
                    # point might be invalid
                    if sample_dict["z"] > 0 and 0 < sample_dict["goal"][0] < 1600 and 0 < sample_dict["goal"][1] < 900:
                        action_dict["goal"] = torch.tensor(
                            [sample_dict["goal"][0] / 1600, sample_dict["goal"][1] / 900]
                        )
                else:
                    raise ValueError(f"Unsupported action mode {action_mode}")

        else:
            raise ValueError(f"Invalid dataset {dataset_name}")
    return path_list, selected_index, total_length, action_dict, occ_semantic_path_list, occ_depth_path_list, pose_list


def load_img(
    file_name,
    target_height=320,
    target_width=576,
    device="cuda",
    dataset=None,
    flag="image",
    current_cam_name=None,
    curren_frame_name=None,
):
    cam_dict = {
        "CAM_FRONT_LEFT": 5,
        "CAM_FRONT": 0,
        "CAM_FRONT_RIGHT": 1,
        "CAM_BACK_RIGHT": 2,
        "CAM_BACK": 3,
        "CAM_BACK_LEFT": 4,
    }

    if file_name is not None:
        with open(file_name, "rb") as file:
            image_data = file.read()
            if flag == "image":
                image = Image.open(io.BytesIO(image_data))
            elif flag == "semantic":
                image = np.load(io.BytesIO(image_data), allow_pickle=True)[current_cam_name]
                image = Image.fromarray(np.array(image, dtype=np.uint8))
            elif flag == "depth":
                image = np.load(io.BytesIO(image_data), allow_pickle=True)[current_cam_name]
                image = Image.fromarray(np.array(image, dtype=np.uint8))

        if not image.mode == "RGB":
            image = image.convert("RGB")
    else:
        raise ValueError(f"Invalid image file {file_name}")
    ori_w, ori_h = image.size

    if ori_w / ori_h > target_width / target_height:
        tmp_w = int(target_width / target_height * ori_h)
        left = (ori_w - tmp_w) // 2
        right = (ori_w + tmp_w) // 2
        image = image.crop((left, 0, right, ori_h))

        if flag == "semantic" or flag == "depth":
            image = image.resize((target_width, target_height), resample=Image.NEAREST)
        else:
            image = image.resize((target_width, target_height), resample=Image.LANCZOS)

    elif ori_w / ori_h < target_width / target_height:
        tmp_h = int(target_height / target_width * ori_w)
        top = (ori_h - tmp_h) // 2
        bottom = (ori_h + tmp_h) // 2
        image = image.crop((0, top, ori_w, bottom))

        if flag == "semantic" or flag == "depth":
            image = image.resize((target_width, target_height), resample=Image.NEAREST)
        else:
            image = image.resize((target_width, target_height), resample=Image.LANCZOS)

    if not image.mode == "RGB":
        image = image.convert("RGB")

    if flag == "semantic":
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)

    elif flag == "depth":
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        image = (image / 100.0) * 2.0 - 1.0
    else:
        image = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])(image)

    return image.to(device)


if __name__ == "__main__":
    parser = parse_args()
    opt, unknown = parser.parse_known_args()
    set_lowvram_mode(opt.low_vram)
    version_dict = VERSION2SPECS[opt.version]

    model = init_model(version_dict)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])

    sample_index = 500
    count_index = 500
    while sample_index >= 0:
        seed_everything(opt.seed)

        (
            frame_list,
            sample_index,
            dataset_length,
            action_dict,
            occ_semantic_path_list,
            occ_depth_path_list,
            pose_list,
        ) = get_sample(
            sample_index,
            opt.dataset,
            opt.n_frames,
            opt.action,
            occ_samples=occ_samples,
            n_rounds=opt.n_rounds,
            occ_data_root=opt.occ_data_root,
        )

        if opt.dataset == "NUSCENES_OCC":
            pose_seq = list()
            img_seq = list()
            occ_semantic_seq = list()
            occ_depth_seq = list()

            if isinstance(frame_list[0], list):  ## modify

                for each_path in frame_list:
                    img_seq_mv = []
                    for each_cam_path in each_path:
                        img = load_img(each_cam_path, opt.height, opt.width, dataset=opt.dataset)
                        img_seq_mv.append(img)
                    img_seq_mv_torch = torch.stack(img_seq_mv)
                    img_seq.append(img_seq_mv_torch)
                i = 0

                for each_path in pose_list:
                    pose_mv = []
                    j = 0
                    for each_pos_path in each_path:
                        pose_mv.append(each_pos_path)
                        j += 1
                    pose_mv = pose_mv[-1:] + pose_mv[:-1]
                    pose_seq.append(pose_mv)
                    i += 1
                i = 0

                for each_path in occ_semantic_path_list:
                    img_seq_mv = []
                    j = 0
                    for each_cam_path in each_path:
                        if each_cam_path == "None":
                            img = torch.zeros(img.shape).cuda().to(torch.float16)
                        else:
                            img = load_img(
                                each_cam_path,
                                opt.height,
                                opt.width,
                                dataset=opt.dataset,
                                flag="semantic",
                                current_cam_name=j,
                                curren_frame_name=i,
                            )
                        img_seq_mv.append(img)
                        j += 1

                    img_seq_mv = img_seq_mv[-1:] + img_seq_mv[:-1]
                    img_seq_mv_torch = torch.stack(img_seq_mv)
                    occ_semantic_seq.append(img_seq_mv_torch)
                    i += 1
                i = 0

                for each_path in occ_depth_path_list:
                    img_seq_mv = []
                    j = 0
                    for each_cam_path in each_path:
                        if each_cam_path == "None":
                            img = torch.zeros(img.shape).cuda().to(torch.float16)
                        else:
                            img = load_img(
                                each_cam_path,
                                opt.height,
                                opt.width,
                                dataset=opt.dataset,
                                flag="depth",
                                current_cam_name=j,
                                curren_frame_name=i,
                            )
                        img_seq_mv.append(img)
                        j += 1

                    img_seq_mv = img_seq_mv[-1:] + img_seq_mv[:-1]
                    img_seq_mv_torch = torch.stack(img_seq_mv)
                    occ_depth_seq.append(img_seq_mv_torch)
                    i += 1
            else:
                for each_path in frame_list:
                    img = load_img(each_path, opt.height, opt.width, dataset=opt.dataset)
                    img_seq.append(img)
            img_seq = torch.stack(img_seq)
            images = img_seq.permute(1, 0, 2, 3, 4)
            occ_semantic_seq = torch.stack(occ_semantic_seq)
            occ_semantic_seq = occ_semantic_seq.permute(1, 0, 2, 3, 4)
            occ_depth_seq = torch.stack(occ_depth_seq)
            occ_depth_seq = occ_depth_seq.permute(1, 0, 2, 3, 4)

            value_dict = init_embedder_options(unique_keys)
            cond_img = img_seq[0][None]

            value_dict["cond_frames_without_noise"] = cond_img
            value_dict["cond_frames"] = cond_img + opt.cond_aug * torch.randn_like(cond_img)
            value_dict["cond_aug"] = opt.cond_aug
            value_dict["occ_semantic"] = occ_semantic_seq
            value_dict["occ_depth"] = occ_depth_seq

            if action_dict is not None:
                for key, value in action_dict.items():
                    value_dict[key] = value

        if opt.n_rounds > 1:
            guider = "TrianglePredictionGuider"
        else:
            guider = "VanillaCFG"
        sampler = init_sampling(guider=guider, steps=opt.n_steps, cfg_scale=opt.cfg_scale, num_frames=opt.n_frames)

        if opt.dataset == "NUSCENES_OCC":
            uc_keys = [
                "cond_frames",
                "cond_frames_without_noise",
                "command",
                "trajectory",
                "speed",
                "angle",
                "goal",
                "occ_semantic",
                "occ_depth",
            ]
            out = do_sample_occ(
                images,
                model,
                sampler,
                value_dict,
                num_rounds=opt.n_rounds,
                num_frames=opt.n_frames,
                num_cameras=6,
                force_uc_zero_embeddings=uc_keys,
                initial_cond_indices=[index for index in range(opt.n_conds)],
                pose_seq=pose_seq,
            )
            if isinstance(out, (tuple, list)):
                samples, samples_z, inputs = out
                inputs = rearrange(inputs, "v t c h w -> (v t) c h w")
                virtual_path = os.path.join(opt.save, "virtual")
                real_path = os.path.join(opt.save, "real")
                print("---------------------------------", virtual_path + "//" + str(sample_index))

                perform_save_locally(
                    real_path,
                    inputs,
                    "videos",
                    opt.dataset,
                    sample_index,
                    opt.n_frames,
                    opt.n_rounds,
                    save_names=frame_list,
                )
                perform_save_locally(
                    virtual_path,
                    samples,
                    "videos",
                    opt.dataset,
                    sample_index,
                    opt.n_frames,
                    opt.n_rounds,
                    save_names=frame_list,
                )  ## modify  -->n_frames

            else:
                raise TypeError

        if opt.rand_gen:
            sample_index += random.randint(1, dataset_length - 1)
        else:
            sample_index += 1
            if dataset_length <= sample_index + 1:
                sample_index = dataset_length - 1
                count_index = -1

        if count_index == -1:
            break

        count_index = count_index + 1
