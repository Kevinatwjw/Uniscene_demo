import io
import json
import os
import sys
import torch
sys.path.append("../data/subsets")
import numpy as np
from common import BaseDataset
from PIL import Image
from tqdm import tqdm


def balance_with_actions(samples, increase_factor=5, exceptions=None):
    if exceptions is None:
        exceptions = [2, 3]
    sample_to_add = list()
    if increase_factor > 1:
        for each_sample in samples:
            if each_sample["cmd"] not in exceptions:
                for _ in range(increase_factor - 1):
                    sample_to_add.append(each_sample)
    return samples + sample_to_add


def resample_complete_samples(samples, increase_factor=5):
    sample_to_add = list()
    if increase_factor > 1:
        for each_sample in samples:
            if (
                each_sample["speed"]
                and each_sample["angle"]
                and each_sample["z"] > 0
                and 0 < each_sample["goal"][0] < 1600
                and 0 < each_sample["goal"][1] < 900
            ):
                for _ in range(increase_factor - 1):
                    sample_to_add.append(each_sample)
    return samples + sample_to_add


class NuScenesDataset(BaseDataset):
    def __init__(
        self,
        data_root="data/nuscenes",
        anno_file="annos/nuScenes.json",
        target_height=320,
        target_width=576,
        num_frames=25,
    ):
        super().__init__(data_root, anno_file, target_height, target_width, num_frames)
        print("nuScenes loaded:", len(self))
        self.samples = balance_with_actions(self.samples, increase_factor=5)
        print("nuScenes balanced:", len(self))
        self.samples = resample_complete_samples(self.samples, increase_factor=2)
        print("nuScenes resampled:", len(self))
        self.action_mod = 0

    def get_image_path(self, sample_dict, current_index):
        return os.path.join(self.data_root, sample_dict["frames"][current_index])

    def build_data_dict(self, image_seq, sample_dict):
        # log_cond_aug = self.log_cond_aug_dist.sample()
        # cond_aug = torch.exp(log_cond_aug)
        cond_aug = torch.tensor([0.0])
        data_dict = {
            "img_seq": torch.stack(image_seq),
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([9]),
            "cond_frames_without_noise": image_seq[0],
            "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
            "cond_aug": cond_aug,
        }
        if self.action_mod == 0:
            data_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
        elif self.action_mod == 1:
            data_dict["command"] = torch.tensor(sample_dict["cmd"])
        elif self.action_mod == 2:
            # scene might be empty
            if sample_dict["speed"]:
                data_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
            # scene might be empty
            if sample_dict["angle"]:
                data_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
        elif self.action_mod == 3:
            # point might be invalid
            if sample_dict["z"] > 0 and 0 < sample_dict["goal"][0] < 1600 and 0 < sample_dict["goal"][1] < 900:
                data_dict["goal"] = torch.tensor([sample_dict["goal"][0] / 1600, sample_dict["goal"][1] / 900])
        else:
            raise ValueError

        return data_dict

    def __getitem__(self, index):
        sample_dict = self.samples[index]
        self.action_mod = (self.action_mod + index) % 4
        image_seq = list()
        for i in range(self.num_frames):
            current_index = i
            img_path = self.get_image_path(sample_dict, current_index)
            image = self.preprocess_image(img_path)
            image_seq.append(image)
        return self.build_data_dict(image_seq, sample_dict)


class NuScenesDatasetMVTOSOCC(NuScenesDataset):
    def __init__(
        self,
        data_root="data/nuscenes",
        anno_file="annos/nuScenes_rings.json",
        target_height=320,
        target_width=576,
        num_frames=1,
        version="v1.0-trainval",
        num_cameras=6,
        occ_anno_file="annos/nuScenes_rings_occ_dict_all_complete_zys.json",
        occ_data_root="nksr-render/occ_render_map/train/",
        depth_path=["depth_cor", "depth_data"],
        semantic_path=["semantic_color", "semantic"],
    ):
        super().__init__(data_root, anno_file, target_height, target_width, num_frames)
        self.num_cameras = num_cameras
        self.cam_names = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
        ]
        self.cam_dict = {
            "CAM_FRONT_LEFT": 5,
            "CAM_FRONT": 0,
            "CAM_FRONT_RIGHT": 1,
            "CAM_BACK_RIGHT": 2,
            "CAM_BACK": 3,
            "CAM_BACK_LEFT": 4,
        }

        self.occ_data_root = occ_data_root
        self.depth_path = depth_path
        self.semantic_path = semantic_path
        with open(occ_anno_file, "r") as anno_json:
            self.occ_samples = json.load(anno_json)

    def get_image_path_tos(self, sample_dict, current_index, current_cam_name):
        image_paths = os.path.join(self.data_root, sample_dict["frames"][current_cam_name][current_index])
        occ_semantic_paths = os.path.join(
            self.occ_data_root,
            self.occ_samples[sample_dict["frames"][current_cam_name][current_index]],
            self.semantic_path[1] + ".npz",
        )
        occ_depth_paths = os.path.join(
            self.occ_data_root,
            self.occ_samples[sample_dict["frames"][current_cam_name][current_index]],
            self.depth_path[1] + ".npz",
        )

        return image_paths, occ_semantic_paths, occ_depth_paths

    def preprocess_image_tos(self, image_path, flag="image", current_cam_name=None):
        for attempt in range(10):
            try:
                with open(image_path, "rb") as file:
                    image_data = file.read()
                    break
            except:
                print("s3 faild, retry")

        if flag == "image":
            image = Image.open(io.BytesIO(image_data))
        elif flag == "semantic":
            # import pdb; pdb.set_trace()
            image = np.load(io.BytesIO(image_data), allow_pickle=True)[self.cam_dict[current_cam_name]]
            # sem = image
            image = Image.fromarray(np.array(image, dtype=np.uint8))
        elif flag == "depth":
            image = np.load(io.BytesIO(image_data), allow_pickle=True)[self.cam_dict[current_cam_name]]
            image = Image.fromarray(np.array(image, dtype=np.float32))

        ori_w, ori_h = image.size
        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = (ori_h - tmp_h) // 2
            bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))

        if flag == "semantic":
            image = image.resize((self.target_width, self.target_height), resample=Image.NEAREST)
        else:
            image = image.resize((self.target_width, self.target_height), resample=Image.LANCZOS)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if flag == "semantic":
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
            # if image.max()!=16:
            #     print(image_path, image.max(), sem.max() )

        elif flag == "depth":
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
            image = (image / 100.0) * 2.0 - 1.0
        else:
            image = self.img_preprocessor(image)
        return image

    def build_data_dict(self, image_seq, sample_dict, num_cameras, occ_semantic_seq=None, occ_depth_seq=None):
        cond_aug = torch.tensor([0.0])
        data_dict = {
            "img_seq": torch.stack(image_seq).permute(1, 0, 2, 3, 4),
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.zeros(num_cameras, 1) + 9,  # torch.tensor([9]),
            "cond_aug": cond_aug,
            "cond_frames_without_noise": image_seq[0],
            "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
            "occ_semantic": torch.stack(occ_semantic_seq).permute(1, 0, 2, 3, 4),
            "occ_depth": torch.stack(occ_depth_seq).permute(1, 0, 2, 3, 4),
        }
        if self.action_mod == 0:
            data_dict["trajectory"] = torch.tensor(
                sample_dict["traj"][2:]
            )  # repeat(torch.tensor(sample_dict["traj"][2:]), "n -> v n", v=num_cameras)
        elif self.action_mod == 1:
            data_dict["command"] = torch.tensor(
                sample_dict["cmd"]
            )  # repeat(torch.tensor(sample_dict["cmd"]), "n -> v n", v=num_cameras)
        elif self.action_mod == 2:
            # scene might be empty
            if sample_dict["speed"]:
                data_dict["speed"] = torch.tensor(
                    sample_dict["speed"][1:]
                )  
            # scene might be empty
            if sample_dict["angle"]:
                data_dict["angle"] = (
                    torch.tensor(sample_dict["angle"][1:]) / 780
                )  # repeat(torch.tensor(sample_dict["angle"][1:]), "n -> v n", v=num_cameras)
        elif self.action_mod == 3:
            # point might be invalid
            if sample_dict["z"] > 0 and 0 < sample_dict["goal"][0] < 1600 and 0 < sample_dict["goal"][1] < 900:
                data_dict["goal"] = torch.tensor([sample_dict["goal"][0] / 1600, sample_dict["goal"][1] / 900])
        else:
            raise ValueError
        return data_dict

    def __getitem__(self, index):
        sample_dict = self.samples[index]
        self.action_mod = (self.action_mod + index) % 4
        image_seq = list()
        occ_semantic_seq = list()
        occ_depth_seq = list()
        for i in range(self.num_frames):
            image_seq_mv = list()

            occ_semantic_seq_mv = list()
            occ_depth_seq_mv = list()

            for current_cam_name in self.cam_names:
                current_index = i
                image_paths, occ_semantic_paths, occ_depth_paths = self.get_image_path_tos(
                    sample_dict, current_index, current_cam_name
                )

                image = self.preprocess_image_tos(image_paths, flag="image")
                image_seq_mv.append(image)

                if occ_semantic_paths == "None" or occ_depth_paths == "None":
                    occ_semantic = torch.zeros(image.shape)
                    occ_depth = torch.zeros(image.shape)
                else:
                    occ_semantic = self.preprocess_image_tos(
                        occ_semantic_paths, flag="semantic", current_cam_name=current_cam_name
                    )
                    occ_depth = self.preprocess_image_tos(
                        occ_depth_paths, flag="depth", current_cam_name=current_cam_name
                    )
                occ_semantic_seq_mv.append(occ_semantic)
                occ_depth_seq_mv.append(occ_depth)

            image_seq_mv_stack = torch.stack(image_seq_mv)
            image_seq.append(image_seq_mv_stack)

            occ_semantic_seq_mv_stack = torch.stack(occ_semantic_seq_mv)
            occ_semantic_seq.append(occ_semantic_seq_mv_stack)
            occ_depth_seq_mv_stack = torch.stack(occ_depth_seq_mv)
            occ_depth_seq.append(occ_depth_seq_mv_stack)
        return self.build_data_dict(
            image_seq, sample_dict, len(self.cam_names), occ_semantic_seq=occ_semantic_seq, occ_depth_seq=occ_depth_seq
        )


if __name__ == "__main__":
    nus = NuScenesDatasetMVTOSOCC()
    for i in tqdm(range(165280)):  ## train: 165280  val:35364  ##keytrain:28130  keyval:6019
        aa = nus.__getitem__(i)
        print(i)
