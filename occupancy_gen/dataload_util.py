import os
import pickle

import numpy as np
import torch
from pyquaternion import Quaternion
from read_occ_nksr import read_occ_oss
from torch.utils.data import Dataset


def nBEV1(data_b, ch_use):  # 18,200,200 -> 1,200,200
    data_b = data_b[ch_use]
    mask = data_b > 0.01
    cumulative_mask = np.cumsum(mask, axis=0)
    max_index_map = np.argmax(cumulative_mask, axis=0)
    max_index_map = max_index_map / (len(ch_use) - 1)
    all_zero_mask = np.all(mask == 0, axis=0)
    max_index_map[all_zero_mask] = -1
    data_b = np.array([max_index_map])
    return data_b


def rot_flip(input):  # (C,H,W)
    # flip for the same xy oridinate like Zmid
    input = torch.rot90(input, k=1, dims=(1, 2))
    input = torch.flip(input, dims=[1])
    return input


def cal_occ_meta(data_occ, meta_num):
    # all
    valid_pts_num = np.sum(data_occ != 17)
    c_score = min(valid_pts_num / 50000, 1)

    terrain_pts = np.sum(data_occ == 14)
    c_terrain = min(terrain_pts / 7000, 1)

    manmade_pts = np.sum(data_occ == 15)
    c_manmade = min(manmade_pts / 16000, 1)

    vegetation_pts = np.sum(data_occ == 16)
    c_vegetation = min(vegetation_pts / 20000, 1)

    pts_other = valid_pts_num - terrain_pts - manmade_pts - vegetation_pts
    c_other = min(pts_other / 25000, 1)

    if meta_num == 1:
        # c_score= torch.tensor([c_score],dtype=torch.float)
        c_score = [c_score]
    elif meta_num == 4:
        # c_score= torch.tensor([c_score,c_terrain,c_manmade,c_vegetation],dtype=torch.float)
        # c_score=[c_score,c_terrain,c_manmade,c_vegetation]
        c_score = [c_other, c_terrain, c_manmade, c_vegetation]

    return c_score


class CustomDataset_Tframe_continuous(Dataset):
    def __init__(
        self,
        imageset,
        gts_path,
        folder_b,
        bev_ch_use=None,
        meta_num=1,
        Tframe=6,
        training=False,
        return_token=False,
        return_ori_bev=False,
    ):
        with open(imageset, "rb") as f:
            data = pickle.load(f)
        self.nusc_infos = data["infos"]
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.gts_path = gts_path
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.training = training
        self.return_token = return_token
        # with open("occ_token.pkl", 'rb') as f:
        #     self.token_dict=pickle.load(f)

    def __getitem__(self, index):

        scene_index = index % len(self.nusc_infos)
        scene_name = self.scene_names[scene_index]
        scene_len = self.scene_lens[scene_index]

        max_return_len = self.Tframe  # scene_len
        idx1 = np.random.randint(0, scene_len - max_return_len)
        idx_s = list(range(idx1, idx1 + max_return_len))

        if self.Tframe == 0:
            max_return_len = scene_len
            idx_s = list(range(0, max_return_len))

        occs = []
        tokens = []

        Bevs = []
        occ_metas = []
        Pose = []
        Bevs_ori = []
        for idx in idx_s:
            token = self.nusc_infos[scene_name][idx]["token"]
            tokens.append(token)
            label_file = os.path.join(self.gts_path, f"{scene_name}/{token}/labels.npz")
            label = np.load(label_file)
            occ = label["semantics"]
            occs.append(occ)

            Bev_file = os.path.join(self.Bev_root, f"{token}.npz")
            Bev = np.load(Bev_file)
            Bev = Bev["arr_0"]
            Bevs_ori.append(Bev)  # 18 200 200
            if self.ch_use:
                Bev = nBEV1(Bev, self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ, self.meta_num)
            occ_metas.append(c_score)

            info = self.nusc_infos[scene_name][idx]
            ego2global_translation = info["ego2global_translation"]
            ego2global_rotation = info["ego2global_rotation"]
            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T

            lidar2global = ego2global @ lidar2ego
            Pose.append(lidar2global)

        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_Bev_ori = np.stack(Bevs_ori)
        data_Bev_ori = torch.from_numpy(data_Bev_ori)
        # data_Bev_ori = torch.rot90(data_Bev_ori, k=1, dims=(2, 3))
        # data_Bev_ori = torch.flip(data_Bev_ori, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas = []
        for i in range(max_return_len - 1):
            Pose_1to2 = np.linalg.inv(Pose[i + 1]) @ Pose[i]
            Pose_t = Pose_1to2[:3, 3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t, Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas, dtype=torch.float)

        if self.return_token == False:
            return (
                data_occ,
                data_Bev,
                data_metas,
                Pose_metas,
            )
        return data_occ, data_Bev, data_metas, Pose_metas, tokens, data_Bev_ori

    def __len__(self):
        if self.training:
            return len(self.scene_names) * 40  # train 32 test 5
        else:
            return len(self.scene_names) * 5


class CustomDataset_wm_continuous(Dataset):
    def __init__(self, imageset, gts_path, folder_a, folder_b, bev_ch_use=None, meta_num=1, Tframe=6):
        with open(imageset, "rb") as f:
            data = pickle.load(f)
        self.nusc_infos = data["infos"]
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.gts_path = gts_path
        self.Zmid_root = folder_a
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        # with open("occ_token.pkl", 'rb') as f:
        #     self.token_dict=pickle.load(f)

    def __getitem__(self, index):

        scene_index = index % len(self.nusc_infos)
        scene_name = self.scene_names[scene_index]
        scene_len = self.scene_lens[scene_index]

        max_return_len = self.Tframe  # scene_len

        idx1 = np.random.randint(0, scene_len - max_return_len)
        # idx2 = idx1 + np.random.randint(1, max_return_len)

        # idx_s = [idx1,idx2]
        idx_s = list(range(idx1, idx1 + max_return_len))
        occs = []
        tokens = []

        Zmids = []
        Bevs = []
        occ_metas = []
        Pose = []
        for idx in idx_s:
            token = self.nusc_infos[scene_name][idx]["token"]
            tokens.append(token)
            label_file = os.path.join(self.gts_path, f"{scene_name}/{token}/labels.npz")
            label = np.load(label_file)
            occ = label["semantics"]
            occs.append(occ)

            # Zmid_file = os.path.join(self.Zmid_root, f"{token}.npz")
            # Zmid = np.load(Zmid_file)
            # Zmid = Zmid["arr_0"]
            # Zmids.append(Zmid)

            Bev_file = os.path.join(self.Bev_root, f"{token}.npz")
            Bev = np.load(Bev_file)
            Bev = Bev["arr_0"]
            if self.ch_use:
                Bev = nBEV1(Bev, self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ, self.meta_num)
            occ_metas.append(c_score)

            info = self.nusc_infos[scene_name][idx]
            ego2global_translation = info["ego2global_translation"]
            ego2global_rotation = info["ego2global_rotation"]
            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T

            lidar2global = ego2global @ lidar2ego
            Pose.append(lidar2global)

        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Z = None
        # data_Z = np.stack(Zmids)
        # data_Z = torch.from_numpy(data_Z)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=1, dims=(2, 3))
        data_Bev = torch.flip(data_Bev, dims=[2])

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas = []
        for i in range(max_return_len - 1):
            Pose_1to2 = np.linalg.inv(Pose[i + 1]) @ Pose[i]
            Pose_t = Pose_1to2[:3, 3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t, Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas, dtype=torch.float)

        return data_Z, data_Bev, data_occ, data_metas, Pose_metas

    def __len__(self):
        return len(self.scene_names) * 32
        # return len(self.scene_names) * 5


class CustomDataset_Tframe_12hz(Dataset):
    def __init__(
        self,
        imageset,
        occ_base_path,
        folder_b,
        bev_ch_use=None,
        meta_num=1,
        Tframe=8,
        training=False,
        use_clip=False,
        return_token=False,
    ):
        with open(imageset, "rb") as f:
            data = pickle.load(f)
        self.nusc_infos = data["infos"]
        # self.scene_names = list(self.nusc_infos.keys())
        # self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]

        self.occ_base_path = occ_base_path
        self.Bev_root = folder_b
        self.ch_use = bev_ch_use
        self.meta_num = meta_num
        self.Tframe = Tframe
        self.return_len = Tframe
        self.return_token = return_token
        # self.training = training
        # with open("occ_token.pkl", 'rb') as f:
        #     self.token_dict=pickle.load(f)
        self.start_on_keyframe = True
        self.start_on_firstframe = False

        self.use_clip = use_clip
        if use_clip:
            # self.clip_infos=self.build_clips(self.nusc_infos,data['scene_tokens'])
            self.clip_infos = self.build_clips_new(self.nusc_infos, data["scene_tokens"], Tframe)
        else:
            self.get_scene_len(self.nusc_infos, data["scene_tokens"])

    # def fliter_clips(self,clip):
    #     for frame in clip:
    #         token = self.nusc_infos[frame]['token']
    #         if os.path.exists(f"{self.occ_base_path}/{token}.npy")==False:
    #             return 0
    #     return 1

    def get_scene_len(self, data_infos, scene_tokens):
        self.token_data_dict = {item["token"]: idx for idx, item in enumerate(data_infos)}
        scene_lens = []
        scene_infos = []
        for scene in scene_tokens:
            scene_lens.append(len(scene))
            scene_token_idx = [self.token_data_dict[token] for token in scene]
            scene_infos.append(scene_token_idx)
        self.scene_lens = scene_lens
        self.scene_infos = scene_infos

    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {item["token"]: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for scene in scene_tokens:
            for start in range(len(scene) - self.return_len + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token] for token in scene[start : start + self.return_len]]
                # if self.fliter_clips(clip)==0:
                #     continue
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        # logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
        #              f"continuous scenes. Cut into {self.video_length}-clip, "
        #              f"which has {len(all_clips)} in total.")
        return all_clips

    def build_clips_new(self, data_infos, scene_tokens, Tframe):
        def generate_list(m, Tframe=5):
            result = []
            for i in range(0, m - Tframe + 1, Tframe):
                result.append(i)
            if m % Tframe != 0:
                result.append(m - Tframe)
            return result

        self.token_data_dict = {item["token"]: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        for scene in scene_tokens:
            m = len(scene)
            gen_start = generate_list(m, Tframe)
            # print(m,gen_start)
            for start in gen_start:
                clip = [self.token_data_dict[token] for token in scene[start : start + Tframe]]
                all_clips.append(clip)
        # logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
        #              f"continuous scenes. Cut into {self.video_length}-clip, "
        #              f"which has {len(all_clips)} in total.")
        return all_clips

    def __getitem__(self, index):

        max_return_len = self.Tframe
        if self.use_clip:
            clip = self.clip_infos[index]

        else:
            scene_index = index % len(self.scene_lens)
            scene_clip = self.scene_infos[scene_index]
            scene_len = self.scene_lens[scene_index]
            # scene_name = self.scene_names[scene_index]
            # scene_len = self.scene_lens[scene_index]
            # max_return_len=self.Tframe#scene_len

            if self.Tframe == 0:
                clip = scene_clip
                max_return_len = scene_len
            else:
                idx1 = np.random.randint(0, scene_len - self.Tframe)
                clip = scene_clip[idx1 : idx1 + self.Tframe]

        # print(clip)
        occs = []
        tokens = []

        Bevs = []
        occ_metas = []
        Pose = []
        Bevs_ori = []
        for frame in clip:
            token = self.nusc_infos[frame]["token"]
            tokens.append(token)
            # label_file = os.path.join(self.gts_path, f'{scene_name}/{token}/labels.npz')
            # label = np.load(label_file)
            # occ = label['semantics']
            # occ = np.load(f"{self.occ_base_path}/{token}.npy")
            occ = read_occ_oss(token, self.occ_base_path, self.nusc_infos[frame])
            occ[occ == 0] = 17
            # print(occ.shape)
            occs.append(occ)

            bevmap_path = os.path.join(self.Bev_root, token + ".npz")
            layout = np.load(open(bevmap_path, "rb"), encoding="bytes", allow_pickle=True)
            Bev = layout["bev_map"]

            # Bev_file =  os.path.join(self.Bev_root, f'{token}.npz')
            # Bev = np.load(Bev_file)
            # Bev = Bev['bev_map']#[:,::4,::4] # 18,200,200

            Bevs_ori.append(Bev)
            if self.ch_use:
                Bev = nBEV1(Bev, self.ch_use)
            Bevs.append(Bev)

            c_score = cal_occ_meta(occ, self.meta_num)
            occ_metas.append(c_score)

            info = self.nusc_infos[frame]
            ego2global_translation = info["ego2global_translation"]
            ego2global_rotation = info["ego2global_rotation"]
            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = np.array(ego2global_translation).T

            lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = lidar2ego_r
            lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T

            lidar2global = ego2global @ lidar2ego
            Pose.append(lidar2global)

        data_occ = np.stack(occs).astype(np.int64)
        data_occ = torch.from_numpy(data_occ)

        data_Bev = np.stack(Bevs).astype(np.float32)
        data_Bev = torch.from_numpy(data_Bev)
        data_Bev = torch.rot90(data_Bev, k=-1, dims=(2, 3))

        # data_Bev = torch.flip(data_Bev, dims=[1])

        data_Bev_ori = np.stack(Bevs_ori)
        data_Bev_ori = torch.from_numpy(data_Bev_ori)
        data_Bev_ori = torch.flip(data_Bev_ori, dims=[2])
        # data_Bev_ori = torch.rot90(data_Bev_ori, k=-1, dims=(2, 3))

        data_metas = np.stack(occ_metas).astype(np.float32)
        data_metas = torch.from_numpy(data_metas)

        Pose_metas = []
        for i in range(max_return_len - 1):
            Pose_1to2 = np.linalg.inv(Pose[i + 1]) @ Pose[i]
            Pose_t = Pose_1to2[:3, 3]
            Pose_r = Pose_1to2[:3, :3].flatten()
            Pose_meta = np.concatenate((Pose_t, Pose_r))
            Pose_metas.append(Pose_meta)
        Pose_metas = np.stack(Pose_metas).astype(np.float32)
        Pose_metas = torch.tensor(Pose_metas, dtype=torch.float)

        if self.return_token == False:
            return data_occ, data_Bev, data_metas, Pose_metas
        return data_occ, data_Bev, data_metas, Pose_metas, tokens, data_Bev_ori

    def __len__(self):
        if self.use_clip:
            return len(self.clip_infos)
        else:
            return len(self.scene_lens)  # training
