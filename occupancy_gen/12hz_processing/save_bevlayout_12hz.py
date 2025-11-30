import argparse
import os
import pickle

import numpy as np
from mmdet3d.structures.bbox_3d import Box3DMode, LiDARInstance3DBoxes
from nuscenes.map_expansion.map_api import NuScenesMap
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class nuScenesSceneDatasetLidar_12hz_saveBEV(Dataset):
    def __init__(
        self,
        # occ_base_path,
        # return_len,
        # offset,
        imageset="train",
        nusc=None,
        nusc_dataroot=None,
        times=5,
        test_mode=False,
        input_dataset="gts",
        output_dataset="gts",
        s_p=0,
        e_p=165280,
    ):
        with open(imageset, "rb") as f:
            data = pickle.load(f)

        self.nusc_infos = data["infos"][s_p:e_p]
        # self.scene_names = list(self.nusc_infos.keys())
        # self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        # self.occ_base_path = occ_base_path
        # self.return_len = return_len
        # self.offset = offset
        # self.nusc = nusc

        # self.nusc = NuScenes(version='advanced_12Hz_trainval', dataroot=nusc_dataroot, verbose=True)
        self.maps = {}
        LOCATIONS = ["singapore-onenorth", "singapore-hollandvillage", "singapore-queenstown", "boston-seaport"]
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(nusc_dataroot, location)

        self.classes = [
            "drivable_area",
            "ped_crossing",
            "walkway",
            "stop_line",
            "carpark_area",
            "road_divider",
            "lane_divider",
            "road_block",
        ]
        self.object_classes = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ["gts", "tpv_dense", "tpv_sparse"]
        assert (
            output_dataset == "gts"
        ), f"only used for evaluation, output_dataset should be gts, but got {output_dataset}"
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

        # xbound=[-50,50,0.125]
        # ybound=[-50,50,0.125]
        xbound = [-40, 40, 0.4]
        ybound = [-40, 40, 0.4]
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.use_valid_flag = True

        self.lidar2canvas = np.array(
            [[canvas_h / patch_h, 0, canvas_h / 2], [0, canvas_w / patch_w, canvas_w / 2], [0, 0, 1]]
        )

        self.start_on_keyframe = True
        self.start_on_firstframe = False

        # self.clip_infos=self.build_clips(self.nusc_infos,data['scene_tokens'])

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.nusc_infos)

    def __getitem__(self, index):
        # index = index % len(self.nusc_infos)
        # scene_name = self.scene_names[index]
        # scene_len = self.scene_lens[index]
        # # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len

        metas = {}

        # for frame in clip:
        token = self.nusc_infos[index]["token"]
        # tokens.append(token)
        # occ = read_occ_HR(token,self.occ_base_path,self.quantize_size)
        # occs.append(occ)

        metas.update(self.get_meta_info(index))
        bevmap = self.get_map_info(metas, index)

        # bevmaps.append(bevmap)

        # metas.update(scene_token=tokens)
        # input_occs = np.stack(occs).astype(np.int64)

        # bevmaps= np.stack(bevmaps).astype(bool)

        # # train vqvae_4  for bev layout

        # for i in range(self.return_len + self.offset):
        # token = self.nusc_infos[scene_name][idx + i]['token']
        # bevmap.update(scene_token=token)
        # bevmap.update(self.get_map_info(metas,scene_name,idx+i) )

        return bevmap, token

    # def get_meta_data(self, scene_name, idx):
    #     gt_modes = []
    #     xys = []
    #     for i in range(self.return_len + self.offset):
    #         xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0]) #1*2
    #         gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
    #     xys = np.asarray(xys)
    #     gt_modes = np.asarray(gt_modes)
    #     return {'rel_poses': xys, 'gt_mode': gt_modes}

    def get_meta_info(self, idx):
        """Get annotation info according to the given index."""
        # T = 6
        # idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[idx]
        fut_valid_flag = info["valid_flag"]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.object_classes:
                gt_labels_3d.append(self.object_classes.index(cat))
            else:
                gt_labels_3d.append(-1)
                # print(f'Warning: {cat} not in CLASSES')
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            # attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,
        )

        return anns_results

    def _project_dynamic_bbox(self, dynamic_mask, data):
        """We use PIL for projection, while CVT use cv2.

        The results are slightly different due to anti-alias of line, but
        should be similar.
        """
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = data["gt_labels_3d"] == cls_id
            boxes = data["gt_bboxes_3d"][cls_mask]
            if len(boxes) < 1:
                continue
            # get coordinates on canvas. the order of points matters.
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]

            bottom_corners_canvas = np.dot(
                # np.pad(flipped_points, ((0, 0), (0, 0), (0, 1)),
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)), constant_values=1.0),
                self.lidar2canvas.T,
            )[
                ..., :2
            ]  # N, 4, xy
            # draw
            # Mod !!!
            points = bottom_corners_canvas
            centers = np.mean(points, axis=1, keepdims=True)
            points[:, :, 1] = 2 * centers[:, :, 1] - points[:, :, 1]
            bottom_corners_canvas = points

            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data):
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label

    # def get_image_info(self, scene_name, idx):
    #     T = 6
    #     idx = idx + self.return_len + self.offset - 1 - T
    #     info = self.nusc_infos[scene_name][idx]
    #     # import pdb; pdb.set_trace()
    #     input_dict = dict(
    #         sample_idx=info['token'],
    #         ego2global_translation = info['ego2global_translation'],
    #         ego2global_rotation = info['ego2global_rotation'],
    #     )
    #     f = 0.0055
    #     image_paths = []
    #     lidar2img_rts = []
    #     lidar2cam_rts = []
    #     cam_intrinsics = []
    #     cam_positions = []
    #     focal_positions = []

    #     lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    #     lidar2ego = np.eye(4)
    #     lidar2ego[:3, :3] = lidar2ego_r
    #     lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
    #     ego2lidar = np.linalg.inv(lidar2ego)
    #     for cam_type, cam_info in info['cams'].items():
    #         image_paths.append(cam_info['data_path'])
    #         # obtain lidar to image transformation matrix
    #         lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    #         lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    #         lidar2cam_rt = np.eye(4)
    #         lidar2cam_rt[:3, :3] = lidar2cam_r.T
    #         lidar2cam_rt[3, :3] = -lidar2cam_t
    #         intrinsic = cam_info['cam_intrinsic']
    #         viewpad = np.eye(4)
    #         viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    #         lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    #         lidar2img_rts.append(lidar2img_rt)
    #         cam_intrinsics.append(viewpad)
    #         lidar2cam_rts.append(lidar2cam_rt.T)
    #         cam_intrinsics.append(viewpad)
    #         lidar2cam_rts.append(lidar2cam_rt.T)
    #         # import pdb; pdb.set_trace()
    #         ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
    #         ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
    #         ego2cam_rt = np.eye(4)
    #         ego2cam_rt[:3, :3] = ego2cam_r.T
    #         ego2cam_rt[3, :3] = -ego2cam_t

    #         cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
    #         focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
    #         #cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
    #         cam_positions.append(cam_position.flatten()[:3])
    #         #focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
    #         focal_positions.append(focal_position.flatten()[:3])

    #     input_dict.update(
    #         dict(
    #             img_filename=image_paths,
    #             lidar2img=lidar2img_rts,
    #             cam_intrinsic=cam_intrinsics,
    #             lidar2cam=lidar2cam_rts,
    #             ego2lidar=ego2lidar,
    #             cam_positions=cam_positions,
    #             focal_positions=focal_positions,
    #             lidar2ego=lidar2ego,
    #         ))

    #     return input_dict

    def get_map_info(self, data, idx):

        info = self.nusc_infos[idx]
        ego2global_translation = info["ego2global_translation"]
        ego2global_rotation = info["ego2global_rotation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = np.array(ego2global_translation).T

        lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T
        np.linalg.inv(lidar2ego)

        lidar2global = ego2global @ lidar2ego

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        # cut semantics from nuscenesMap
        # sample_token=info["token"]
        # scene_token=self.nusc.field2token('scene', 'name', scene_name)
        # log_token= self.nusc.get('scene', scene_token[0])['log_token']
        # location = self.nusc.get('log', log_token)['location']
        # location = info["location"]
        location = self.nusc_infos[idx]["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool)

        # here we handle possible combinations of semantics
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        # data={}
        bevmap = {}
        if self.object_classes is not None:
            bevmap["gt_masks_bev_static"] = labels
            final_labels = self._project_dynamic(labels, data)
            # aux_labels = self._get_dynamic_aux(data)
            bevmap["gt_masks_bev"] = final_labels
            # data["gt_aux_bev"] = aux_labels
        else:
            bevmap["gt_masks_bev_static"] = labels
            bevmap["gt_masks_bev"] = labels

        # bevmap["gt_masks_bev"] = bevmap["gt_masks_bev"][:, ::-1, :].copy()

        return bevmap["gt_masks_bev"]


def main():
    parser = argparse.ArgumentParser(description="Quantize and resample labels.")
    parser.add_argument("--s_p", type=int, default=0)
    parser.add_argument("--e_p", type=int, default=10)
    parser.add_argument(
        "--imageset", type=str, default="./data/nuscenes_mmdet3d-12Hz/nuscenes_advanced_12Hz_infos_val.pkl"
    )
    args = parser.parse_args()

    imageset = args.imageset
    nusc_dataroot = "./data/nuscenes"
    tmp_save_root = "12hz_bevlayout_200_200"
    dataset = nuScenesSceneDatasetLidar_12hz_saveBEV(
        imageset=imageset, nusc_dataroot=nusc_dataroot, s_p=args.s_p, e_p=args.e_p
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(tmp_save_root, exist_ok=True)

    for i_iter, (bevmaps, tokens) in enumerate(tqdm(loader)):

        bz = len(bevmaps)
        # print(bevmaps.shape,len(tokens))
        for i in range(bz):

            # print(sum(bevmaps[i]))
            # np.save(f"{tmp_save_root}/{tokens[i]}.npy",bevmaps[i].numpy().astype(np.int8))
            # bevmap_read = np.load(f"{tmp_save_root}/{tokens[i]}.npy")

            np.savez_compressed(f"{tmp_save_root}/{tokens[i]}.npz", bev_map=bevmaps[i].numpy().astype(np.bool))

            # bevmap_read = np.load(f"{tmp_save_root}/{tokens[i]}.npz")["bev_map"]
            # map_img_np=visualize_map(bevmap_read)
            # map_img=Image.fromarray(map_img_np)
            # map_img.save(os.path.join('z_vis', f"{i_iter}_{i}.png"))
            # # print(f"{i_iter}_{i}.png")
            # print(f"npz_{i_iter}_{i}_{tokens[i]}")

        # print(bevmap.shape)


if __name__ == "__main__":
    main()
