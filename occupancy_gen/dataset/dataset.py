import os
import pickle

import numpy as np
from mmdet3d.structures.bbox_3d import Box3DMode, LiDARInstance3DBoxes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

from . import OPENOCC_DATASET


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar_ori:
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset="train",
        nusc=None,
        nusc_dataroot=None,
        times=5,
        test_mode=False,
        input_dataset="gts",
        output_dataset="gts",
    ):
        with open(imageset, "rb") as f:
            data = pickle.load(f)

        self.nusc_infos = data["infos"]
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        # self.nusc = nusc

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

        xbound = [-40, 40, 0.4]
        ybound = [-40, 40, 0.4]
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        # self.use_valid_flag=True

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.nusc_infos) * self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        # idx=0
        # self.return_len=scene_len

        occs = []
        tokens = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]["token"]
            tokens.append(token)
        # ==================== 修改开始：Input 读取逻辑 ===================
        #     label_file = os.path.join(self.data_path, f"{self.input_dataset}/{scene_name}/{token}/labels.npz")
        #     label = np.load(label_file)
        #     occ = label["semantics"]
        #     occs.append(occ)
        # # input_occs = np.stack(occs, dtype=np.int64)
        # input_occs = np.stack(occs).astype(np.int64)
        
            # 定义候选路径列表 (优先级：自制Mini结构 -> 官方Occ3D结构)
            candidate_paths = [
                # 1. 自制结构: data/gts/dense_voxels_with_semantic/<token>/labels.npz
                os.path.join(self.data_path, self.input_dataset, "dense_voxels_with_semantic", token, "labels.npz"),
                # 2. 官方结构: data/gts/<scene_name>/<token>/labels.npz
                os.path.join(self.data_path, self.input_dataset, scene_name, token, "labels.npz")
            ]

            label_file = None
            for path in candidate_paths:
                if os.path.exists(path):
                    label_file = path
                    break
            
            try:
                if label_file:
                    # allow_pickle=True 是必须的
                    label = np.load(label_file, allow_pickle=True)
                    occ = label["semantics"]
                else:
                    raise FileNotFoundError
            except Exception:
                # 容错：如果找不到或读取失败，返回全0空网格
                # print(f"[Warn] Input GT Missing for token {token}")
                occ = np.zeros((200, 200, 16), dtype=np.uint8)

            occs.append(occ)
        
        input_occs = np.stack(occs).astype(np.int64)
        # ==================== 修改结束：Input 读取逻辑 ====================
        
        
        
        
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]["token"]
        # ==================== 修改开始：Output 读取逻辑 ====================
            
            
        #     label_file = os.path.join(self.data_path, f"{self.output_dataset}/{scene_name}/{token}/labels.npz")
        #     label = np.load(label_file)
        #     occ = label["semantics"]
        #     occs.append(occ)
        # # output_occs = np.stack(occs, dtype=np.int64)
        # output_occs = np.stack(occs).astype(np.int64)
                    # 定义候选路径列表 (注意换成 self.output_dataset)
            candidate_paths = [
                # 自制: data/gts/dense_voxels_with_semantic/<token>/labels.npz
                os.path.join(self.data_path, self.output_dataset, "dense_voxels_with_semantic", token, "labels.npz"),
                # 官方: data/gts/<scene_name>/<token>/labels.npz
                os.path.join(self.data_path, self.output_dataset, scene_name, token, "labels.npz")
            ]

            label_file = None
            for path in candidate_paths:
                if os.path.exists(path):
                    label_file = path
                    break
            
            try:
                if label_file:
                    label = np.load(label_file, allow_pickle=True)
                    occ = label["semantics"]
                else:
                    raise FileNotFoundError
            except Exception:
                occ = np.zeros((200, 200, 16), dtype=np.uint8)

            occs.append(occ)
            
        output_occs = np.stack(occs).astype(np.int64)
        # ==================== 修改结束：Output 读取逻辑 ====================
        
        metas = {}
        metas.update(scene_token=tokens)
        metas.update(scene_name=scene_name)
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name, idx))
        # metas.update(self.get_meta_data(scene_name, idx))
        # metas.update(self.get_meta_info(scene_name, idx))

        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))

        return input_occs[: self.return_len], output_occs[self.offset :], metas

    def get_meta_data(self, scene_name, idx):
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx + i]["gt_ego_fut_trajs"][0])  # 1*2
            gt_modes.append(self.nusc_infos[scene_name][idx + i]["pose_mode"])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {"rel_poses": xys, "gt_mode": gt_modes}

    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info["token"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info["cams"].items():
            image_paths.append(cam_info["data_path"])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
            lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info["cam_intrinsic"]
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix)
            ego2cam_t = cam_info["sensor2ego_translation"] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t

            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0.0, 0.0, 0.0, 1.0]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0.0, 0.0, f, 1.0]).reshape([4, 1])
            # cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            # focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            )
        )

        return input_dict


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar:
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset="train",
        nusc=None,
        nusc_dataroot=None,
        times=5,
        test_mode=False,
        input_dataset="gts",
        output_dataset="gts",
    ):
        with open(imageset, "rb") as f:
            data = pickle.load(f)

        self.nusc_infos = data["infos"]
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        # self.nusc = nusc

        # self.nusc = NuScenes(version="v1.0-trainval", dataroot=nusc_dataroot, verbose=True)
        # [修改] 优先使用传入的 nusc 对象（如果已初始化），否则强制使用 v1.0-mini 初始化
        # 注意：如果传入的是字典配置（来自 __init__.py），这里最好忽略或重新初始化，
        # 为了适配 Mini 数据集，最稳妥的方式是确保这里使用 v1.0-mini。
        if nusc is not None and not isinstance(nusc, dict):
            self.nusc = nusc
        else:
            self.nusc = NuScenes(version="v1.0-mini", dataroot=nusc_dataroot, verbose=True)
            
            
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

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.nusc_infos)  # *self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        # idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        idx = 0
        self.return_len = scene_len

        occs = []
        tokens = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]["token"]
            tokens.append(token)
            # ==================== 修改开始：Input 读取逻辑 ====================
        #     label_file = os.path.join(self.data_path, f"{self.input_dataset}/{scene_name}/{token}/labels.npz")
        #     label = np.load(label_file)
        #     occ = label["semantics"]
        #     occs.append(occ)
        # # input_occs = np.stack(occs, dtype=np.int64)
        # input_occs = np.stack(occs).astype(np.int64)
            # 定义候选路径列表
            candidate_paths = [
                # 1. 自制结构
                os.path.join(self.data_path, self.input_dataset, "dense_voxels_with_semantic", token, "labels.npz"),
                # 2. 官方结构
                os.path.join(self.data_path, self.input_dataset, scene_name, token, "labels.npz")
            ]

            label_file = None
            for path in candidate_paths:
                if os.path.exists(path):
                    label_file = path
                    break
            
            try:
                if label_file:
                    label = np.load(label_file, allow_pickle=True)
                    occ = label["semantics"]
                else:
                    raise FileNotFoundError
            except Exception:
                occ = np.zeros((200, 200, 16), dtype=np.uint8)

            occs.append(occ)
            
        input_occs = np.stack(occs).astype(np.int64)        
        # ==================== 修改结束：Input 读取逻辑 ====================
        
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]["token"]
         # ==================== 修改开始：Output 读取逻辑 ====================    
        #     label_file = os.path.join(self.data_path, f"{self.output_dataset}/{scene_name}/{token}/labels.npz")
        #     label = np.load(label_file)
        #     occ = label["semantics"]
        #     occs.append(occ)
        # # output_occs = np.stack(occs, dtype=np.int64)
        # output_occs = np.stack(occs).astype(np.int64)
            # 定义候选路径列表 (注意换成 self.output_dataset)
            candidate_paths = [
                os.path.join(self.data_path, self.output_dataset, "dense_voxels_with_semantic", token, "labels.npz"),
                os.path.join(self.data_path, self.output_dataset, scene_name, token, "labels.npz")
            ]

            label_file = None
            for path in candidate_paths:
                if os.path.exists(path):
                    label_file = path
                    break
            
            try:
                if label_file:
                    label = np.load(label_file, allow_pickle=True)
                    occ = label["semantics"]
                else:
                    raise FileNotFoundError
            except Exception:
                occ = np.zeros((200, 200, 16), dtype=np.uint8)

            occs.append(occ)
            
        output_occs = np.stack(occs).astype(np.int64)
        # ==================== 修改结束：Output 读取逻辑 ====================        
        
        metas = {}
        metas.update(scene_token=tokens)
        metas.update(scene_name=scene_name)
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name, idx))
        # metas.update(self.get_meta_data(scene_name, idx))
        # metas.update(self.get_meta_info(scene_name, idx))

        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))

        # # train vqvae_4  for bev layout
        bevmaps = []

        for i in range(self.return_len + self.offset):
            # token = self.nusc_infos[scene_name][idx + i]['token']
            # bevmap.update(scene_token=token)
            # bevmap.update(self.get_map_info(metas,scene_name,idx+i) )
            metas.update(self.get_meta_info(scene_name, idx + i))
            bevmap = self.get_map_info(metas, scene_name, idx + i)
            bevmaps.append(bevmap)
        bevmaps = np.stack(bevmaps).astype(bool)

        return input_occs[: self.return_len], output_occs[self.offset :], metas, bevmaps

    def get_meta_data(self, scene_name, idx):
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx + i]["gt_ego_fut_trajs"][0])  # 1*2
            gt_modes.append(self.nusc_infos[scene_name][idx + i]["pose_mode"])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {"rel_poses": xys, "gt_mode": gt_modes}

    def get_meta_info(self, scene_name, idx):
        """Get annotation info according to the given index."""
        # T = 6
        # idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
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

        # if self.with_attr:
        #     gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
        #     gt_fut_masks = info['gt_agent_fut_masks'][mask]
        #     gt_fut_goal = info['gt_agent_fut_goal'][mask]
        #     gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
        #     gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
        #     attr_labels = np.concatenate(
        #         [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
        #     ).astype(np.float32)

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

    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        # import pdb; pdb.set_trace()
        input_dict = dict(
            sample_idx=info["token"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"]).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info["cams"].items():
            image_paths.append(cam_info["data_path"])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
            lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info["cam_intrinsic"]
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            # import pdb; pdb.set_trace()
            ego2cam_r = np.linalg.inv(Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix)
            ego2cam_t = cam_info["sensor2ego_translation"] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t

            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0.0, 0.0, 0.0, 1.0]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0.0, 0.0, f, 1.0]).reshape([4, 1])
            # cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            # focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            )
        )

        return input_dict

    def get_map_info(self, data, scene_name, idx):

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
        scene_token = self.nusc.field2token("scene", "name", scene_name)
        log_token = self.nusc.get("scene", scene_token[0])["log_token"]
        location = self.nusc.get("log", log_token)["location"]

        # location = info["location"]
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

        bevmap["gt_masks_bev"] = bevmap["gt_masks_bev"][:, ::-1, :].copy()

        return bevmap["gt_masks_bev"]
