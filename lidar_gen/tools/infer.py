import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, Occ2LiDARDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def cartesian_to_spherical(coords):
   
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

 
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

  
    theta = np.arctan2(y, x)
 
    phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)

 
    return np.stack((theta, phi, r), axis=-1)


class DemoDataset(Occ2LiDARDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        DatasetTemplate.__init__(
            self,
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        occ_root = dataset_cfg.occ_root
        all_files = dataset_cfg.all_files
        self.class_names = class_names
        self.point_cloud_range = dataset_cfg.POINT_CLOUD_RANGE
        self.occ_size = dataset_cfg.GRID_SIZE
        self.occ_root = occ_root

        with open(all_files, "r") as split_json:
            self.full_list = json.load(split_json)

        self.length = len(self.full_list)

        self.scale_xyz = self.occ_size[0] * self.occ_size[1] * self.occ_size[2]
        self.scale_yz = self.occ_size[1] * self.occ_size[2]
        self.scale_z = self.occ_size[2]

    def __getitem__(self, idx):
        input_dict = {}
        occ_filename = self.full_list[str(idx)]
        input_dict["frame_id"] = "-".join(occ_filename.split("/"))
        occ_filepath = self.occ_root + "/" + occ_filename.split("/")[0] + ".npz"
        if not os.path.exists(occ_filepath):
            return self.__getitem__(np.random.randint(0, len(self)))

        occ = np.load(occ_filepath)
        occ_loc = np.stack(occ.nonzero(), axis=-1)[:, [2, 1, 0]]
        occ = np.concatenate([occ_loc, occ[occ_loc[:, 2], occ_loc[:, 1], occ_loc[:, 0]][:, None]], axis=-1)

        # to xyz(absolute coords) for data augmentor
        input_dict["occ"] = occ[:, [2, 1, 0, 3]].astype("float32")
        voxel_size = np.array(self.voxel_size).reshape((-1, 3))
        pc_range = np.array(self.point_cloud_range[:3]).reshape((-1, 3))
        input_dict["occ"][:, :3] = (input_dict["occ"][:, :3] + 0.5) * voxel_size + pc_range

        data_dict = self.prepare_data(data_dict=input_dict)

        # occ feature (x, y, z, theta, phi, r, cls)
        # to zyx for voxelization
        data_dict["occ"][:, :3] = (data_dict["occ"][:, :3] - pc_range) / voxel_size
        data_dict["occ"] = data_dict["occ"].astype(occ.dtype)
        occ_range_mask = (
            (data_dict["occ"][:, 0] >= 0)
            & (data_dict["occ"][:, 0] < self.grid_size[0])
            & (data_dict["occ"][:, 1] >= 0)
            & (data_dict["occ"][:, 1] < self.grid_size[1])
            & (data_dict["occ"][:, 2] >= 0)
            & (data_dict["occ"][:, 2] < self.grid_size[2])
        )
        data_dict["occ"] = data_dict["occ"][occ_range_mask]
        data_dict["occ"] = data_dict["occ"][:, [2, 1, 0, 3]]

        xyz = data_dict["occ"][:, [2, 1, 0]]
        occ_labels = data_dict["occ"][:, -1]
        xyz = (xyz + 0.5) * voxel_size + pc_range
        tpr = cartesian_to_spherical(xyz)
        cls_encoded = np.eye(len(self.class_names))[occ_labels]
        occ_feature = np.concatenate([xyz, tpr, cls_encoded], axis=-1)
        data_dict["occ"] = np.concatenate([data_dict["occ"], occ_feature], axis=-1)

        xyz = data_dict["occ"][:, [2, 1, 0]].astype(np.int32)
        data_dict["grid"] = np.zeros(self.occ_size, dtype=bool)
        data_dict["grid"][xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
        data_dict["grid"] = torch.from_numpy(data_dict["grid"])

        return data_dict


CFG = "tools/cfgs/nuscenes_occ_models/occ2lidar_sparseunet_renderv2_s_nopriorsampler_p36_r200_intenw10_raydropw02.yaml"
CKPT = "exps/occ2lidar_sparseunet_renderv2_s_priorsampler_p54_r200_intenw10_raydropw02/ckpt/checkpoint_epoch_20.pth"


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--cfg_file", type=str, default=CFG, help="specify the config for demo")
    parser.add_argument(
        "--data_path", type=str, default="demo_data", help="specify the point cloud data file or directory"
    )
    parser.add_argument("--ckpt", type=str, default=CKPT, help="specify the pretrained model")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        logger=logger,
    )
    logger.info(f"Total number of samples: \t{len(demo_dataset)}")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)


if __name__ == "__main__":
    main()
