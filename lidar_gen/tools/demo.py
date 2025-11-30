import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.nuscenes_occ.occ_lidar_gen import cartesian_to_spherical, load_occ_gt
from pcdet.models import build_network, load_data_to_gpu

CFG = "tools/cfgs/nuscenes_occ_models/occ2lidar_sparseunet_renderv2_s_priorsampler_p54_r200_intenw10_raydropw02_prerays.yaml"
CKPT = "checkpoints/occ2lidar.pth"
OCC_SIZE = [200, 200, 16]
POINT_CLOUD_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
VOXEL_SIZE = [0.5, 0.5, 0.5]
CLASS_NAMES = [
    "noise",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction",
    "motorcycle",
    "pedestrian",
    "trafficcone",
    "trailer",
    "truck",
    "driveable_surface",
    "other",
    "sidewalk",
    "terrain",
    "mannade",
    "vegetation",
]
N_CLASSES = len(CLASS_NAMES)
OCC_FILE = "asserts/occ_demo.npy"


def prepare_occ(occ_file):
    with open(occ_file, "rb") as f:
        occ = load_occ_gt(f, grid_size=np.array(OCC_SIZE))
        occ[occ == 255] = 0
    occ_loc = np.stack(occ.nonzero(), axis=-1)[:, [2, 1, 0]]
    occ = np.concatenate([occ_loc, occ[occ_loc[:, 2], occ_loc[:, 1], occ_loc[:, 0]][:, None]], axis=-1)

    # to xyz
    occ = occ[:, [2, 1, 0, 3]]

    data_dict = dict()
    # occ feature (x, y, z, theta, phi, r, cls)
    # to zyx for voxelization
    data_dict["occ"] = occ
    occ_range_mask = (
        (data_dict["occ"][:, 0] >= 0)
        & (data_dict["occ"][:, 0] < OCC_SIZE[0])
        & (data_dict["occ"][:, 1] >= 0)
        & (data_dict["occ"][:, 1] < OCC_SIZE[1])
        & (data_dict["occ"][:, 2] >= 0)
        & (data_dict["occ"][:, 2] < OCC_SIZE[2])
    )
    data_dict["occ"] = data_dict["occ"][occ_range_mask]
    data_dict["occ"] = data_dict["occ"][:, [2, 1, 0, 3]]

    voxel_size = np.array(VOXEL_SIZE).reshape((-1, 3))
    pc_range = np.array(POINT_CLOUD_RANGE[:3]).reshape((-1, 3))
    xyz = data_dict["occ"][:, [2, 1, 0]]
    occ_labels = data_dict["occ"][:, -1]
    xyz = (xyz + 0.5) * voxel_size + pc_range
    tpr = cartesian_to_spherical(xyz)
    cls_encoded = np.eye(N_CLASSES)[occ_labels]
    occ_feature = np.concatenate([xyz, tpr, cls_encoded], axis=-1)
    data_dict["occ"] = np.concatenate([data_dict["occ"], occ_feature], axis=-1)

    xyz = data_dict["occ"][:, [2, 1, 0]].astype(np.int32)
    data_dict["grid"] = np.zeros(OCC_SIZE, dtype=bool)
    data_dict["grid"][xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    data_dict["grid"] = torch.from_numpy(data_dict["grid"])

    data_dict["frame_id"] = "demo"
    return data_dict


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, occ_file, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.occ_file = occ_file

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return prepare_occ(self.occ_file)

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        ret = {}
        ret["batch_size"] = len(batch_list)
        for key, val in data_dict.items():
            if key in ["points", "occ", "did_return"]:
                coors = []
                if isinstance(val[0], list):
                    val = [i for item in val for i in item]
                for i, coor in enumerate(val):
                    if key == "did_return":
                        coor = coor[:, np.newaxis].astype(np.int32)
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ["points_in_occ", "grid"]:
                ret[key] = val
            elif key in ["frame_id", "end_flag", "tra"]:
                ret[key] = val
        return ret


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("-c", "--cfg_file", type=str, default=CFG, help="specify the config for training")
    parser.add_argument("--ckpt", type=str, default=CKPT, help="checkpoint to start from")
    parser.add_argument("--occ_file", type=str, default=OCC_FILE, help="occ file to start from")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    return args, cfg


def main():
    args, cfg = parse_config()

    test_set = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        occ_file=args.occ_file,
        training=False,
        logger=None,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logging.getLogger())
    model.cuda()
    batch_dict = test_set[0]
    batch_dict = test_set.collate_batch([batch_dict])
    load_data_to_gpu(batch_dict)
    model.eval()
    with torch.no_grad():
        pred_dicts, _ = model(batch_dict)
        pred_dicts = pred_dicts["pc_out"][0][:, :3].cpu().numpy()
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pred_dicts)
        o3d.io.write_point_cloud("asserts/pc_demo.ply", pc)


if __name__ == "__main__":
    main()
