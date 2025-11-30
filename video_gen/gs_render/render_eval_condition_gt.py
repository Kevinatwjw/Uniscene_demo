import os
import pickle

import cv2
import numpy as np
import torch
from gaussian_renderer import apply_depth_colormap, apply_semantic_colormap, render
from pyquaternion import Quaternion

## 200 occ3d
cams = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]


def replace_occ_grid_with_bev(
    input_occ, bevlayout, driva_area_idx=11, bev_replace_idx=[1, 5, 6], occ_replace_new_idx=[17, 18, 19]
):
    # self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
    # occ road [11] drivable area

    # default ped_crossing->18; stop_line->19 (del); roal_divider->20; lane_divider->21
    # default shape: input_occ: [200,200,16]; bevlayout: [18,200,200]

    roal_divider_mask = bevlayout[5, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[6, :, :].astype(np.uint8)

    roal_divider_mask = cv2.erode(roal_divider_mask, np.ones((2, 2), np.uint8))
    lane_divider_mask = cv2.erode(lane_divider_mask, np.ones((2, 2), np.uint8))

    bevlayout[5, :, :] = roal_divider_mask.astype(bool)
    bevlayout[6, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ.shape[0], input_occ.shape[1]
    output_occ = input_occ.copy()  # numpy copy() ; tensor clone()
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)

    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ[x, y, :]

                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ


def load_occ_layout(layout_path):
    # load layout data
    layout = np.load(open(layout_path, "rb"), encoding="bytes", allow_pickle=True)
    layout = layout["bev_map"]
    return layout


def render_occ_semantic_map(item_data, base_path, occ_base_path, layout_base_path, is_vis=False):
    # dict_keys(['lidar_path', 'token', 'prev', 'next', 'can_bus', 'frame_idx', 'sweeps', 'cams', 'scene_token',
    #            'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation',
    #            'timestamp', 'occ_gt_path', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts',
    #            'valid_flag'])

    sample_token = item_data["token"]

    occ_path = os.path.join(occ_base_path, sample_token + ".npz")
    occ_label = load_occ_gt(occ_path=occ_path)
    occ_label[occ_label == 17] = 0  # 17 -> 0

    layout_path = os.path.join(layout_base_path, sample_token + ".npz")
    bevlayout = load_occ_layout(layout_path=layout_path)
    bevlayout = torch.from_numpy(bevlayout.astype(np.float64))
    bevlayout = torch.rot90(bevlayout, k=3, dims=(1, 2))
    bevlayout = bevlayout.numpy()

    semantics = occ_label
    semantics = replace_occ_grid_with_bev(semantics, bevlayout)

    image_shape = (450, 800)

    semantics = torch.from_numpy(semantics.astype(np.float32))  # 200, 200, 16
    xyz = (
        create_full_center_coords(shape=(200, 200, 16), x_range=(-40.0, 40.0), y_range=(-40.0, 40.0), z_range=(-1, 5.4))
        .view(-1, 3)
        .cuda()
        .float()
    )

    # load semantic data ------------------------------------------------------------------------------
    semantics_gt = semantics.view(-1, 1)  # (200, 200, 16) -> (640000, 16)
    occ_mask = semantics_gt[:, 0] != 0
    semantics_gt = semantics_gt.permute(1, 0)

    opacity = (semantics_gt.clone() != 0).float()
    opacity = opacity.permute(1, 0).cuda()

    semantics = torch.zeros((20, semantics_gt.shape[1])).cuda().float()
    color = torch.zeros((3, semantics_gt.shape[1])).cuda()
    for i in range(20):
        semantics[i] = semantics_gt == i

    rgb = color.permute(1, 0).float()
    feat = semantics.permute(1, 0).float()
    rot = torch.zeros((xyz.shape[0], 4)).cuda().float()
    rot[:, 0] = 1
    scale = torch.ones((xyz.shape[0], 3)).cuda().float() * 0.2

    camera_semantic = []
    camera_depth = []
    if not os.path.exists(os.path.join(base_path, sample_token)):
        os.makedirs(os.path.join(base_path, sample_token))
    sem_data_all_path = os.path.join(base_path, sample_token, "semantic.npz")
    depth_data_all_path = os.path.join(base_path, sample_token, "depth_data.npz")

    for cam in cams:
        cam_info = item_data["cams"][cam]
        camera_intrinsic = np.eye(3).astype(np.float32)
        camera_intrinsic[:3, :3] = cam_info["camera_intrinsics"]
        camera_intrinsic = torch.from_numpy(camera_intrinsic).cuda().float()

        c2e = Quaternion(cam_info["sensor2ego_rotation"]).transformation_matrix
        c2e[:3, 3] = np.array(cam_info["sensor2ego_translation"])
        c2e = torch.from_numpy(c2e).cuda().float()

        camera_extrinsic = c2e

        camera_intrinsic[0][0] = camera_intrinsic[0][0] / 2
        camera_intrinsic[1][1] = camera_intrinsic[1][1] / 2
        camera_intrinsic[0][2] = camera_intrinsic[0][2] / 2
        camera_intrinsic[1][2] = camera_intrinsic[1][2] / 2

        render_pkg = render(
            camera_extrinsic,
            camera_intrinsic,
            image_shape,
            xyz[occ_mask],
            rgb[occ_mask],
            feat[occ_mask],
            rot[occ_mask],
            scale[occ_mask],
            opacity[occ_mask],
            bg_color=[0, 0, 0],
        )

        render_pkg["render_color"]
        render_semantic = render_pkg["render_feat"]
        render_depth = render_pkg["render_depth"]
        render_pkg["render_alpha"]

        if is_vis:
            if not os.path.exists(os.path.join(base_path, sample_token, "semantic_color")):
                os.makedirs(os.path.join(base_path, sample_token, "semantic_color"))
            sem_save_path = os.path.join(base_path, sample_token, "semantic_color", cam + ".png")
            with open(sem_save_path, "wb") as f:
                data = apply_semantic_colormap(render_semantic).cpu().permute(1, 2, 0).detach().numpy() * 255
                f.write(cv2.imencode(".png", data)[1])

            if not os.path.exists(os.path.join(base_path, sample_token, "depth_color")):
                os.makedirs(os.path.join(base_path, sample_token, "depth_color"))
            depth_save_path = os.path.join(base_path, sample_token, "depth_color", cam + ".png")
            with open(depth_save_path, "wb") as f:
                render_depth = torch.clamp(render_depth, min=0.1, max=40.0)
                data = apply_depth_colormap(render_depth).cpu().permute(1, 2, 0).detach().numpy() * 255
                f.write(cv2.imencode(".png", data)[1])

        semantic = torch.max(render_semantic, dim=0)[1].squeeze().cpu().numpy().astype(np.int8)
        camera_semantic.append(semantic)

        depth_data = render_depth[0].detach().cpu().numpy()
        camera_depth.append(depth_data)

    ################################### update object to local ####################################
    np.savez(sem_data_all_path, camera_semantic)
    np.savez(depth_data_all_path, camera_depth)

    print(f"Rendered {sample_token} to {base_path}{sample_token}")


def load_occ_gt(occ_path: str):
    layout = np.load(open(occ_path, "rb"), encoding="bytes", allow_pickle=True)
    layout = layout["occ"]
    return layout


def create_full_center_coords(shape, x_range, y_range, z_range):
    x = torch.linspace(x_range[0], x_range[1], shape[0]).view(-1, 1, 1).expand(shape)
    y = torch.linspace(y_range[0], y_range[1], shape[1]).view(1, -1, 1).expand(shape)
    z = torch.linspace(z_range[0], z_range[1], shape[2]).view(1, 1, -1).expand(shape)

    center_coords = torch.stack((x, y, z), dim=-1)

    return center_coords


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--index_list", nargs="+", type=int, default=[0])
    parser.add_argument("--dataset_path", type=str, default="./data/nuScenes/")
    parser.add_argument("--occ_path", type=str, default="./data/occ_data/")
    parser.add_argument("--layout_path", type=str, default="./data/layout/")
    parser.add_argument("--render_path", type=str, default="./data/occ_render_map/")
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()

    # render train data
    train_pkl_path = os.path.join(args.dataset_path, "nuscenes_advanced_12Hz_infos_val.pkl")
    render_base_path = os.path.join(args.render_path, "val")
    occ_base_path = os.path.join(args.occ_path, "val")
    layout_base_path = os.path.join(args.layout_path, "val")

    data = pickle.load(open(train_pkl_path, "rb"))
    items_data = data["infos"]

    all_train_items = len(items_data)
    index_list = args.index_list

    for index in index_list:
        try:
            item = items_data[index]
            render_occ_semantic_map(
                item, render_base_path, occ_base_path=occ_base_path, layout_base_path=layout_base_path, is_vis=args.vis
            )

        except Exception as e:
            print(f"Error: {e}")
            with open("./error_list.txt", "a") as f:
                f.write(str(index) + "\n")
            continue

        with open("./success_list.txt", "a") as f:
            f.write(str(index) + "\n")
