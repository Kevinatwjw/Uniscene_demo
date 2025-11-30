import os
from os.path import join as smart_path_join

import cv2
import numba as nb
import numpy as np
import open3d
import torch
from pyquaternion import Quaternion


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0])  # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1])  # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2])  # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw_return(
    voxels,  # semantic occupancy predictions
    pred_pts,  # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,  # voxel coordinates of point cloud
    pt_label=None,  # label of point cloud
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
):
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size) + np.array(
        vox_origin, dtype=np.float32
    ).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        # (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
        (fov_grid_coords[:, 3] > 0)
        & (fov_grid_coords[:, 3] != 17)
    ]
    print(len(fov_voxels))

    # import pdb; pdb.set_trace()
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111,projection='3d')

    voxel_size = sum(voxel_size) / 3
    colors = np.array(
        [
            [255, 120, 50, 255],  # barrier              orange
            [255, 192, 203, 255],  # bicycle              pink
            [255, 255, 0, 255],  # bus                  yellow
            [0, 150, 245, 255],  # car                  blue
            [0, 255, 255, 255],  # construction_vehicle cyan
            [255, 127, 0, 255],  # motorcycle           dark orange
            [255, 0, 0, 255],  # pedestrian           red
            [255, 240, 150, 255],  # traffic_cone         light yellow
            [135, 60, 0, 255],  # trailer              brown
            [160, 32, 240, 255],  # truck                purple
            [255, 0, 255, 255],  # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red  # add
            [139, 137, 137, 255],
            [75, 0, 75, 255],  # sidewalk             dard purple
            [150, 240, 80, 255],  # terrain              light green
            [230, 230, 250, 255],  # manmade              white
            [0, 175, 0, 255],  # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
            [0, 0, 0, 0],  # 17 empty
            [0, 128, 0, 255],  # 18
            [128, 0, 0, 255],  # 19
            [0, 128, 128, 255],  # 20
            [0, 255, 0, 255],  # 21
        ]
    ).astype(np.uint8)
    # print(fov_voxels[:, 3])
    p_colors = colors[fov_voxels[:, 3].astype(np.uint8) - 1] / 255

    return fov_voxels, p_colors


def replace_occ_grid_with_bev(
    input_occ, bevlayout, driva_area_idx=11, bev_replace_idx=[1, 5, 6], occ_replace_new_idx=[18, 20, 21]
):
    # self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
    # occ road [11] drivable area
     
    # default ped_crossing->18; stop_line->19 ( ); roal_divider->20; lane_divider->21
    # default shape: input_occ: [200,200,16]; bevlayout: [18,200,200],  

    roal_divider_mask = bevlayout[5, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[6, :, :].astype(np.uint8)

    roal_divider_mask = cv2.dilate(roal_divider_mask, np.ones((3, 3), np.uint8))
    lane_divider_mask = cv2.dilate(lane_divider_mask, np.ones((3, 3), np.uint8))

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
                        # print(x,y,i)
                        # print(occupancy_data)
                        # max_11_index = np.argmax(occupancy_data == driva_area_idx)
                        max_11_index = np.where(occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ


def voxel2world(
    voxel, voxel_size=np.array([0.125, 0.125, 0.125]), pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])
):
    """
    voxel: [N, 3]
    """
    return voxel * voxel_size[None, :] + pc_range[:3][None, :]


def world2voxel(
    wolrd, voxel_size=np.array([0.125, 0.125, 0.125]), pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])
):
    """
    wolrd: [N, 3]
    """
    return (wolrd - pc_range[:3][None, :]) / voxel_size[None, :]


def load_occ_gt(occ_path, grid_size, item, unoccupied=0):
    #############################
    #  occ data
    #  [z y x cls] or [z y x vx vy vz cls]
    pcd = np.load(open(occ_path, "rb"), encoding="bytes", allow_pickle=True)
    pcd_label = pcd[..., -1:]
    pcd_label[pcd_label == 0] = 255
    pcd_np_cor = voxel2world(pcd[..., [0, 1, 2]] + 0.5)  # x y z

    l2e = Quaternion(item["lidar2ego_rotation"]).transformation_matrix
    l2e[:3, 3] = np.array(item["lidar2ego_translation"])
    # l2e = torch.from_numpy(l2e).cuda().float()
    l2e = torch.from_numpy(l2e).float()

    #############################
    # add for filter floaters
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcd_np_cor)
    colors = pcd_label.repeat(3, axis=1)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    pcd, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    inlier_xyz = np.array(pcd.points)  # (n, 3)
    pcd_label = np.array(pcd.colors)[:, 0:1]

    #############################
    # convert lidar to ego
    # xyz = torch.from_numpy(inlier_xyz).cuda().float()
    xyz = torch.from_numpy(inlier_xyz).float()
    xyz = l2e[:3, :3] @ xyz.t() + l2e[:3, 3:4]
    pcd_np_cor = xyz.t()
    # pcd_np_cor = pcd_np_cor.cpu().numpy()
    pcd_np_cor = pcd_np_cor.numpy()

    #############################
    # delete points beyond the range
    mask = (
        (pcd_np_cor[:, 0] > -40.0)
        & (pcd_np_cor[:, 0] < 40.0)
        & (pcd_np_cor[:, 1] > -40.0)
        & (pcd_np_cor[:, 1] < 40.0)
        & (pcd_np_cor[:, 2] > -1.0)
        & (pcd_np_cor[:, 2] < 5.4)
    )
    pcd_np_cor = pcd_np_cor[mask]
    pcd_label = pcd_label[mask]

    #############################
    # convert ego points to voxel
    pcd_np_cor = world2voxel(
        pcd_np_cor, voxel_size=np.array([0.4, 0.4, 0.4]), pc_range=np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])
    )

    # make sure the point is in the grid
    pcd_np_cor = np.clip(pcd_np_cor, np.array([0, 0, 0]), grid_size - 1)
    pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

    # 255: noise, 1-16 normal classes, 0 unoccupied
    pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
    pcd_np = pcd_np.astype(np.int64)
    processed_label = np.ones(grid_size, dtype=np.uint8) * unoccupied
    processed_label = nb_process_label(processed_label, pcd_np)

    noise_mask = processed_label == 255
    processed_label[noise_mask] = 0
    return processed_label


# u1: uint8, u8: uint16, i8: int64
@nb.jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)

    return processed_label


def load_occ_layout(layout_path):
    # load layout data
    layout = np.load(open(layout_path, "rb"), encoding="bytes", allow_pickle=True)
    layout = layout["bev_map"]
    return layout


def read_occ_oss(sample_token, occ_base_path, sample_item):
    lidar_token = os.listdir(smart_path_join(occ_base_path, sample_token))[0][:-4]
    occ_path = os.path.join(occ_base_path, sample_token, lidar_token + ".npy")
    occ = load_occ_gt(occ_path=occ_path, grid_size=np.array([200, 200, 16]), item=sample_item)
    return occ
