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
    把体素网格的整数索引 (0~199, 0~199, 0~15) 转换成世界坐标中的体素中心点真实坐标
    例如：dims=[200,200,16], resolution=0.4 → 每个体素 0.4m×0.4m×0.4m
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
    mode=0,        # 0=直接画 occupancy；1=画 LiDARSeg 预测；2=画 GT 点云标签
    sem=False,
):
    """
    用于 Open3D 可视化的辅助函数：
    把 occupancy 网格 → 点云（每个有物的体素取中心点）→ 按类别上色
    最后返回点云坐标 + RGBA 颜色，方便后面 open3d.geometry.PointCloud() 使用
    """
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
    """
    把 BEV Map（矢量语义（ped_crossing、road_divider、lane_divider）
    强行写回 3D occupancy 网格的 drivable_surface (11) 那一层，
    用来生成更精细的“分层”标签（OpenOccupancy 标准做法）

    输入
    ----
    input_occ     : (200,200,16) uint8，原始 occupancy
    bevlayout     : (18,200,200) uint8，BEV 矢量地图（来自 nuScenes map）

    输出
    ----
    output_occ    : 同 shape，被替换后的新 occupancy
    """
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
    体素索引 → 世界坐标（体素中心）
    voxel: [N, 3]
    """
    return voxel * voxel_size[None, :] + pc_range[:3][None, :]


def world2voxel(
    wolrd, voxel_size=np.array([0.125, 0.125, 0.125]), pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])
):
    """
    世界坐标 → 体素索引（浮点，后面需要取 floor 或 round）
    wolrd: [N, 3]
    """
    return (wolrd - pc_range[:3][None, :]) / voxel_size[None, :]


def load_occ_gt(occ_path, grid_size, item, unoccupied=0):
    """
    读取你自己生成的 .npz/.npy 形式的 occupancy GT（体素级）
    并转换成 (200,200,16) 的 uint8 网格，兼容 OpenOccupancy 评估

    参数
    ----
    occ_path : .npz 文件路径（里面存的是稀疏体素 [x,y,z,cls]）
    grid_size: [200,200,16]
    item     : nuScenes sample 数据（包含 lidar2ego 外参）
    unoccupied: 空体素填 0（默认）

    返回
    ----
    processed_label: (200,200,16) uint8，值 0=empty, 1~16=官方类, 17=unknown（会被转成0）
    """
    #############################
    #  occ data
    #  [z y x cls] or [z y x vx vy vz cls]
    # ----------------修改：兼容 .npz (字典) 和 .npy (数组) 读取--------------------------
    # pcd = np.load(open(occ_path, "rb"), encoding="bytes", allow_pickle=True)
    
    try:
        # allow_pickle=True 是必须的
        raw_data = np.load(occ_path, allow_pickle=True)
        
        # 判断是否为 npz 压缩包 (包含 files 属性) 且包含 semantics 键
        if hasattr(raw_data, 'files') and 'semantics' in raw_data.files:
            pcd = raw_data['semantics']  # 解包取出数据
        else:
            pcd = raw_data  # 兼容旧逻辑，直接是数组
            
    except Exception as e:
        print(f"[Error] Failed to load {occ_path}: {e}")
        # 读取失败返回全0空网格，防止程序崩溃
        return np.zeros(grid_size, dtype=np.uint8)
    # ----------------修改：兼容 .npz (字典) 和 .npy (数组) 读取--------------------------
    
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
    """
    Numba 加速版：对同一个体素内可能出现多个类的情况，做“投票”取最多的类
    这是整个 pipeline 里最慢的一步，用 numba 能快 50~100 倍
    """
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
    """读取 nuScenes 的 BEV 矢量地图（.npz）"""
    # load layout data
    layout = np.load(open(layout_path, "rb"), encoding="bytes", allow_pickle=True)
    layout = layout["bev_map"]
    return layout


def read_occ_oss(sample_token, occ_base_path, sample_item):
    """
    兼容你自己存的目录结构：
    occ_base_path/
        {sample_token}/
            {lidar_token}.npy
    """
     # === [修改开始] 智能路径查找逻辑 ===
    # lidar_token = os.listdir(smart_path_join(occ_base_path, sample_token))[0][:-4]
    # occ_path = os.path.join(occ_base_path, sample_token, lidar_token + ".npy")

    target_dir = smart_path_join(occ_base_path, sample_token)
    
    # 1. 优先尝试标准名 labels.npz (这是我们 generate_occ.py 生成的新标准)
    std_path = os.path.join(target_dir, "labels.npz")
    
    if os.path.exists(std_path):
        occ_path = std_path
    else:
        # 2. 回退逻辑：尝试寻找旧版 .npy (如果有的话)
        # 原逻辑：假设文件夹下有一个 .npy 文件，取第一个
        if os.path.exists(target_dir):
            files = [f for f in os.listdir(target_dir) if f.endswith('.npy')]
            if len(files) > 0:
                occ_path = os.path.join(target_dir, files[0])
            else:
                # 目录存在但没文件，返回全0
                return np.zeros((200, 200, 16), dtype=np.uint8)
        else:
            # 目录不存在，返回全0
            return np.zeros((200, 200, 16), dtype=np.uint8)
     # === [修改结束] ===
    occ = load_occ_gt(occ_path=occ_path, grid_size=np.array([200, 200, 16]), item=sample_item)
    return occ
