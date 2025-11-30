# 文件路径: data_process/generate_occ_mini.py

import os
import pickle
from argparse import ArgumentParser
from copy import deepcopy
from os.path import join as smart_path_join

import torch
import numpy as np
import open3d as o3d
import yaml
import nksr  # 确保你已安装 nksr

# === ChamferDist 适配 ===
from chamferdist.chamfer import knn_points

from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

# === 自定义 Chamfer Forward 函数 (适配 chamferdist 库) ===
def custom_chamfer_forward(x, y):
    """
    使用 chamferdist 的 knn_points 实现双向索引查找
    x: [B, N, 3], y: [B, M, 3]
    Returns: d1, d2, idx1, idx2
    """
    # 1. Forward: Find nearest neighbor in y for each point in x
    out_xy = knn_points(x, y, K=1)
    d1 = out_xy.dists[..., 0]
    idx1 = out_xy.idx[..., 0]

    # 2. Backward: Find nearest neighbor in x for each point in y
    out_yx = knn_points(y, x, K=1)
    d2 = out_yx.dists[..., 0]
    idx2 = out_yx.idx[..., 0]

    return d1, d2, idx1.int(), idx2.int()

# === NKSR 辅助函数 ===
def nksr_mesh_normal(input_xyz, input_normal, detail_level=0.5, mise_iter=1, cpu_=False):
    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(
        input_xyz, input_normal, chunk_size=20.0, detail_level=detail_level
    )
    if cpu_:
        field.to_("cpu")
        reconstructor.network.to("cpu")
    mesh = field.extract_dual_mesh(mise_iter=mise_iter)
    return mesh

# === 预处理函数 ===
def preprocess_cloud(pcd, max_nn=20, normals=None):
    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()
    return cloud

def preprocess(pcd, config):
    return preprocess_cloud(pcd, config["max_nn"], normals=True)

def lidar_to_world_to_lidar(pc, lidar_calibrated_sensor, lidar_ego_pose, cam_calibrated_sensor, cam_ego_pose):
    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor["translation"]))
    pc.rotate(Quaternion(lidar_ego_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose["translation"]))
    pc.translate(-np.array(cam_ego_pose["translation"]))
    pc.rotate(Quaternion(cam_ego_pose["rotation"]).rotation_matrix.T)
    pc.translate(-np.array(cam_calibrated_sensor["translation"]))
    pc.rotate(Quaternion(cam_calibrated_sensor["rotation"]).rotation_matrix.T)
    return pc

# === 主处理逻辑 ===
def main(nusc, indice, nuscenesyaml, args, config):
    save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml["learning_map"]
    pc_range = config["pc_range"]
    occ_size = config["occ_size"]
    voxel_size = config["voxel_size"]

    my_scene = nusc.scene[indice]
    sensor = "LIDAR_TOP"
    
    # 获取第一个 sample
    first_sample_token = my_scene["first_sample_token"]
    my_sample = nusc.get("sample", first_sample_token)
    lidar_data = nusc.get("sample_data", my_sample["data"][sensor])
    lidar_ego_pose0 = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    lidar_calibrated_sensor0 = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    dict_list = []

    # 遍历序列收集点云
    while True:
        flag_has_lidarseg = True
        try:
            lidar_sd_token = lidar_data["token"]
            nusc.get("lidarseg", lidar_sd_token)
        except (KeyError, ValueError):
            flag_has_lidarseg = False

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data["token"])
        
        # 获取语义标签转换
        boxes_token = [box.token for box in boxes]
        object_category = [nusc.get("sample_annotation", box_token)["category_name"] for box_token in boxes_token]
        converted_object_category = []
        for category in object_category:
            for (j, label) in enumerate(nuscenesyaml["labels"]):
                if category == nuscenesyaml["labels"][label]:
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        # 获取 bbox 属性
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.0
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.0 - 0.1
        gt_bbox_3d[:, 3:6] *= 1.1

        # 读取点云
        pc0 = np.fromfile(os.path.join(data_root, lidar_data["filename"]), dtype=np.float32, count=-1).reshape(-1, 5)[..., :4]
        
        # 读取语义 (如果是关键帧)
        if lidar_data["is_key_frame"] and flag_has_lidarseg:
            lidar_sd_token = lidar_data["token"]
            lidarseg_labels_filename = os.path.join(nusc.dataroot, nusc.get("lidarseg", lidar_sd_token)["filename"])
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)
            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

        # 过滤动态物体点
        points_in_boxes = points_in_boxes_cpu(
            torch.from_numpy(pc0[:, :3][np.newaxis, :, :]), torch.from_numpy(gt_bbox_3d[np.newaxis, :])
        )
        object_points_list = []
        j = 0
        while j < points_in_boxes.shape[-1]:
            object_points_mask = points_in_boxes[0][:, j].bool()
            object_points_list.append(pc0[object_points_mask])
            j += 1

        moving_mask = torch.ones_like(points_in_boxes)
        points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
        points_mask = ~(points_in_boxes[0])

        # 过滤自身范围
        range_cfg = config["self_range"]
        oneself_mask = torch.from_numpy(
            (np.abs(pc0[:, 0]) > range_cfg[0]) | (np.abs(pc0[:, 1]) > range_cfg[1]) | (np.abs(pc0[:, 2]) > range_cfg[2])
        )
        points_mask = points_mask & oneself_mask
        pc = pc0[points_mask]

        # 坐标转换统一
        lidar_ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
        lidar_calibrated_sensor = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        lidar_pc = lidar_to_world_to_lidar(
            pc.copy(), lidar_calibrated_sensor.copy(), lidar_ego_pose.copy(), lidar_calibrated_sensor0, lidar_ego_pose0
        )

        data_dict = {
            "object_tokens": [nusc.get("sample_annotation", t)["instance_token"] for t in boxes_token],
            "object_points_list": object_points_list,
            "lidar_pc": lidar_pc.points,
            "lidar_ego_pose": lidar_ego_pose,
            "lidar_calibrated_sensor": lidar_calibrated_sensor,
            "lidar_token": lidar_data["token"],
            "sample_token": lidar_data["sample_token"],
            "is_key_frame": lidar_data["is_key_frame"],
            "has_lidarseg": flag_has_lidarseg,
            "gt_bbox_3d": gt_bbox_3d,
            "converted_object_category": converted_object_category
        }

        if lidar_data["is_key_frame"] and flag_has_lidarseg:
            pc_with_semantic = pc_with_semantic[points_mask]
            lidar_pc_with_semantic = lidar_to_world_to_lidar(
                pc_with_semantic.copy(),
                lidar_calibrated_sensor.copy(),
                lidar_ego_pose.copy(),
                lidar_calibrated_sensor0,
                lidar_ego_pose0,
            )
            data_dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points

        dict_list.append(data_dict)

        # 下一帧
        curr_sample_token = lidar_data["sample_token"]
        next_sample_token = nusc.get("sample", curr_sample_token)["next"]
        if next_sample_token != "":
            next_lidar_token = nusc.get("sample", next_sample_token)["data"][sensor]
            lidar_data = nusc.get("sample_data", next_lidar_token)
        else:
            break

    # === 合并点云 ===
    lidar_pc = np.concatenate([d["lidar_pc"] for d in dict_list], axis=1).T
    
    semantic_list = [d["lidar_pc_with_semantic"] for d in dict_list if d["is_key_frame"] and d["has_lidarseg"]]
    if semantic_list:
        lidar_pc_with_semantic = np.concatenate(semantic_list, axis=1).T
    else:
        print(f"Warning: No semantic data found for sequence {indice}")
        return

    # === NKSR 重建 ===
    point_cloud_original = o3d.geometry.PointCloud()
    with_normal = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(lidar_pc[:, :3]) # Use all static points
    with_normal = preprocess(point_cloud_original, config) # Estimate Normals

    point = np.asarray(with_normal.points)
    normal = np.asarray(with_normal.normals)
    point = torch.from_numpy(point).float().cuda()
    normal = torch.from_numpy(normal).float().cuda()

    # 使用 NKSR 重建表面 (High Quality)
    with torch.no_grad():
        nksr_mesh = nksr_mesh_normal(point, normal, detail_level=0.5, mise_iter=1, cpu_=False)
    
    scene_points = np.asarray(nksr_mesh.v.cpu(), dtype=float)

    # === 体素化 ===
    pcd_np = scene_points.copy()
    pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size[0]
    pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size[1]
    pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size[2]
    pcd_np = np.floor(pcd_np).astype(np.int32)
    
    # 过滤越界点
    mask = (pcd_np[:, 0] >= 0) & (pcd_np[:, 0] < occ_size[0]) & \
           (pcd_np[:, 1] >= 0) & (pcd_np[:, 1] < occ_size[1]) & \
           (pcd_np[:, 2] >= 0) & (pcd_np[:, 2] < occ_size[2])
    pcd_np = pcd_np[mask]
    
    # 生成 Dense Voxel Grid
    voxel = np.zeros(occ_size, dtype=np.uint8)
    voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

    # === 语义迁移 (Chamfer) ===
    # 1. 找到所有非空体素的中心坐标
    x, y, z = np.nonzero(voxel)
    fov_voxels = np.stack([x, y, z], axis=-1).astype(np.float32)
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * np.array(voxel_size)
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]

    # 2. 准备稀疏语义点
    sparse_semantic = lidar_pc_with_semantic
    mask_s = (sparse_semantic[:,0] > pc_range[0]) & (sparse_semantic[:,0] < pc_range[3]) & \
             (sparse_semantic[:,1] > pc_range[1]) & (sparse_semantic[:,1] < pc_range[4]) & \
             (sparse_semantic[:,2] > pc_range[2]) & (sparse_semantic[:,2] < pc_range[5])
    sparse_semantic = sparse_semantic[mask_s]

    # 3. 最近邻搜索
    x_tensor = torch.from_numpy(fov_voxels).cuda().unsqueeze(0).float()
    y_tensor = torch.from_numpy(sparse_semantic[:, :3]).cuda().unsqueeze(0).float()
    
    # 使用自定义 Chamfer
    _, _, idx1, _ = custom_chamfer_forward(x_tensor, y_tensor)
    indices = idx1[0].cpu().numpy()

    # 4. 赋值语义
    dense_semantic = sparse_semantic[:, 3][indices]
    
    # 5. 构建最终语义体素网格
    final_voxel_grid = np.zeros(occ_size, dtype=np.uint8) # 默认 0 (empty)
    # 将体素坐标重新映射回 grid
    grid_coords = ((fov_voxels - np.array(pc_range[:3])) / np.array(voxel_size)).astype(int)
    final_voxel_grid[grid_coords[:,0], grid_coords[:,1], grid_coords[:,2]] = dense_semantic

    # === 保存逻辑 (核心修改) ===
    # 遍历当前序列的所有关键帧进行保存
    for d in dict_list:
        if d["is_key_frame"]:
            # 构造路径: save_path/dense_voxels_with_semantic/<sample_token>/labels.npz
            target_dir = smart_path_join(save_path, "dense_voxels_with_semantic", d["sample_token"])
            os.makedirs(target_dir, exist_ok=True)
            
            save_file = smart_path_join(target_dir, "labels.npz")
            # 存为标准格式：key='semantics'
            np.savez_compressed(save_file, semantics=final_voxel_grid)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument("--dataset", type=str, default="nuscenes")
    parse.add_argument("--config_path", type=str, default="./data_process/config-200.yaml")
    parse.add_argument("--save_path", type=str, default="../data/gts/")
    parse.add_argument("--dataroot", type=str, default="../data/nuscenes/")
    parse.add_argument("--label_mapping", type=str, default="./nuscenes.yaml")
    parse.add_argument("--index_list", nargs="+", type=int)
    args = parse.parse_args()

    if args.dataset == 'nuscenes':
        nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=True)
    
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)
    with open(args.label_mapping, "r") as stream:
        nuscenesyaml = yaml.safe_load(stream)

    index_list = args.index_list
    for index in index_list:
        print(f"Processing sequence: {index}")
        try:
            main(nusc, index, nuscenesyaml, args, config)
        except Exception as e:
            print(f"Error in sequence {index}: {e}")