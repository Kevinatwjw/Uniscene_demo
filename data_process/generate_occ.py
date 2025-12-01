import os
import pickle
from argparse import ArgumentParser
from copy import deepcopy
from os.path import join as smart_path_join

# import chamfer
import chamferdist as chamfer
from chamferdist.chamfer import knn_points

import nksr
import numpy as np
import open3d as o3d
import torch
import yaml
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

# === 添加这个适配函数 ===
def custom_chamfer_forward(x, y):
    """
    使用 chamferdist 的 knn_points 实现双向 Chamfer 计算
    x: [B, N, 3]
    y: [B, M, 3]
    Returns: d1, d2, idx1, idx2
    """
    # 1. 计算 x 到 y 的最近邻 (Forward)
    # knn_points 返回的是 namedtuple(dists, idx, knn)
    # dists 形状为 [B, N, K], 这里 K=1
    out_xy = knn_points(x, y, K=1)
    d1 = out_xy.dists[..., 0]  # 取第0个邻居的距离, 形状 [B, N]
    idx1 = out_xy.idx[..., 0]  # 取第0个邻居的索引, 形状 [B, N]

    # 2. 计算 y 到 x 的最近邻 (Backward)
    out_yx = knn_points(y, x, K=1)
    d2 = out_yx.dists[..., 0]  # [B, M]
    idx2 = out_yx.idx[..., 0]  # [B, M]

    # 这里的 indices 已经是 Long/Int 类型，无需转换
    # chamferdist 默认返回的就是平方距离 (squared L2)，符合通常习惯
    return d1, d2, idx1.int(), idx2.int()
# ========================

def nksr_mesh_normal(input_xyz, input_normal, detail_level=0.5, mise_iter=1, cpu_=False):
    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)
    # reconstructor.chunk_tmp_device = torch.device("cuda:0")

    field = reconstructor.reconstruct(
        input_xyz, input_normal, chunk_size=20.0, detail_level=detail_level  # This could be smaller
    )

    if cpu_:
        # Put everything onto CPU.
        field.to_("cpu")
        reconstructor.network.to("cpu")

    mesh = field.extract_dual_mesh(mise_iter=mise_iter)
    return mesh


def nksr_mesh_sensor(input_xyz, input_sensor):
    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(
        input_xyz,
        sensor=input_sensor,
        chunk_size=50.0,  # This could be smaller
        preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0),
    )
    # Put everything onto CPU.
    field.to_("cuda")
    reconstructor.network.to("cpu")
    # [WARNING] Slow operation...
    mesh = field.extract_dual_mesh(mise_iter=1)
    return mesh


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities


def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original=None):
    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)


def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):
    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(pcd, config["max_nn"], normals=True)


def nn_correspondance(verts1, verts2):
    """For each vertex in verts2 find the nearest vertex in verts1.

    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


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


def main(nusc, indice, nuscenesyaml, args, config):
    save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml["learning_map"]
    voxel_size = config["voxel_size"]
    pc_range = config["pc_range"]
    occ_size = config["occ_size"]
    # ----------修改1-------------
    my_scene = nusc.scene[indice]
    scene_name = my_scene['name'] # [关键新增] 获取场景名
    sensor = "LIDAR_TOP"
    # ----------修改1-------------
    

    # if args.split == 'train':
    #     if my_scene['token'] in val_list:
    #         return
    # elif args.split == 'val':
    #     if my_scene['token'] not in val_list:
    #         return
    # elif args.split == 'all':
    #     pass
    # else:
    #     raise NotImplementedError

    # load the first sample to start
    first_sample_token = my_scene["first_sample_token"]
    my_sample = nusc.get("sample", first_sample_token)
    lidar_data = nusc.get("sample_data", my_sample["data"][sensor])
    lidar_ego_pose0 = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    lidar_calibrated_sensor0 = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    # collect LiDAR sequence
    dict_list = []

    while True:
        ############################# Init: is has lidarseg labels ##########################
        flag_has_lidarseg = True
        try:
            lidar_sd_token = lidar_data["token"]
            nusc.get("lidarseg", lidar_sd_token)
        except KeyError:
            flag_has_lidarseg = False

        ############################# get boxes ##########################
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data["token"])
        boxes_token = [box.token for box in boxes]
        object_tokens = [nusc.get("sample_annotation", box_token)["instance_token"] for box_token in boxes_token]
        object_category = [nusc.get("sample_annotation", box_token)["category_name"] for box_token in boxes_token]

        ############################# get object categories ##########################
        converted_object_category = []
        for category in object_category:
            for (j, label) in enumerate(nuscenesyaml["labels"]):
                if category == nuscenesyaml["labels"][label]:
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.0
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.0
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1  # Slightly expand the bbox to wrap all object points
        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data["filename"]  # load LiDAR names
        pc0 = np.fromfile(os.path.join(data_root, pc_file_name), dtype=np.float32, count=-1).reshape(-1, 5)[..., :4]
        if lidar_data["is_key_frame"] and flag_has_lidarseg:  # only key frame has semantic annotations
            lidar_sd_token = lidar_data["token"]
            lidarseg_labels_filename = os.path.join(nusc.dataroot, nusc.get("lidarseg", lidar_sd_token)["filename"])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

        ############################# cut out movable object points and masks ##########################
        points_in_boxes = points_in_boxes_cpu(
            torch.from_numpy(pc0[:, :3][np.newaxis, :, :]), torch.from_numpy(gt_bbox_3d[np.newaxis, :])
        )
        object_points_list = []
        j = 0
        while j < points_in_boxes.shape[-1]:
            object_points_mask = points_in_boxes[0][:, j].bool()
            object_points = pc0[object_points_mask]
            object_points_list.append(object_points)
            j = j + 1

        moving_mask = torch.ones_like(points_in_boxes)
        points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
        points_mask = ~(points_in_boxes[0])

        ############################# get point mask of the vehicle itself ##########################
        range = config["self_range"]
        oneself_mask = torch.from_numpy(
            (np.abs(pc0[:, 0]) > range[0]) | (np.abs(pc0[:, 1]) > range[1]) | (np.abs(pc0[:, 2]) > range[2])
        )

        ############################# get static scene segment ##########################
        points_mask = points_mask & oneself_mask
        pc = pc0[points_mask]

        ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
        lidar_ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
        lidar_calibrated_sensor = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        lidar_pc = lidar_to_world_to_lidar(
            pc.copy(), lidar_calibrated_sensor.copy(), lidar_ego_pose.copy(), lidar_calibrated_sensor0, lidar_ego_pose0
        )
        ################## record Non-key frame information into a dict  ########################
        dict = {
            "object_tokens": object_tokens,
            "object_points_list": object_points_list,
            "lidar_pc": lidar_pc.points,
            "lidar_ego_pose": lidar_ego_pose,
            "lidar_calibrated_sensor": lidar_calibrated_sensor,
            "lidar_token": lidar_data["token"],
            "sample_token": lidar_data["sample_token"],
            "is_key_frame": lidar_data["is_key_frame"],
            "has_lidarseg": flag_has_lidarseg,
            "gt_bbox_3d": gt_bbox_3d,
            "converted_object_category": converted_object_category,
            "pc_file_name": pc_file_name.split("/")[-1],
        }
        ################# record semantic information into the dict if it's a key frame  ########################
        if lidar_data["is_key_frame"] and flag_has_lidarseg:
            pc_with_semantic = pc_with_semantic[points_mask]
            lidar_pc_with_semantic = lidar_to_world_to_lidar(
                pc_with_semantic.copy(),
                lidar_calibrated_sensor.copy(),
                lidar_ego_pose.copy(),
                lidar_calibrated_sensor0,
                lidar_ego_pose0,
            )
            dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points

        dict_list.append(dict)
        ################## go to next frame of the sequence  ########################
        curr_sample_token = lidar_data["sample_token"]
        next_sample_token = nusc.get("sample", curr_sample_token)["next"]
        if next_sample_token != "":
            next_lidar_token = nusc.get("sample", next_sample_token)["data"][sensor]
            lidar_data = nusc.get("sample_data", next_lidar_token)
        else:
            break
        # next_token = lidar_data['next']
        # if next_token != '':
        #     lidar_data = nusc.get('sample_data', next_token)
        # else:
        #     break

    ################## concatenate all static scene segments (including non-key frames)  ########################
    lidar_pc_list = [dict["lidar_pc"] for dict in dict_list]
    lidar_pc = np.concatenate(lidar_pc_list, axis=1).T

    ################## concatenate all semantic scene segments (only key frames)  ########################
    lidar_pc_with_semantic_list = []
    for dict in dict_list:
        if dict["is_key_frame"] and dict["has_lidarseg"]:
            lidar_pc_with_semantic_list.append(dict["lidar_pc_with_semantic"])
    lidar_pc_with_semantic = np.concatenate(lidar_pc_with_semantic_list, axis=1).T

    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = []
    object_semantic = []
    for dict in dict_list:
        for i, object_token in enumerate(dict["object_tokens"]):
            if object_token not in object_token_zoo:
                if dict["object_points_list"][i].shape[0] > 0:
                    object_token_zoo.append(object_token)
                    object_semantic.append(dict["converted_object_category"][i])
                else:
                    continue

    object_points_dict = {}

    for query_object_token in object_token_zoo:
        object_points_dict[query_object_token] = []
        for dict in dict_list:
            for i, object_token in enumerate(dict["object_tokens"]):
                if query_object_token == object_token:
                    object_points = dict["object_points_list"][i]
                    if object_points.shape[0] > 0:
                        object_points = object_points[:, :3] - dict["gt_bbox_3d"][i][:3]
                        rots = dict["gt_bbox_3d"][i][6]
                        Rot = Rotation.from_euler("z", -rots, degrees=False)
                        rotated_object_points = Rot.apply(object_points)
                        object_points_dict[query_object_token].append(rotated_object_points)
                else:
                    continue
        object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token], axis=0)

    object_points_vertice = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        object_points_vertice.append(point_cloud[:, :3])
    # print('object finish')

    i = 0
    while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames
        if i >= len(dict_list):
            print("finish scene!")
            return
        dict = dict_list[i]
        is_key_frame = dict["is_key_frame"]
        if not is_key_frame:  # only use key frame as GT
            i = i + 1
            continue

        ################## convert the static scene to the target coordinate system ##############
        lidar_calibrated_sensor = dict["lidar_calibrated_sensor"]
        lidar_ego_pose = dict["lidar_ego_pose"]
        lidar_pc_i = lidar_to_world_to_lidar(
            lidar_pc.copy(),
            lidar_calibrated_sensor0.copy(),
            lidar_ego_pose0.copy(),
            lidar_calibrated_sensor,
            lidar_ego_pose,
        )
        lidar_pc_i_semantic = lidar_to_world_to_lidar(
            lidar_pc_with_semantic.copy(),
            lidar_calibrated_sensor0.copy(),
            lidar_ego_pose0.copy(),
            lidar_calibrated_sensor,
            lidar_ego_pose,
        )
        point_cloud = lidar_pc_i.points.T[:, :3]
        point_cloud_with_semantic = lidar_pc_i_semantic.points.T

        ################# load bbox of target frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(dict["lidar_token"])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.0
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.0
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1
        rots = gt_bbox_3d[:, 6:7]
        locs = gt_bbox_3d[:, 0:3]

        ################# bbox placement ##############
        object_points_list = []
        object_semantic_list = []
        for j, object_token in enumerate(dict["object_tokens"]):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token == object_token_in_zoo:
                    points = object_points_vertice[k]
                    Rot = Rotation.from_euler("z", rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    points = rotated_object_points + locs[j]
                    if points.shape[0] >= 5:
                        points_in_boxes = points_in_boxes_cpu(
                            torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                            torch.from_numpy(gt_bbox_3d[j : j + 1][np.newaxis, :]),
                        )
                        points = points[points_in_boxes[0, :, 0].bool()]

                    object_points_list.append(points)
                    semantics = np.ones_like(points[:, 0:1]) * object_semantic[k]
                    object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))

        try:  # avoid concatenate an empty array
            temp = np.concatenate(object_points_list)
            scene_points = np.concatenate([point_cloud, temp])
        except:
            scene_points = point_cloud
        try:
            temp = np.concatenate(object_semantic_list)
            scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
        except:
            scene_semantic_points = point_cloud_with_semantic

        ################## remain points with a spatial range ##############
        mask = (
            (np.abs(scene_points[:, 0]) < 50)
            & (np.abs(scene_points[:, 1]) < 50)
            & (scene_points[:, 2] > -5.0)
            & (scene_points[:, 2] < 3.0)
        )
        scene_points = scene_points[mask]

        ################## get mesh via Possion Surface Reconstruction ##############
        point_cloud_original = o3d.geometry.PointCloud()
        with_normal2 = o3d.geometry.PointCloud()
        point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])
        with_normal = preprocess(point_cloud_original, config)
        with_normal2.points = with_normal.points
        with_normal2.normals = with_normal.normals
        # mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'], config['min_density'], with_normal2)

        point = np.asarray(with_normal.points)
        normal = np.asarray(with_normal.normals)

        point = torch.from_numpy(point).float().cuda()
        normal = torch.from_numpy(normal).float().cuda()

        with torch.no_grad():
            nksr_mesh = nksr_mesh_normal(point, normal, detail_level=0.5, mise_iter=1, cpu_=False)

        scene_points = np.asarray(nksr_mesh.v.cpu(), dtype=float)

        ################## remain points with a spatial range ##############
        mask = (
            (np.abs(scene_points[:, 0]) < 50)
            & (np.abs(scene_points[:, 1]) < 50)
            & (scene_points[:, 2] > -5.0)
            & (scene_points[:, 2] < 3.0)
        )
        scene_points = scene_points[mask]

        ################## convert points to voxels ##############
        pcd_np = scene_points
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
        pcd_np = np.floor(pcd_np).astype(np.int32)
        voxel = np.zeros(occ_size)
        voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

        ################## convert voxel coordinates to LiDAR system  ##############
        gt_ = voxel
        x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
        y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
        z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        vv = np.stack([X, Y, Z], axis=-1)
        fov_voxels = vv[gt_ > 0]
        fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
        fov_voxels[:, 0] += pc_range[0]
        fov_voxels[:, 1] += pc_range[1]
        fov_voxels[:, 2] += pc_range[2]

        ################## get semantics of sparse points  ##############
        mask = (
            (np.abs(scene_semantic_points[:, 0]) < 50)
            & (np.abs(scene_semantic_points[:, 1]) < 50)
            & (scene_semantic_points[:, 2] > -5.0)
            & (scene_semantic_points[:, 2] < 3.0)
        )
        scene_semantic_points = scene_semantic_points[mask]

        ################## Nearest Neighbor to assign semantics ##############
        dense_voxels = fov_voxels
        sparse_voxels_semantic = scene_semantic_points

        x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
        y = torch.from_numpy(sparse_voxels_semantic[:, :3]).cuda().unsqueeze(0).float()
        
        # d1, d2, idx1, idx2 = chamfer.forward(x, y)
        # === 修改：调用我们上面定义的适配函数 ===
        d1, d2, idx1, idx2 = custom_chamfer_forward(x, y)
        # ======================================
        
        indices = idx1[0].cpu().numpy()

        dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
        dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

        # to voxel coordinate
        pcd_np = dense_voxels_with_semantic
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
        # -------------修改2-------------
        # dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int32)
        # -------------修改2-------------

        # ================== [核心修改：保存逻辑] ==================
        # # dirs = os.path.join(save_path, 'dense_voxels_with_semantic/')
        # dirs = smart_path_join(save_path, "dense_voxels_with_semantic/")
        # if not os.path.exists(dirs):
        #     os.makedirs(dirs)
        # # np.save(os.path.join(dirs, dict['pc_file_name'] + '.npy'), dense_voxels_with_semantic)
        # # lidar_data_path = smart_path_join(dirs, dict["sample_token"], dict["lidar_token"] + ".npy")
        # # === 修改开始：构建完整目录并确保存在 ===
        # # 1. 构造该样本的具体保存文件夹路径
        # target_dir = smart_path_join(dirs, dict["sample_token"])
        
        # # 2. 如果文件夹不存在，创建它 (包含父级目录)
        # if not os.path.exists(target_dir):
        #     os.makedirs(target_dir, exist_ok=True)

        # # 3. 构造完整文件路径
        # lidar_data_path = smart_path_join(target_dir, dict["lidar_token"] + ".npy")
        # # === 修改结束 ===
        
        # with open(lidar_data_path, "wb") as f:
        #     pickle.dump(dense_voxels_with_semantic, f)

        # i = i + 1

        # torch.cuda.empty_cache()

        # continue
        # [修改后] 支持双模式保存为 .npz
        token = dict["lidar_token"]
        sample_token = dict["sample_token"]
        
        # 模式 1: Standard (适配 dataset.py, 类似 Occ3D 官方结构)
        # 路径: save_path/scene_name/sample_token/labels.npz
        if args.save_mode == 'standard':
            if dict["is_key_frame"]: # 标准模式通常只存关键帧
                target_dir = smart_path_join(save_path, scene_name, sample_token)
                os.makedirs(target_dir, exist_ok=True)
                save_file = smart_path_join(target_dir, "labels.npz")
                np.savez_compressed(save_file, semantics=dense_voxels_with_semantic)

        # 模式 2: 12Hz (适配视频生成插值，扁平结构)
        # 路径: save_path/dense_voxels_with_semantic/sample_token/labels.npz
        # 注意：这里我们把文件名也改为 labels.npz 以保持格式统一，但在 read_occ_nksr.py 里要做适配
        elif args.save_mode == '12hz':
            target_dir = smart_path_join(save_path, "dense_voxels_with_semantic", sample_token)
            os.makedirs(target_dir, exist_ok=True)
            save_file = smart_path_join(target_dir, "labels.npz") 
            np.savez_compressed(save_file, semantics=dense_voxels_with_semantic)

        i = i + 1
        torch.cuda.empty_cache()
        continue

def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


if __name__ == "__main__":

    parse = ArgumentParser()
    parse.add_argument("--dataset", type=str, default="nuscenes")
    parse.add_argument("--config_path", type=str, default="./data_process/config-800.yaml")
    parse.add_argument("--split", type=str, default="val")
    # ---------------修改3---------------
    parse.add_argument("--save_path", type=str, default="./data/gts/")
    # ---------------修改3---------------
    # parse.add_argument("--dataroot", type=str, default="./data/nuScenes/")
    parse.add_argument("--dataroot", type=str, default="./data/nuscenes/")
    # parse.add_argument("--label_mapping", type=str, default="./nuscenes.yaml")
    parse.add_argument("--label_mapping", type=str, default="./nuscenes.yaml")
    parse.add_argument("--index_list", nargs="+", type=int)
    # <details>
    #   #### --config_path
    #   The configuration file path.
    #   #### --split
    #   train or val.
    #   #### --save_path
    #   Path to save the generated occupancy data.
    #   #### --dataroot
    #   Path to the dataset.
    #   #### --label_mapping
    #   Path to the label mapping file.
    #   #### --index_list
    #   List of indices to generate the occupancy data.
    # ==================== [新增代码开始] ====================
    # 添加 save_mode 参数，控制是生成标准分层结构(standard)还是扁平结构(12hz)
    parse.add_argument("--save_mode", type=str, default="standard", 
                       choices=["standard", "12hz"], 
                       help="standard: for dataset.py training; 12hz: for video generation")
    # ==================== [新增代码结束] ====================
    args = parse.parse_args()

    # if args.dataset == "nuscenes":
    #     nusc = NuScenes(version="advanced_12Hz_trainval", dataroot=args.dataroot, verbose=True)
    #     train_scenes = splits.train
    #     val_scenes = splits.val
    # --- 修改后 ---
    if args.dataset == 'nuscenes':
        # [修改] 版本改为 v1.0-mini
        print(f"Initializing NuScenes (v1.0-mini) from {args.dataroot}...")
        nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=True)
        
        # [修改] Mini 数据集没有官方 split，直接读取所有场景
        all_scenes = [s['name'] for s in nusc.scene]
        # 3. 这里的 train_scenes 和 val_scenes 变量其实在后面没用到，
        # 但为了防止潜在的引用错误，我们先赋值
        train_scenes = all_scenes
        val_scenes = all_scenes
    else:
        print("Dataset not supported")

    # load config
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # load learning map
    label_mapping = args.label_mapping
    with open(label_mapping, "r") as stream:
        nuscenesyaml = yaml.safe_load(stream)

    index_list = args.index_list
    
    # [新增] 如果没指定 index_list，默认跑所有场景
    if index_list is None:
        index_list = range(len(nusc.scene))

    for index in index_list:
        print("processing sequecne:", index)
        try:
            main(nusc, indice=index, nuscenesyaml=nuscenesyaml, args=args, config=config)
        except Exception as e:
            import traceback
            traceback.print_exc() # 打印完整报错堆栈，方便调试
            with open("./scene_error_nksr.txt", "a") as f:
                f.write(str(index) + "\n")
            print(e)
            continue

        with open("./scene_nksr.txt", "a") as f:
            f.write(str(index) + "\n")
