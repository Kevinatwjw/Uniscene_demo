import numpy as np
import torch
import math
from tqdm import tqdm
#compile CUDA kernal
from torch.utils.cpp_extension import load
from dda3d_gpu import dda3d_gpu
from dda_cpu import check_rays_cpu
from check_rays_interp import check_rays_interp

 
def check_rays_cuda(grid, rays, Xmin, Ymin, Zmin, vx, vy, vz):
 
    W, H, D = grid.shape
    num_rays = rays.shape[0]

 
    intersections = torch.zeros(num_rays, dtype=torch.bool, device='cuda')

 
    dda3d_gpu(
        rays, grid, intersections,
        Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays,
    )

 
    return intersections#.cpu().numpy()


if __name__ == '__main__':
    # Example usage:
    # Define grid dimensions and voxel sizes
    W, H, D = 200, 200, 16  # Grid size in voxels
    vx, vy, vz = 0.5, 0.5, 0.5  # Voxel sizes in meters

    # Define grid boundaries
    Xmin, Ymin, Zmin = -50.0, -50.0, -5.0
    Xmax, Ymax, Zmax = 50.0, 50.0, 3.0

    # Create occupancy grid
    # grid = np.zeros((W, H, D), dtype=np.uint8)
    # Assume some occupied voxels
    grid = np.load('/code/OpenPCDet/bd00c003806e418d88c4315d17eb09b7.npy').astype(bool)
    # grid = np.random.randint(0, 2, (W, H, D)).astype(np.bool_)
    
    # Define rays
    rays = np.random.randn(100, 3)



    intersections2 = check_rays_cpu(grid, rays, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D)
    
    grid = torch.from_numpy(grid).contiguous().cuda().bool()
    rays = torch.from_numpy(rays).contiguous().cuda().float()
    intersections = check_rays_cuda(grid, rays, Xmin, Ymin, Zmin, vx, vy, vz).cpu().numpy()
    
    #intersections2 = check_rays_interp(grid, rays, Xmin, Ymin, Zmin, Xmax, Ymax, Zmax, vx, vy, vz, W, H, D)#.cpu().numpy()
    print(intersections.sum())
    print(intersections2.sum())
    print(np.logical_xor(intersections, intersections2).sum())

    # import time
    # t1 = time.perf_counter()
    # for i in tqdm(range(1000)):
    #     # Check intersections
    #     intersections = check_rays_cuda(grid, rays, Xmin, Ymin, Zmin, vx, vy, vz)
    # t2 = time.perf_counter()
    # print(t2-t1)


    # t1 = time.perf_counter()
    
    # for i in tqdm(range(1000)):
    #     #intersections2 = check_rays(grid, rays, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D)

    #     intersections2 = check_rays_interp(grid, rays, Xmin, Ymin, Zmin, Xmax, Ymax, Zmax, vx, vy, vz, W, H, D, bins)#.cpu().numpy()
    # t2 = time.perf_counter()
    # print(t2-t1)
    # pass