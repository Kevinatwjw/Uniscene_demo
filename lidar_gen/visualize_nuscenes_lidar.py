import argparse
import os

import numpy as np
from pyvirtualdisplay import Display
from tqdm import tqdm

display = Display(visible=False, size=(1280, 1024))
display.start()
import mayavi
from mayavi import mlab

mlab.options.offscreen = True


def draw_nusc_occupancy(
    pointcloud,
    save_folder=None,
    ori_name=None
    # cat_save_file=None,
):

    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point

    # r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    np.degrees(np.arctan(z / d))
    vals = "-height"
    if vals == "height":
        col = z
    else:
        col = d
    fig = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    mayavi.mlab.points3d(
        x,
        y,
        z,
        col,  # Values used for Color
        mode="cube",
        colormap="copper",  # 'bone', 'copper', 'gnuplot'
        # color=(1, 1, 1),   # Used a fixed (r,g,b) instead
        figure=fig,
        scale_mode="none",
        scale_factor=0.25,
    )

    scene = fig.scene

    os.makedirs(save_folder, exist_ok=True)
    visualize_keys = ["DRIVING_VIEW", "BIRD_EYE_VIEW"]

    for i in range(2):
        # bird-eye-view and facing front
        if i == 2:
            scene.camera.position = [0.75131739, -35.08337438, 16.71378558]
            scene.camera.focal_point = [0.75131739, -34.21734897, 16.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.0]
            scene.camera.compute_view_plane_normal()
            scene.render()

        # bird-eye-view
        else:
            scene.camera.position = [0.75131739, 0.78265103, 93.21378558]
            scene.camera.focal_point = [0.75131739, 0.78265103, 92.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0.0, 1.0, 0.0]
            scene.camera.clipping_range = [0.01, 400.0]
            scene.camera.compute_view_plane_normal()
            scene.render()

        save_file = os.path.join(save_folder, visualize_keys[i], ori_name)
        print(save_file)

        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        mlab.savefig(save_file)

    mlab.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pred_dir", default="lidar//pred//")
    parser.add_argument("--save_path", default="lidar//pred//vis")
    args = parser.parse_args()

    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    occ_size = [256, 256, 32]
    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    voxel_size = [voxel_x, voxel_y, voxel_z]

    # noqa
    constant_f = 0.0055
    sample_files = os.listdir(args.pred_dir)
    save_path = args.save_path

    for filename in tqdm(sample_files):

        pointcloud = np.load(os.path.join(args.pred_dir, filename))

        draw_nusc_occupancy(
            pointcloud=pointcloud, save_folder=save_path, ori_name=filename.split("//")[-1].split("_")[0] + ".npy"
        )
