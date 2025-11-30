### Framework:

<div align=center><img width="760" height="340" src="./asserts/occupancy_model.png"/></div>

## Installation

1. Create conda environment with python version 3.9

```
conda create -n uniscene python=3.9
```

2. Install all the packages with requirements.txt

3. Anything about the installation of mmdetection3d, please refer to [mmdetection3d](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation)

## Preparing

1. Link nuscenes dataset to `data/nuscenes`

2. Prepare the gts semantic occupancy in "./data/gts", which is introduced in [Occ3d](https://github.com/Tsinghua-MARS-Lab/Occ3D)

3. Download the train/val pickle files from [OccWorld](https://github.com/wzzheng/OccWorld) and put them in "./data/" :

- [nuscenes_infos_train_temporal_v3_scene.pkl](https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/)

- [nuscenes_infos_val_temporal_v3_scene.pkl](https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/)

3. Download [checkpoint](https://nbeitech-my.sharepoint.com/:f:/g/personal/bli_eitech_edu_cn/EpYIjg5_l2VFoYJd2vZcl9wBFeVQV1XI_NPQQhXOB-wUqQ?e=I3vmYQ) and put them in "./ckpt/".

4. (Optional) Prepare 12HZ BEV maps.

```
python save_bev_layout.py \
    --py-config="./config/save_step2_me.py" \
    --work-dir="./ckpt/VAE/"
```

## Getting Started

1. 2Hz keyframe Occupancy Inference on nuScenes validation set:

```
bash ./run_eval_dit.sh
```

2. 12Hz Occupancy Inference on nuScenes validation set:

```
bash ./12hz_processing/occgen_12hz.sh
```

3. Visulization of Seamntic Occupancy Grids

```
python visualize_nuscenes_occupancy.py  --input_dir $input_occupancy_path  --save_path  $output_vis_path
```

## Acknowledgements

Many thanks to these excellent open source projects:

- [Occworld](https://github.com/wzzheng/OccWorld)

- [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)

- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)
