### Framework:

<div align=center><img width="690" height="450" src="./asserts/lidar_model.png"/></div>

## Installation

1. Create conda environment with python version 3.9

```
conda create -n uniscene python=3.9
```

2. Install all the packages with requirements.txt

3. Install OpenPCDet and dda

```
    cd pcdet && pip install -e . -v
```

```
    cd pcdet/datasets/nuscenes_occ/utils/dda && pip install . -v
```

Anything about the installation of OpenPCDet, please refer to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Preparing

1. Link nuscenes dataset to `data/nuscenes`

2. Download [checkpoint](https://nbeitech-my.sharepoint.com/:u:/g/personal/bli_eitech_edu_cn/EcMXzN216PpHtvnfWubStf0BrqsfuNG4WSy8fmH08Qt_8Q?e=528VAR), put it in `checkpoints`

3. Download nuscenes [pickle files](https://drive.google.com/drive/folders/15geieBuTUxRJGqOYgzRVd20sZ4RCsisR?usp=share_link), put them in `data/infos`. 

(Optional) Or convert the ["data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos\_\*.pkl"](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155157018_link_cuhk_edu_hk/EXunN1j0OmNLtaPoh2VrkgQBGpyXiMlltuCX5GBuYc00YQ?e=bVI9AC) with the provided script:

```
python convert_nuscenes_info.py
```

1. Download split mapping files [nuScenes_occ2lidar_nksr.json](https://nbeitech-my.sharepoint.com/:u:/g/personal/bli_eitech_edu_cn/EaTyjCQqjMxGr7AWjF49JjMBa1joUVqaMI5Lz_fIezdIpw?e=ogGIYg), [nuScenes_nksr_occ_train.json](https://nbeitech-my.sharepoint.com/:u:/g/personal/bli_eitech_edu_cn/EfPoB8rz5HtErXONEBxU9YIBjQV73xzMmXKHVU1hBGzaog?e=3Js5CP) and [nuScenes_nksr_occ_val.json](https://nbeitech-my.sharepoint.com/:u:/g/personal/bli_eitech_edu_cn/EUqHsyju9nFKiRqNArrOgiABMjGO_YCUWXUMnfg2tH7Xhg?e=dOk0Im), put them in `data/split`.

2. Prepare occupancy files with [Ground-truth Occupancy](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/blob/b93bfc98b18c8017d97e8fff1a6bfc4a9d5a2deb/README.md?plain=1#L87) or [Occupancy Generation](https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/blob/master/occupancy_gen/README.md), put them in `data/nuscenes_occ`

## Getting Started

1. Inference on nuScenes validation set

```
    python tools/test.py --save_to_file  $output_path  --ckpt_dir  $lidar_pretrain
```

2. Evaluation on nuScenes validation set

```
    python lidar_gen/pcdet/datasets/nuscenes_occ/eval_utils/eval_mmd_jsd_gpu_batch.py $gt_path $pred_path
```

3. Costumed Inference Demo

```
    python tools/demo.py --occ_file  $input_occupancy_path  --ckpt_dir $lidar_pretrain_path
```

4. Visulization of LiDAR Point Clouds

```
    python visualize_nuscenes_lidar.py  --input_dir $input_lidar_path  --save_path  $output_vis_path
```


## Acknowledgements

Many thanks to these excellent open source projects:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

- [NeuS](https://github.com/Totoro97/NeuS)

- [NeuRAD](https://github.com/georghess/neurad-studio/tree/main)
