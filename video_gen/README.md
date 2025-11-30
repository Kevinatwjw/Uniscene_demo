### Framework:

<div align=center><img width="740" height="420" src="./asserts/video_model.png"/></div>

## Installation

1. Create conda environment with python version 3.9

2. Install all the packages with requirements.txt

## Preparing

1. Link nuscenes dataset to `./data/nuscenes`

2. Download Video [checkpoint](https://huggingface.co/Arlolo0/UniScene/resolve/main/video_pretrained.safetensors?download=true), put them in `./ckpts/`

3. Download [video annotation files](https://nbeitech-my.sharepoint.com/:u:/g/personal/bli_eitech_edu_cn/ESW1JZ4t2LVKqEXh_-Dkn08BVbmPlpj7SUW5rfFIoe34kQ?e=lA7Jh7), put them in `./annos/`

4. Prapare the GS rendering maps for video sampling:

```
python  ./gs_render/render_eval_condition_gt.py  --occ_path  $input_occ_path  --layout_path $input_bev_path --render_path $output_render_path  --vis
```

## Getting Started

Inference on nuScenes validation set

```
python  inference_video.py  --occ_data_root $output_render_path   --save  $output_video_path   --dataset  NUSCENES_OCC
```

## Acknowledgements

Many thanks to these excellent open source projects:

- [SVD](https://github.com/Stability-AI/generative-models)

- [Vista](https://github.com/OpenDriveLab/Vista/tree/main)

- [HUGS](https://github.com/hyzhou404/HUGS)
