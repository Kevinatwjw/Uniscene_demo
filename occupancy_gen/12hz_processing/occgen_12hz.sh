# save 12hz 200*200 bevlayout
# python save_bevlayout_12hz.py

#infer 12hz occupancy with OccDiT training on Occ3D
# python eval_12hz_with_occ3d.py
# 核心修改：
# 1. 显式指定 --imageset 为 mini pkl
# 2. 限制 batch size

# 第一步：生成 12Hz BEV
echo "Generating 12Hz BEV Layouts..."
python save_bevlayout_12hz.py \
    --imageset="../../data/nuscenes_mmdet3d-12Hz/nuscenes_infos_val_mini_dict.pkl" \
    --s_p 0 --e_p 10

# 第二步：运行 12Hz 推理 (依赖 eval_12hz_with_occ3d.py 里的路径修改)
echo "Inferring 12Hz Occupancy..."
python eval_12hz_with_occ3d.py \
    --global-batch-size 1 \
    --vis