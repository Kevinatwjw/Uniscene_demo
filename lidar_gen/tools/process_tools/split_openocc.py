import json
import os
import pickle
from glob import glob


lidar_root = "/nuscenes/data/nuscenes/samples/LIDAR_TOP/"
# full_list = glob( lidar_root+"*" )
# train_set, test_set = train_test_split(full_list, test_size=0.15, random_state=42)
# train_set = [item+'\n' for item in train_set]
# test_set = [item+'\n' for item in test_set]
# with open('data/openocc/train.txt', 'w') as f:
#     f.writelines(train_set)
# with open('data/openocc/test.txt', 'w') as f:
#     f.writelines(test_set)
# pass

 
lidar_root = "nuscenes/data/nuscenes/samples/LIDAR_TOP/"
full_list = glob(lidar_root + "*")
full_list_set = set(map(lambda x: os.path.join(*x.split("/")[-3:]), full_list))
occ_anno_file = "/code/occ_lidargen/annos/nuScenes_lidar2occ_openocc.json"
with open(occ_anno_file, "r") as anno_json:
    sample_dict = json.load(anno_json)

sample_dict_filtered = dict(filter(lambda kv: kv[0] in full_list_set, sample_dict.items()))
split = "val"
nuscenes_info = f"data/openocc/nuscenes_infos_temporal_{split}_converted.pkl"
with open(nuscenes_info, "rb") as f:
    infos = pickle.load(f)

lidar_top_data_tokens = []
for scene_info in infos:
    lidar_top_data_tokens_scene, _ = list(zip(*scene_info))
    lidar_top_data_tokens.extend(lidar_top_data_tokens_scene)

lidar_top_data_tokens = set(lidar_top_data_tokens)
split_set = []
for k, v in sample_dict_filtered.items():
    if v.split("/")[1] in lidar_top_data_tokens:
        split_set.append("nuscenes/data/nuscenes/" + k + "\n")

with open(f"data/openocc/{split}.txt", "w") as f:
    f.writelines(split_set)
