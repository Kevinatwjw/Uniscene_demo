import pickle
from collections import OrderedDict

from nuscenes import NuScenes
from tqdm import tqdm

if __name__ == "__main__":
    new_infos = []
    split = "val"
    ori_info_path = f"data/nuScenes/nuscenes_mmdet3d-12Hz/nuscenes_advanced_12Hz_infos_{split}.pkl"

    info_filename = ori_info_path.split("/")[-1].split(".")[0]
    with open(ori_info_path, "rb") as f:
        infos = pickle.load(f)
    all_frame_tokens = [item["token"] for item in infos["infos"]]

    if "12Hz" not in ori_info_path:
        nusc = NuScenes("v1.0-trainval", "data/nuScenes/v1.0-trainval", verbose=True)
        for i in range(len(infos["infos"])):
            infos["infos"][i]["lidar_top_data_token"] = nusc.get("sample", infos["infos"][i]["token"])["data"][
                "LIDAR_TOP"
            ]

        all_frame_tokens = [item["lidar_top_data_token"] for item in infos["infos"]]
        scene_tokens = OrderedDict()
        for item in infos["infos"]:
            if item["scene_token"] not in scene_tokens:
                scene_tokens[item["scene_token"]] = []
            # scene_tokens[item['scene_token']].append(item['token'])
            scene_tokens[item["scene_token"]].append(item["lidar_top_data_token"])
        infos["scene_tokens"] = list(scene_tokens.values())

    for item in tqdm(infos["scene_tokens"]):
        scene_info = []
        # scene_info = {}
        for frame_token in item:
            scene_info.append((frame_token, infos["infos"][all_frame_tokens.index(frame_token)]))
            # scene_info.update({frame_token: infos['infos'][all_frame_tokens.index(frame_token)]})
        new_infos.append(scene_info)
    with open(f"./{info_filename}_converted.pkl", "wb") as f:
        pickle.dump(new_infos, f)
