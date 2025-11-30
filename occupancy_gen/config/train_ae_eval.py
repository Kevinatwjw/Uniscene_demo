grad_max_norm = 35
print_freq = 10
# max_epochs = 200
# [修改] 训练轮次
max_epochs = 5
warmup_iters = 1000
return_len_ = 5

# [新增]
nusc_version = "v1.0-mini"

multisteplr = False
multisteplr_config = dict(
    decay_t=[87 * 500], decay_rate=0.1, warmup_t=warmup_iters, warmup_lr_init=1e-6, t_in_epochs=False
)


optimizer = dict(
    optimizer=dict(
        type="AdamW",
        lr=1e-3,  # 1e-3,
        weight_decay=0.01,
    ),
)

data_path = "data/nuscenes/"

# [修改] 训练集配置
train_dataset_config = dict(
    type="nuScenesSceneDatasetLidar_ori",
    data_path=data_path,
    return_len=return_len_,
    offset=0,
    nusc_dataroot="data/nuscenes",
    # imageset="data/nuscenes_infos_train_temporal_v3_scene.pkl",
    imageset="data/nuscenes_mmdet3d-12Hz/nuscenes_infos_train_mini_dict.pkl", # [修改]
)

val_dataset_config = dict(
    type="nuScenesSceneDatasetLidar_ori",
    data_path=data_path,
    return_len=return_len_,
    offset=0,
    nusc_dataroot="data/nuscenes",
    # imageset="data/nuscenes_infos_val_temporal_v3_scene.pkl",
    imageset="data/nuscenes_mmdet3d-12Hz/nuscenes_infos_val_mini_dict.pkl", # [修改]
)

train_wrapper_config = dict(
    type="tpvformer_dataset_nuscenes",
    phase="train",
)

val_wrapper_config = dict(
    type="tpvformer_dataset_nuscenes",
    phase="val",
)

train_loader = dict(
    batch_size=1,
    shuffle=True,
    num_workers=5,
)

val_loader = dict(
    batch_size=1,
    shuffle=False,
    num_workers=5,
)


loss = dict(
    type="MultiLoss",
    loss_cfgs=[
        dict(
            type="ReconLoss",
            weight=10.0,
            ignore_label=-100,
            use_weight=False,
            cls_weight=None,
            input_dict={"logits": "logits", "labels": "inputs"},
        ),
        dict(type="LovaszLoss", weight=1.0, input_dict={"logits": "logits", "labels": "inputs"}),
    ],
)

loss_input_convertion = dict(
    logits="logits",
    # kl_loss ='kl_loss'
    # embed_loss='embed_loss'
)


load_from = ""


expansion = 4
num_classes = 18


# model = dict(
#     num_classes=18,
#     expansion=expansion)

shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"
