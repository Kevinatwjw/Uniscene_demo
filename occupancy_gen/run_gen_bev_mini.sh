#!/bin/bash
# run_gen_bev_mini.sh

# 这里的参数对应你修改后的代码逻辑：
# 1. work-dir: 结果会保存在 ckpt/VAE_Mini/val/bevmap_4/
# 2. py-config: 读取修改过的配置文件

python save_bev_layout.py \
    --py-config="./config/save_step2_me.py" \
    --work-dir="./ckpt/VAE_Mini/"