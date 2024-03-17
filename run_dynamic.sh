#!/usr/bin/env bash

# 结果存在哪
SAVE_DIR=checkpoints/dynamic/mochang/
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path
SPLIT_ID=1 #先做一个split
DATASET=rdd

for seed in 1
do
    for shot in 3
    do
        python3 tools/create_config.py --dataset ${DATASET} --config_root configs/${DATASET}               \
            --shot ${shot} --seed ${seed} --split ${SPLIT_ID}
        CONFIG_PATH=configs/${DATASET}/DGT_rdd_split${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/${SPLIT_ID}/${shot}shot_seed${seed}
        python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH}                            \
            --opts OUTPUT_DIR ${OUTPUT_DIR}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done

#python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID} --shot-list 1 2 3 5 10  # surmarize all results
