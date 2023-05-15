#!/usr/bin/env bash

# 结果存在哪
SAVE_DIR=checkpoints/rdd1/
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path
SPLIT_ID=1


# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus 1 --config-file configs/rdd/defrcn_det_r101_base1.yaml     \
    --opts OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}

# ----------------------------- Model Preparation --------------------------------- #
# 基类训练完后把model_reset_surgery.pth存在SAVE_DIR里
python3 tools/model_surgery.py --dataset voc --method randinit                                \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                    \
     --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth


# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 1 2 3 4 5 6 7 8 9 10
do
    for shot in 1 2 3 5 10
    do
        python3 tools/create_config.py --dataset voc --config_root configs/rdd               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/rdd/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
#        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like --shot-list 1 2 3 5 10  # surmarize all results