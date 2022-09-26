#!/usr/bin/env bash

set -x

EXP_DIR=results/r50_omni_coco_20fully16Unsup0tagsU0tagsK0pointsU0pointsK64boxesEC0boxesU_ep150_burnin20
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 20 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'instances_train2017_coco_omni_label_seed1709_20fully16Unsup0tagsU0tagsK0pointsU0pointsK64boxesEC0boxesU.json' \
    --annotation_json_unlabel 'instances_train2017_coco_omni_unlabel_seed1709_20fully16Unsup0tagsU0tagsK0pointsU0pointsK64boxesEC0boxesU.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path './coco' \
    --lr 2e-4 \
    --epochs 150 \
    --lr_drop 150 \
    --pixels 600 \
    --save_freq 20 \
    --dataset_file 'coco_omni' \
    --resume '' \
