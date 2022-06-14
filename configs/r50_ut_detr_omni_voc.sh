#!/usr/bin/env bash

set -x

EXP_DIR=results/r50_omni_voc_20fully19Unsup0tagsU0tagsK0pointsU0pointsK61boxesEC0boxesU_ep500_burnin100
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 100 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'instances_VOC_trainval20072012_voc_omni_label_seed1709_20fully19Unsup0tagsU0tagsK0pointsU0pointsK61boxesEC0boxesU.json' \
    --annotation_json_unlabel 'instances_VOC_trainval20072012_voc_omni_unlabel_seed1709_20fully19Unsup0tagsU0tagsK0pointsU0pointsK61boxesEC0boxesU.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path './voc' \
    --lr 2e-4 \
    --epochs 500 \
    --lr_drop 500 \
    --pixels 600 \
    --save_freq 100 \
    --eval_freq 5 \
    --dataset_file 'voc_omni' \
    --resume '' \
