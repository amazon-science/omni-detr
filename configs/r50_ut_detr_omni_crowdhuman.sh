#!/usr/bin/env bash

set -x

EXP_DIR=results/r50_omni_crowdhuman_20fully0Unsup0tagsU0tagsK34pointsU0pointsK46boxesEC0boxesU_ep500_burnin100
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 100 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'train_fullbody_crowdhuman_omni_label_seed1709_20fully0Unsup0tagsU0tagsK34pointsU0pointsK46boxesEC0boxesU.json' \
    --annotation_json_unlabel 'train_fullbody_crowdhuman_omni_unlabel_seed1709_20fully0Unsup0tagsU0tagsK34pointsU0pointsK46boxesEC0boxesU.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path './crowdhuman' \
    --lr 2e-4 \
    --epochs 500 \
    --lr_drop 500 \
    --pixels 600 \
    --save_freq 100 \
    --eval_freq 5 \
    --dataset_file 'crowdhuman_omni' \
    --resume '' \
