#!/usr/bin/env bash

set -x

EXP_DIR=results/r50_omni_bees_20fully0Unsup0tagsU34tagsK0pointsU0pointsK46boxesEC0boxesU_ep1k_burnin200
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 200 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'instances_bees_train_bees_omni_label_seed1709_20fully0Unsup0tagsU34tagsK0pointsU0pointsK46boxesEC0boxesU.json' \
    --annotation_json_unlabel 'instances_bees_train_bees_omni_unlabel_seed1709_20fully0Unsup0tagsU34tagsK0pointsU0pointsK46boxesEC0boxesU.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path './bees' \
    --lr 2e-4 \
    --epochs 1000 \
    --lr_drop 1000 \
    --pixels 600 \
    --save_freq 100 \
    --eval_freq 20 \
    --dataset_file 'bees_omni' \
    --resume '' \
