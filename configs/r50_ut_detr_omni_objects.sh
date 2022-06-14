#!/usr/bin/env bash

set -x

EXP_DIR=results/r50_omni_objects_25fully0Unsup0tagsU0tagsK34pointsU0pointsK41boxesEC0boxesU_ep200_burnin40
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 40 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'objects365_train_sampled_objects_omni_label_seed1709_25fully0Unsup0tagsU0tagsK34pointsU0pointsK41boxesEC0boxesU.json' \
    --annotation_json_unlabel 'objects365_train_sampled_objects_omni_unlabel_seed1709_25fully0Unsup0tagsU0tagsK34pointsU0pointsK41boxesEC0boxesU.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path './objects365' \
    --lr 2e-4 \
    --epochs 200 \
    --lr_drop 200 \
    --pixels 600 \
    --save_freq 20 \
    --eval_freq 5 \
    --dataset_file 'objects_omni' \
    --resume '' \

