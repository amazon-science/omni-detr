#!/usr/bin/env bash

set -x

EXP_DIR=results/r50_coco_additional_ep500_burnin10
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 10 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'instances_train2017_w_indicator.json' \
    --annotation_json_unlabel 'image_info_unlabeled2017_w_indicator.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path './coco' \
    --lr 2e-4 \
    --epochs 1000 \
    --lr_drop 1000 \
    --pixels 800 \
    --dataset_file 'coco_add_semi' \
    --resume '' \

