#!/usr/bin/env bash

set -x

EXP_DIR=results/r50_voc07to12_ep500_burnin80
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} \
    --BURN_IN_STEP 80 \
    --TEACHER_UPDATE_ITER 1 \
    --EMA_KEEP_RATE 0.9996 \
    --annotation_json_label 'instances_VOC_trainval2007_semi_label.json' \
    --annotation_json_unlabel 'instances_VOC_trainval2012_semi_unlabel.json' \
    --CONFIDENCE_THRESHOLD 0.7 \
    --data_path './voc' \
    --lr 2e-4 \
    --epochs 500 \
    --lr_drop 500 \
    --pixels 800 \
    --dataset_file 'voc_semi' \
    --resume '' \

