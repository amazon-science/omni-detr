# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection
from .coco import build as build_coco
from .coco import build_semi_label as build_coco_semi_label
from .coco import build_semi_unlabel as build_coco_semi_unlabel

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco

def build_dataset(image_set, label, args):
    if args.dataset_file == 'coco' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'coco_omni' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'coco_omni' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'coco_omni' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_omni' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'coco_add_semi' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'coco_add_semi' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'coco_add_semi' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_add_semi' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'coco_35to80_tagsU' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'coco_35to80_tagsU' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'coco_35to80_tagsU' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_35to80_tagsU' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'coco_35to80_point' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'coco_35to80_point' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'coco_35to80_point' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_35to80_point' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'coco_objects_tagsU' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'coco_objects_tagsU' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'coco_objects_tagsU' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_objects_tagsU' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'coco_objects_points' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'coco_objects_points' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'coco_objects_points' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_objects_points' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'bees_omni' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'bees_omni' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'bees_omni' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'bees_omni' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'voc_semi' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'voc_semi' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'voc_semi' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'voc_semi' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'voc_omni' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'voc_omni' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'voc_omni' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'voc_omni' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'objects_omni' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'objects_omni' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'objects_omni' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'objects_omni' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    if args.dataset_file == 'crowdhuman_omni' and image_set == 'train' and label == True:
        return build_coco_semi_label(image_set, args)
    if args.dataset_file == 'crowdhuman_omni' and image_set == 'train' and label == False:
        return build_coco_semi_unlabel(image_set, args)
    if args.dataset_file == 'crowdhuman_omni' and image_set == 'val' and label == True:
        return build_coco(image_set, args)
    if args.dataset_file == 'crowdhuman_omni' and image_set == 'burnin' and label == True:
        return build_coco('train', args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
