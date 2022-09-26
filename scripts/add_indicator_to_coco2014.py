# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
from pycocotools.coco import COCO

def main():

    root_dir = '../coco/annotations/'
    json_file = root_dir + 'instances_valminusminival2014.json'
    coco_api = COCO(json_file)
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)

    # add indicator
    for i_img in imgs:
        i_img['indicator'] = 1
        i_img['label_type'] = 'fully'
    dataset_anns = [coco_api.imgToAnns[i] for i in img_ids]
    anns = [ann for img_anns in dataset_anns for ann in img_anns]

    ith = 0
    for i_ann in anns:
        mask_i = coco_api.annToMask(i_ann)
        # sample a point
        valid_idx = np.where(mask_i == 1)
        if np.sum(mask_i) > 0:
            sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
            sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
            sampled_point_i = [float(item) for item in sampled_point_i]
            i_ann['point'] = sampled_point_i
        else:
            if len(i_ann['bbox']) > 0:
                boxes = i_ann['bbox']
                boxes = np.array(boxes)
                mask_i[int(boxes[1]):(int(boxes[1]) + int(boxes[3])),
                int(boxes[0]):(int(boxes[0]) + int(boxes[2]))] = 1
                valid_idx = np.where(mask_i == 1)
                if np.sum(mask_i) > 0:
                    sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
                    sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
                    sampled_point_i = [float(item) for item in sampled_point_i]
                    i_ann['point'] = sampled_point_i
                else:  # at least one of the box size less than 1 pixel
                    if int(boxes[2]) < 1:
                        boxes[2] = boxes[2] + 1
                    if int(boxes[3]) < 1:
                        boxes[3] = boxes[3] + 1
                    mask_i[int(boxes[1]):(int(boxes[1]) + int(boxes[3])),
                    int(boxes[0]):(int(boxes[0]) + int(boxes[2]))] = 1
                    valid_idx = np.where(mask_i == 1)
                    sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
                    sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
                    sampled_point_i = [float(item) for item in sampled_point_i]
                    i_ann['point'] = sampled_point_i
            else:
                i_ann['point'] = []
                print(i_ann['bbox'])
        ith = ith + 1
        if ith % 10000 == 0:
            print(ith)

    data = {}
    data['images'] = imgs
    data['annotations'] = anns
    data['categories'] = list(coco_api.cats.values())
    data['info'] = coco_api.dataset['info']
    data['licenses'] = coco_api.dataset['licenses']
    output_file_label = root_dir + 'instances_valminusminival2014_w_indicator.json'

    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(data, f, sort_keys=True)

    # unlabel part
    json_file = root_dir + 'instances_train2014.json'
    coco_api = COCO(json_file)
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)

    # add indicator
    for i_img in imgs:
        i_img['indicator'] = 0
        i_img['label_type'] = 'tagsU'  # need to assign correct string for different types of annotations, tagsU, pointsK, Unsup
    dataset_anns = [coco_api.imgToAnns[i] for i in img_ids]
    anns = [ann for img_anns in dataset_anns for ann in img_anns]

    ith = 0
    for i_ann in anns:
        mask_i = coco_api.annToMask(i_ann)
        # sample a point
        valid_idx = np.where(mask_i == 1)
        if np.sum(mask_i) > 0:
            sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
            sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
            sampled_point_i = [float(item) for item in sampled_point_i]
            i_ann['point'] = sampled_point_i
        else:
            if len(i_ann['bbox']) > 0:
                boxes = i_ann['bbox']
                boxes = np.array(boxes)
                mask_i[int(boxes[1]):(int(boxes[1]) + int(boxes[3])),
                int(boxes[0]):(int(boxes[0]) + int(boxes[2]))] = 1
                valid_idx = np.where(mask_i == 1)
                if np.sum(mask_i) > 0:
                    sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
                    sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
                    sampled_point_i = [float(item) for item in sampled_point_i]
                    i_ann['point'] = sampled_point_i
                else:  # at least one of the box size less than 1 pixel
                    if int(boxes[2]) < 1:
                        boxes[2] = boxes[2] + 1
                    if int(boxes[3]) < 1:
                        boxes[3] = boxes[3] + 1
                    mask_i[int(boxes[1]):(int(boxes[1]) + int(boxes[3])),
                    int(boxes[0]):(int(boxes[0]) + int(boxes[2]))] = 1
                    valid_idx = np.where(mask_i == 1)
                    sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
                    sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
                    sampled_point_i = [float(item) for item in sampled_point_i]
                    i_ann['point'] = sampled_point_i
            else:
                i_ann['point'] = []
                print(i_ann['bbox'])
        ith = ith + 1
        if ith % 10000 == 0:
            print(ith)

    data = {}
    data['images'] = imgs
    data['annotations'] = anns
    data['categories'] = list(coco_api.cats.values())
    data['info'] = coco_api.dataset['info']
    data['licenses'] = coco_api.dataset['licenses']
    output_file_label = root_dir + 'instances_train2014_w_indicator_tagsU.json'
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(data, f, sort_keys=True)

    # unlabel part
    json_file = root_dir + 'instances_train2014.json'
    coco_api = COCO(json_file)
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)

    # add indicator
    for i_img in imgs:
        i_img['indicator'] = 0
        i_img['label_type'] = 'pointsK'  # need to assign correct string for different types of annotations, tagsU, pointsK, Unsup
    dataset_anns = [coco_api.imgToAnns[i] for i in img_ids]
    anns = [ann for img_anns in dataset_anns for ann in img_anns]

    ith = 0
    for i_ann in anns:
        mask_i = coco_api.annToMask(i_ann)
        # sample a point
        valid_idx = np.where(mask_i == 1)
        if np.sum(mask_i) > 0:
            sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
            sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
            sampled_point_i = [float(item) for item in sampled_point_i]
            i_ann['point'] = sampled_point_i
        else:
            if len(i_ann['bbox']) > 0:
                boxes = i_ann['bbox']
                boxes = np.array(boxes)
                mask_i[int(boxes[1]):(int(boxes[1]) + int(boxes[3])),
                int(boxes[0]):(int(boxes[0]) + int(boxes[2]))] = 1
                valid_idx = np.where(mask_i == 1)
                if np.sum(mask_i) > 0:
                    sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
                    sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
                    sampled_point_i = [float(item) for item in sampled_point_i]
                    i_ann['point'] = sampled_point_i
                else:  # at least one of the box size less than 1 pixel
                    if int(boxes[2]) < 1:
                        boxes[2] = boxes[2] + 1
                    if int(boxes[3]) < 1:
                        boxes[3] = boxes[3] + 1
                    mask_i[int(boxes[1]):(int(boxes[1]) + int(boxes[3])),
                    int(boxes[0]):(int(boxes[0]) + int(boxes[2]))] = 1
                    valid_idx = np.where(mask_i == 1)
                    sampled_idx = np.random.choice(np.arange(np.size(valid_idx[0])), 1)
                    sampled_point_i = [valid_idx[1][sampled_idx][0], valid_idx[0][sampled_idx][0]]
                    sampled_point_i = [float(item) for item in sampled_point_i]
                    i_ann['point'] = sampled_point_i
            else:
                i_ann['point'] = []
                print(i_ann['bbox'])
        ith = ith + 1
        if ith % 10000 == 0:
            print(ith)

    data = {}
    data['images'] = imgs
    data['annotations'] = anns
    data['categories'] = list(coco_api.cats.values())
    data['info'] = coco_api.dataset['info']
    data['licenses'] = coco_api.dataset['licenses']
    output_file_label = root_dir + 'instances_train2014_w_indicator_pointsK.json'
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(data, f, sort_keys=True)


if __name__ == '__main__':
    main()
    print("finished!")