# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import random
from pycocotools.coco import COCO

def main():

    root_dir = '../bees/'
    data_set = 'instances_bees'
    json_file = root_dir + '{}.json'.format(data_set)
    coco_api = COCO(json_file)
    random_seed = 1709

    ratio_label = 0.08
    ratio_unlabel = 0.72
    ratio_val = 0.2

    random.seed(random_seed)

    img_ids = sorted(coco_api.imgs.keys())
    num_imgs = len(img_ids)
    num_samples = round(num_imgs*ratio_label)
    sample_ids = random.sample(img_ids, num_samples)
    sample_ids = sorted(sample_ids)
    imgs_label = coco_api.loadImgs(sample_ids)

    dataset_anns = [coco_api.imgToAnns[img_id] for img_id in sample_ids]
    anns_label = [ann for img_anns in dataset_anns for ann in img_anns]

    # sample unlabel data
    unsampled_ids = list(set(img_ids) - set(sample_ids))
    unsampled_ids = sorted(unsampled_ids)
    num_samples = round(num_imgs * ratio_unlabel)

    sample_ids = random.sample(unsampled_ids, num_samples)
    sample_ids = sorted(sample_ids)
    imgs_unlabel = coco_api.loadImgs(sample_ids)

    dataset_anns_u = [coco_api.imgToAnns[img_id] for img_id in sample_ids]
    anns_unlabel = [ann for img_anns in dataset_anns_u for ann in img_anns]

    imgs = []
    imgs.extend(imgs_label)
    imgs.extend(imgs_unlabel)

    anns = []
    anns.extend(anns_label)
    anns.extend(anns_unlabel)

    sample_data = {}
    sample_data['images'] = imgs
    sample_data['annotations'] = anns
    sample_data['categories'] = list(coco_api.cats.values())

    output_file_label = '{}{}_train.json'.format(root_dir, data_set)
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)

    # update the rest sample pool
    unsampled_ids = list(set(unsampled_ids) - set(sample_ids))
    unsampled_ids = sorted(unsampled_ids)

    num_samples = round(num_imgs * ratio_val)

    if num_samples > len(unsampled_ids):  #
        num_samples = len(unsampled_ids)

    sample_ids = random.sample(unsampled_ids, num_samples)
    sample_ids = sorted(sample_ids)
    imgs = coco_api.loadImgs(sample_ids)

    # add indicator
    for i_img in imgs:
        i_img['indicator'] = 0
        i_img['label_type'] = 'fully'

    dataset_anns = [coco_api.imgToAnns[img_id] for img_id in sample_ids]
    anns = [ann for img_anns in dataset_anns for ann in img_anns]

    ith = 0
    for i_ann in anns:
        if len(i_ann['bbox']) > 0:
            boxes = i_ann['bbox']
            boxes = np.array(boxes)
            x0 = int(boxes[0])
            y0 = int(boxes[1])
            x1 = int(boxes[0]) + int(boxes[2])
            y1 = int(boxes[1]) + int(boxes[3])
            point_x = random.randint(x0, x1)
            point_y = random.randint(y0, y1)
            i_ann['point'] = [float(point_x), float(point_y)]
        else:
            i_ann['point'] = []
            print(i_ann['bbox'])
        ith = ith + 1
        if ith % 1000 == 0:
            print(ith)

    sample_data = {}
    sample_data['images'] = imgs
    sample_data['annotations'] = anns
    sample_data['categories'] = list(coco_api.cats.values())

    output_file_label = '{}{}_val.json'.format(root_dir, data_set)
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)


if __name__ == '__main__':
    main()
    print("finished!")
