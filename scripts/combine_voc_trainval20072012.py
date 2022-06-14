# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
from pycocotools.coco import COCO

def main():
    root_dir = '../voc/VOCdevkit/VOC2007trainval/'
    data_set = 'instances_VOC_trainval2007'
    json_file = root_dir + '{}.json'.format(data_set)
    coco_api = COCO(json_file)

    img_ids = sorted(coco_api.imgs.keys())
    num_imgs = len(img_ids)
    num_samples = num_imgs
    sample_ids = random.sample(img_ids, num_samples)
    sample_ids = sorted(sample_ids)
    imgs_2007 = coco_api.loadImgs(sample_ids)

    dataset_anns = [coco_api.imgToAnns[img_id] for img_id in sample_ids]
    anns_2007 = [ann for img_anns in dataset_anns for ann in img_anns]

    root_dir = '../voc/VOCdevkit/VOC2012trainval/'
    data_set = 'instances_VOC_trainval2012'
    json_file = root_dir + '{}.json'.format(data_set)
    coco_api = COCO(json_file)

    img_ids = sorted(coco_api.imgs.keys())
    num_imgs = len(img_ids)
    num_samples = num_imgs
    sample_ids = random.sample(img_ids, num_samples)
    sample_ids = sorted(sample_ids)
    imgs_2012 = coco_api.loadImgs(sample_ids)

    dataset_anns = [coco_api.imgToAnns[img_id] for img_id in sample_ids]
    anns_2012 = [ann for img_anns in dataset_anns for ann in img_anns]

    imgs = []
    imgs.extend(imgs_2007)
    imgs.extend(imgs_2012)

    anns = []
    anns.extend(anns_2007)
    anns.extend(anns_2012)

    sample_data = {}
    sample_data['images'] = imgs
    sample_data['annotations'] = anns
    sample_data['categories'] = list(coco_api.cats.values())

    root_dir_new = '../voc/VOCdevkit/VOC20072012trainval/'
    data_set_name = 'instances_VOC_trainval20072012'
    output_file_label = '{}{}.json'.format(root_dir_new, data_set_name)
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)


if __name__ == '__main__':
    main()
    print("finished!")
