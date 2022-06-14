# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
from pycocotools.coco import COCO

def main():

    # we also add point annotation to the validation set, although not used in the code
    root_dir = '../coco/annotations/'
    data_set = 'instances_val2017'
    json_file = root_dir + '{}.json'.format(data_set)
    coco_api = COCO(json_file)
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)

    # add indicator
    for i_img in imgs:
        i_img['indicator'] = 0
        i_img['label_type'] = 'fully'
    dataset_anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
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

    sample_data = {}
    sample_data['images'] = imgs
    sample_data['annotations'] = anns
    sample_data['categories'] = list(coco_api.cats.values())
    sample_data['info'] = coco_api.dataset['info']
    sample_data['licenses'] = coco_api.dataset['licenses']

    output_file_label = root_dir + 'instances_w_indicator_val2017.json'
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)

if __name__ == '__main__':
    main()
    print("finished!")
