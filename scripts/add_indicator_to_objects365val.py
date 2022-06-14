# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import random
from pycocotools.coco import COCO

def main():

    root_dir = '../objects365/annotations/'
    json_file = root_dir + 'objects365_val.json'
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
        if ith % 10000 == 0:
            print(ith)

    data = {}
    data['images'] = imgs
    data['annotations'] = anns
    data['categories'] = list(coco_api.cats.values())
    output_file_label = root_dir + 'objects365_val_w_indicator.json'
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(data, f, sort_keys=True)

if __name__ == '__main__':
    main()
    print("finished!")
