# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import random
import cv2
from pycocotools.coco import COCO

def main():

    root_dir = '../crowdhuman/'
    data_set = 'train'
    json_file = root_dir + '{}.json'.format(data_set)
    coco_api = COCO(json_file)
    catIds = coco_api.getCatIds(catNms=['person'])
    imgIds = coco_api.getImgIds(catIds=catIds)
    imgIds.sort()

    imgs = coco_api.loadImgs(imgIds)
    dataset_anns = [coco_api.imgToAnns[img_id] for img_id in imgIds]
    anns = [ann for img_anns in dataset_anns for ann in img_anns]

    anns_new = []
    for i_ann in anns:
        if i_ann['category_id'] == 1:
            anns_new.append(i_ann)
    anns = anns_new

    id_to_downscale = []
    ratio_to_downscale = []
    ith = 0
    for i_img in imgs:
        if i_img['width'] > 600 or i_img['height'] > 600:
            width = i_img['width']
            height = i_img['height']
            if (width <= height and width == 600) or (height <= width and height == 600):
                oh = height
                ow = width
            if width < height:
                ow = 600
                oh = int(600 * height / width)
            else:
                oh = 600
                ow = int(600 * width / height)

            id_to_downscale.append(i_img['id'])
            original_img = cv2.imread(root_dir + 'Images/' + i_img['file_name'])
            resized_img = cv2.resize(original_img, (ow, oh))
            ratios = [float(ow) / float(width), float(oh) / float(height)]
            ratio_width, ratio_height = ratios
            cv2.imwrite(root_dir + 'Images/' + i_img['file_name'], resized_img)
            ratio_to_downscale.append([ratio_width, ratio_height])
        ith = ith + 1
        if ith % 10 == 0:
            print(ith)
    jth = 0
    for i_ann in anns:
        if i_ann['image_id'] in id_to_downscale:
            ith = id_to_downscale.index(i_ann['image_id'])
            ratio_width, ratio_height = ratio_to_downscale[ith]
            box_i = i_ann['bbox']
            boxes = np.array(box_i)
            boxes[2:] += boxes[:2]
            xmin = boxes[0]
            ymin = boxes[1]
            xmax = boxes[2]
            ymax = boxes[3]
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            i_ann['bbox'] = [xmin, ymin, o_width, o_height]
            i_ann['area'] = o_width * o_height
        jth = jth + 1
        if jth % 1000 == 0:
            print(jth)

    sample_data = {}
    sample_data['images'] = imgs
    sample_data['annotations'] = anns
    sample_data['categories'] = [list(coco_api.cats.values())[0]]

    output_file_label = '{}{}_fullbody.json'.format(root_dir, data_set)
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)

    data_set = 'val'
    json_file = root_dir + '{}.json'.format(data_set)
    coco_api = COCO(json_file)

    catIds = coco_api.getCatIds(catNms=['person'])
    imgIds = coco_api.getImgIds(catIds=catIds)
    imgIds.sort()
    imgs = coco_api.loadImgs(imgIds)
    dataset_anns = [coco_api.imgToAnns[img_id] for img_id in imgIds]
    anns = [ann for img_anns in dataset_anns for ann in img_anns]

    # add indicator
    for i_img in imgs:
        i_img['indicator'] = 1
        i_img['label_type'] = 'fully'

    anns_new = []
    for i_ann in anns:
        if i_ann['category_id'] == 1:
            anns_new.append(i_ann)
    anns = anns_new

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
    sample_data['categories'] = [list(coco_api.cats.values())[0]]
    output_file_label = '{}{}_fullbody.json'.format(root_dir, 'test')
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)

if __name__ == '__main__':
    main()
    print("finished!")
