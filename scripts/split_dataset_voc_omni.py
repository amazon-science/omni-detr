# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import random
from pycocotools.coco import COCO

def main():

    root_dir = '../voc/VOCdevkit/VOC20072012trainval/'
    data_set = 'instances_VOC_trainval20072012'
    json_file = root_dir + '{}.json'.format(data_set)
    coco_api = COCO(json_file)
    random_seed = 1709

    # assign the ratio, the sum should be 1
    fully_labeled = 0.2
    Unsup = 0.19
    tagsU = 0.0
    tagsK = 0.0
    pointsU = 0.0
    pointsK = 0.0
    boxesEC = 0.61
    boxesU = 0.0
    assert sum([fully_labeled, Unsup, tagsU, tagsK, pointsU, pointsK, boxesEC, boxesU]) == 1.0

    # we first sample the fully label data
    # original statistics
    num_classes = len(coco_api.cats)
    class_index_list = []
    category = coco_api.cats
    for key, value in category.items():
        class_index_list.append(key)
    histogram = np.zeros((num_classes,), dtype=np.int)

    img_ids = sorted(coco_api.imgs.keys())
    num_imgs = len(img_ids)

    random.seed(random_seed)

    According_to_num = False  # if it is False, we split by percentage
    if According_to_num:
        num_samples = 578
    else:
        ratio = fully_labeled
        num_samples = round(num_imgs * ratio)

    sample_ids = random.sample(img_ids, num_samples)
    sample_ids = sorted(sample_ids)
    imgs = coco_api.loadImgs(sample_ids)

    # add indicator
    for i_img in imgs:
        i_img['indicator'] = 1
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


    # sampled statistics
    classes = [x["category_id"] for x in anns]
    for i in classes:
        index = class_index_list.index(int(i))
        histogram[index] = histogram[index] + 1
    class_ratios = histogram / np.sum(histogram)
    print("sampled class ratios: {}".format(class_ratios))
    print("each class has at least one example ", np.min(histogram) > 0)

    sample_data = {}
    sample_data['images'] = imgs
    sample_data['annotations'] = anns
    sample_data['categories'] = list(coco_api.cats.values())

    output_file_label = '{}{}_voc_omni_label_seed{}_{}fully{}Unsup{}tagsU{}tagsK{}pointsU{}pointsK{}boxesEC{}boxesU.json'.format(root_dir, data_set, random_seed, round(100*fully_labeled), round(100*Unsup), round(100*tagsU), round(100*tagsK), round(100*pointsU), round(100*pointsK), round(100*boxesEC), round(100*boxesU))
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)

    # next deal with unlabel (weakly label) data
    unsampled_ids = list(set(img_ids) - set(sample_ids))
    unsampled_ids = sorted(unsampled_ids)

    splitting = {}
    imgs_all = []
    anns_all = []
    if According_to_num:
        splitting['tagsU'] = 2500
        splitting['tagsK'] = 1250
        splitting['points'] = 2255
        splitting['Unsup'] = 5417
        splitting['pointsOnly'] = 1111
        splitting['boxesN'] = 1111
        splitting['boxes'] = 1111
    else:
        splitting['tagsU'] = round(num_imgs * tagsU)
        splitting['tagsK'] = round(num_imgs * tagsK)
        splitting['pointsU'] = round(num_imgs * pointsU)
        splitting['pointsK'] = round(num_imgs * pointsK)
        splitting['Unsup'] = round(num_imgs * Unsup)
        splitting['boxesEC'] = round(num_imgs * boxesEC)
        splitting['boxesU'] = round(num_imgs * boxesU)

    for key, value in splitting.items():
        histogram = np.zeros((num_classes,), dtype=np.int)
        num_samples = value
        if num_samples > len(unsampled_ids):  #
            num_samples = len(unsampled_ids)

        sample_ids = random.sample(unsampled_ids, num_samples)
        sample_ids = sorted(sample_ids)
        imgs = coco_api.loadImgs(sample_ids)

        # add indicator
        for i_img in imgs:
            i_img['indicator'] = 0
            i_img['label_type'] = key

        dataset_anns_u = [coco_api.imgToAnns[img_id] for img_id in sample_ids]
        anns = [ann for img_anns in dataset_anns_u for ann in img_anns]

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

        if key == 'boxesEC':
            # -----the corresponding between delta and mIoU, I got this by emperical experiments
            delta = [25, 9, 3, 1, 0.5, 0.3, 0.15, 0.06, 0.03, 0.01]
            # mIoU = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]

            IoU = np.load('IoU_extreme.npy')

            # compute the distribution, bin is 0.1
            IoU_list = list(IoU)
            distribution_extreme = []
            for i in np.arange(0.1, 1.1, 0.1):
                bin_i = [x for x in IoU_list if x >= i - 0.1 and x < i]
                distribution_extreme.append(len(bin_i) / len(IoU_list))
            distribution_extreme = distribution_extreme[::-1]

            random.shuffle(anns)
            ith_bin = 0
            cur_delta = delta[ith_bin]
            skip_to_next = distribution_extreme[ith_bin] * len(anns)
            ith = 0
            for i_ann in anns:
                if ith > skip_to_next:
                    ith_bin += 1
                    cur_delta = delta[ith_bin]
                    skip_to_next = skip_to_next + distribution_extreme[ith_bin] * len(anns)
                    print('to next')

                if 'iscrowd' not in i_ann or i_ann['iscrowd'] == 0:
                    box_i = i_ann['bbox']
                    boxes = np.array(box_i)
                    boxes[2:] += boxes[:2]
                    x0 = boxes[0]
                    y0 = boxes[1]
                    x1 = boxes[2]
                    y1 = boxes[3]

                    # add noise to each of the two nodes
                    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
                    mean = (y0, x0)
                    cov = [[h / cur_delta, 0], [0, w / cur_delta]]
                    sampled_point_i = np.random.multivariate_normal(mean, cov, 1)
                    x0_new, y0_new = sampled_point_i[0, 1], sampled_point_i[0, 0]

                    mean = (y1, x1)
                    sampled_point_i = np.random.multivariate_normal(mean, cov, 1)
                    x1_new, y1_new = sampled_point_i[0, 1], sampled_point_i[0, 0]

                    x0 = min(x0_new, x1_new)
                    x1 = max(x0_new, x1_new)
                    y0 = min(y0_new, y1_new)
                    y1 = max(y0_new, y1_new)
                    w = x1 - x0
                    h = y1 - y0


                    x0 = float(x0)
                    y0 = float(y0)
                    w = float(w)
                    h = float(h)
                    i_ann['bbox'] = [x0, y0, w, h]
                ith = ith + 1
                if ith % 10000 == 0:
                    print(ith)

        # sampled statistics
        classes = [x["category_id"] for x in anns]
        for i in classes:
            index = class_index_list.index(int(i))
            histogram[index] = histogram[index] + 1
        class_ratios = histogram / np.sum(histogram)
        print("sampled class ratios: {}".format(class_ratios))
        print("each class has at least one example ", np.min(histogram) > 0)

        imgs_all.extend(imgs)
        anns_all.extend(anns)

        # update the rest sample pool
        unsampled_ids = list(set(unsampled_ids) - set(sample_ids))
        unsampled_ids = sorted(unsampled_ids)

    unsample_data = {}
    unsample_data['images'] = imgs_all
    unsample_data['annotations'] = anns_all
    unsample_data['categories'] = list(coco_api.cats.values())

    output_file_unlabel = '{}{}_voc_omni_unlabel_seed{}_{}fully{}Unsup{}tagsU{}tagsK{}pointsU{}pointsK{}boxesEC{}boxesU.json'.format(root_dir, data_set, random_seed, round(100*fully_labeled), round(100*Unsup), round(100*tagsU), round(100*tagsK), round(100*pointsU), round(100*pointsK), round(100*boxesEC), round(100*boxesU))
    ## save to json
    with open(output_file_unlabel, 'w') as f:
        print('writing to json output:', output_file_unlabel)
        json.dump(unsample_data, f, sort_keys=True)


if __name__ == '__main__':
    main()
    print("finished!")
