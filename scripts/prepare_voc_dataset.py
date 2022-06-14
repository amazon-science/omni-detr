# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import random
from pycocotools.coco import COCO

def main():

    VOC2007trainval = True
    VOC2007test = True
    VOC2012trainval = True

    if VOC2007trainval:
        root_dir = '../voc/VOCdevkit/VOC2007trainval/'
        data_set = 'instances_VOC_trainval2007'
        json_file = root_dir + '{}.json'.format(data_set)
        coco_api = COCO(json_file)

        # original statistics
        num_classes = len(coco_api.cats)

        class_index_list = []
        category = coco_api.cats
        for key, value in category.items():
            class_index_list.append(key)

        histogram = np.zeros((num_classes,), dtype=np.int)

        img_ids = sorted(coco_api.imgs.keys())
        num_imgs = len(img_ids)
        num_samples = num_imgs
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
                boxes[2:] += boxes[:2]
                x0 = boxes[0]
                y0 = boxes[1]
                x1 = boxes[2]
                y1 = boxes[3]
                cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
                mean = (cy, cx)
                cov = [[h / 2, 0], [0, w / 2]]
                sampled_point_i = np.random.multivariate_normal(mean, cov, 1)
                i_ann['point'] = [sampled_point_i[0, 1], sampled_point_i[0, 0]]
            else:
                i_ann['point'] = []
                print(i_ann['bbox'])
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
        print("each class has at least one example ", np.min(histogram)>0)

        sample_data = {}
        sample_data['images'] = imgs
        sample_data['annotations'] = anns
        sample_data['categories'] = list(coco_api.cats.values())

        output_file_label = '{}{}_semi_label.json'.format(root_dir, data_set)
        ## save to json
        with open(output_file_label, 'w') as f:
            print('writing to json output:', output_file_label)
            json.dump(sample_data, f, sort_keys=True)

    if VOC2007test:
        root_dir = '../voc/VOCdevkit/VOC2007test/'
        data_set = 'instances_VOC_test2007'
        json_file = root_dir + '{}.json'.format(data_set)
        coco_api = COCO(json_file)

        # original statistics
        num_classes = len(coco_api.cats)

        class_index_list = []
        category = coco_api.cats
        for key, value in category.items():
            class_index_list.append(key)

        histogram = np.zeros((num_classes,), dtype=np.int)

        img_ids = sorted(coco_api.imgs.keys())
        num_imgs = len(img_ids)
        num_samples = num_imgs
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
                boxes[2:] += boxes[:2]
                x0 = boxes[0]
                y0 = boxes[1]
                x1 = boxes[2]
                y1 = boxes[3]
                cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
                mean = (cy, cx)
                cov = [[h / 2, 0], [0, w / 2]]
                sampled_point_i = np.random.multivariate_normal(mean, cov, 1)
                i_ann['point'] = [sampled_point_i[0, 1], sampled_point_i[0, 0]]
            else:
                i_ann['point'] = []
                print(i_ann['bbox'])
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

        sample_data = {}
        sample_data['images'] = imgs
        sample_data['annotations'] = anns
        sample_data['categories'] = list(coco_api.cats.values())

        output_file_label = '{}{}.json'.format(root_dir, data_set)
        ## save to json
        with open(output_file_label, 'w') as f:
            print('writing to json output:', output_file_label)
            json.dump(sample_data, f, sort_keys=True)

    if VOC2012trainval:
        root_dir = '../voc/VOCdevkit/VOC2012trainval/'
        data_set = 'instances_VOC_trainval2012'
        json_file = root_dir + '{}.json'.format(data_set)
        coco_api = COCO(json_file)

        # original statistics
        num_classes = len(coco_api.cats)

        class_index_list = []
        category = coco_api.cats
        for key, value in category.items():
            class_index_list.append(key)

        histogram = np.zeros((num_classes,), dtype=np.int)

        img_ids = sorted(coco_api.imgs.keys())
        num_imgs = len(img_ids)
        num_samples = num_imgs
        sample_ids = random.sample(img_ids, num_samples)
        sample_ids = sorted(sample_ids)
        imgs = coco_api.loadImgs(sample_ids)

        # add indicator
        for i_img in imgs:
            i_img['indicator'] = 0
            i_img['label_type'] = 'Unsup'

        dataset_anns = [coco_api.imgToAnns[img_id] for img_id in sample_ids]
        anns = [ann for img_anns in dataset_anns for ann in img_anns]

        ith = 0
        for i_ann in anns:
            if len(i_ann['bbox']) > 0:
                boxes = i_ann['bbox']
                boxes = np.array(boxes)
                boxes[2:] += boxes[:2]
                x0 = boxes[0]
                y0 = boxes[1]
                x1 = boxes[2]
                y1 = boxes[3]
                cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
                mean = (cy, cx)
                cov = [[h / 2, 0], [0, w / 2]]
                sampled_point_i = np.random.multivariate_normal(mean, cov, 1)
                i_ann['point'] = [sampled_point_i[0, 1], sampled_point_i[0, 0]]
            else:
                i_ann['point'] = []
                print(i_ann['bbox'])
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

        sample_data = {}
        sample_data['images'] = imgs
        sample_data['annotations'] = anns
        sample_data['categories'] = list(coco_api.cats.values())

        output_file_label = '{}{}_semi_unlabel.json'.format(root_dir, data_set)
        ## save to json
        with open(output_file_label, 'w') as f:
            print('writing to json output:', output_file_label)
            json.dump(sample_data, f, sort_keys=True)

if __name__ == '__main__':
    main()
    print("finished!")
