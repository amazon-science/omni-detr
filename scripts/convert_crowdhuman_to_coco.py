# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import json
import cv2
import shutil

DATA_PATH = '../crowdhuman/'
OUT_PATH = DATA_PATH
SPLITS = ['val', 'train']
DEBUG = False


def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH + split
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'person'}]}
        ann_path = DATA_PATH + 'annotation_{}.odgt'.format(split)
        anns_data = load_func(ann_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for ann_data in anns_data:
            image_cnt += 1

            file_name_org = '{}.jpg'.format(ann_data['ID'])
            file_name = file_name_org.replace(',', '_')
            img_path = DATA_PATH + 'Images/' + file_name
            if not os.path.exists(img_path):
                img_path_org = DATA_PATH + 'Images/' + file_name_org
                shutil.move(img_path_org, img_path)
            print(img_path)
            img = cv2.imread(img_path)
            dimensions = img.shape

            image_info = {'file_name': file_name, 'height': dimensions[0], 'width': dimensions[1], 'id': image_cnt}

            out['images'].append(image_info)
            if split != 'test':
                anns = ann_data['gtboxes']
                for i in range(len(anns)):
                    ann_cnt += 1
                    fbox = anns[i]['fbox']
                    ann = {'id': ann_cnt, 'category_id': 1, 'image_id': image_cnt, 'bbox_vis': anns[i]['vbox'],
                           'bbox': fbox, 'area': fbox[2] * fbox[3],
                           'iscrowd': 1 if 'extra' in anns[i] and
                                           'ignore' in anns[i]['extra'] and
                                           anns[i]['extra']['ignore'] == 1 else 0}
                    out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(
          split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
