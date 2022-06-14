# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import random
from pycocotools.coco import COCO

def main():

    root_dir = '../objects365/annotations/'
    data_set = 'objects365_train'
    json_file = root_dir + '{}.json'.format(data_set)
    coco_api = COCO(json_file)
    random_seed = 1709
    maximum_example_per_category = 300

    random.seed(random_seed)

    class_index_list = []  # 1->365
    category = coco_api.cats
    category_list = []
    for key, value in category.items():
        class_index_list.append(key)
        category_list.append(value['name'])

    # for each category we sample some examples
    category_count = 0
    all_satisfied_imgID = []
    for i_category in category_list:
        catIds = coco_api.getCatIds(catNms=[i_category])
        imgIds_i = coco_api.getImgIds(catIds=catIds)
        augmented_list = list(set(imgIds_i) - set(all_satisfied_imgID))
        if len(augmented_list) > maximum_example_per_category:
            augmented_list = random.sample(augmented_list, maximum_example_per_category)
        all_satisfied_imgID.extend(augmented_list)
        category_count += 1

    print(category_count)

    # original statistics
    num_classes = len(coco_api.cats)
    histogram = np.zeros((num_classes,), dtype=np.int)
    random.seed(random_seed)

    sample_ids = sorted(all_satisfied_imgID)
    imgs = coco_api.loadImgs(sample_ids)

    dataset_anns = [coco_api.imgToAnns[img_id] for img_id in sample_ids]
    anns = [ann for img_anns in dataset_anns for ann in img_anns]

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

    output_file_label = '{}{}_sampled.json'.format(root_dir, data_set)
    ## save to json
    with open(output_file_label, 'w') as f:
        print('writing to json output:', output_file_label)
        json.dump(sample_data, f, sort_keys=True)


if __name__ == '__main__':
    main()
    print("finished!")
