# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import xml.etree.ElementTree as ET
import glob
import cv2

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = {'bees': 0}
    bnd_id = START_BOUNDING_BOX_ID
    i_th = 0
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename_original = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename_original = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        # image_id = get_filename_as_int(filename)
        filename = filename_original.split('_')
        filename = filename[-1]
        filename = filename.split('.')
        image_id = int(filename[0])
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)

        i_th = i_th + 1
        if i_th % 10 == 0:
            print(i_th)

        # we rescale the image if its size greater than 800, because in this dataset, the image is too big, for weak aug, we can't accept such big image because of memory issue
        if width > 600 or height > 600:
            if (width <= height and width == 600) or (height <= width and height == 600):
                oh = height
                ow = width
            if width < height:
                ow = 600
                oh = int(600 * height / width)
            else:
                oh = 600
                ow = int(600 * width / height)

            original_img = cv2.imread('../bees/ML-Data/' + filename_original)
            resized_img = cv2.resize(original_img, (ow, oh))
            ratios = [float(ow)/float(width), float(oh)/float(height)]
            ratio_width, ratio_height = ratios
            cv2.imwrite('../bees/ML-Data/' + filename_original, resized_img)

            image = {
                "file_name": filename_original,
                "height": oh,
                "width": ow,
                "id": image_id,
            }
            json_dict["images"].append(image)
            ## Currently we do not support segmentation.
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            for obj in get(root, "object"):
                category_id = 0
                bndbox = get_and_check(obj, "bndbox", 1)
                xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
                # ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
                ymin = int(float(get_and_check(bndbox, "ymin", 1).text)) - 1
                xmax = int(get_and_check(bndbox, "xmax", 1).text)
                ymax = int(get_and_check(bndbox, "ymax", 1).text)
                assert xmax > xmin
                assert ymax > ymin
                xmin = xmin * ratio_width
                xmax = xmax * ratio_width
                ymin = ymin * ratio_height
                ymax = ymax * ratio_height
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)

                ann = {
                    "area": o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "category_id": category_id,
                    "id": bnd_id,
                    "ignore": 0,
                    "segmentation": [],
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1
        else:
            image = {
                "file_name": filename_original,
                "height": height,
                "width": width,
                "id": image_id,
            }
            json_dict["images"].append(image)
            ## Currently we do not support segmentation.
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            for obj in get(root, "object"):
                category_id = 0
                bndbox = get_and_check(obj, "bndbox", 1)
                xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
                # ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
                ymin = int(float(get_and_check(bndbox, "ymin", 1).text)) - 1
                xmax = int(get_and_check(bndbox, "xmax", 1).text)
                ymax = int(get_and_check(bndbox, "ymax", 1).text)
                assert xmax > xmin
                assert ymax > ymin
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {
                    "area": o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "category_id": category_id,
                    "id": bnd_id,
                    "ignore": 0,
                    "segmentation": [],
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1

    cat = {"supercategory": "none", "id": 0, "name": 'bees'}
    json_dict["categories"].append(cat)
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format."
    )

    parser.add_argument('--xml_dir', default="../bees/ML-Data/",
                        help="Directory path to xml files.", type=str)
    parser.add_argument('--json_file',
                        default="../bees/instances_bees.json",
                        help="Output COCO format json file.", type=str)

    args = parser.parse_args()
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))

    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, args.json_file)
    print("Success: {}".format(args.json_file))
