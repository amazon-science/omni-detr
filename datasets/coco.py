# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from .torchvision_datasets import CocoDetection as TvCocoDetection
from .torchvision_datasets import CocoDetection_semi as TvCocoDetection_semi
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import datasets.transforms_with_record as Tr


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class CocoDetection_semi(TvCocoDetection_semi):
    def __init__(self, img_folder, ann_file, transforms_strong, transforms_weak, return_masks, cache_mode=False, local_rank=0,
                 local_size=1):
        super(CocoDetection_semi, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms_strong = transforms_strong
        self._transforms_weak = transforms_weak
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target, indicator, labeltype = super(CocoDetection_semi, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms_strong is not None:
            record_q = {}
            record_q['OriginalImageSize'] = [img.height, img.width]
            img_q, target_q, record_q = self._transforms_strong(img, target, record_q)
        if self._transforms_weak is not None:
            record_k = {}
            record_k['OriginalImageSize'] = [img.height, img.width]
            img_k, target_k, record_k = self._transforms_weak(img, target, record_k)
        return img_q, target_q, record_q, img_k, target_k, record_k, indicator, labeltype



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        points = [obj["point"] for obj in anno]
        points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        points = points[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}

        target["boxes"] = boxes
        target["labels"] = classes
        target["points"] = points
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # for a small of number samples on objects365, the instance number > 300, so we simply only pick up the first 300 annotaitons. 300 is the query number
        if target["boxes"].shape[0] > 300:
            fields = ["boxes", "labels", "area", "points", "iscrowd"]
            for field in fields:
                target[field] = target[field][:300]

        return image, target


def make_coco_transforms600(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [int(i * 600 / 800) for i in scales]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1000),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1000),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_strong_transforms_with_record(image_set):

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            Tr.RandomColorJiter(),
            Tr.RandomGrayScale(),
            Tr.RandomGaussianBlur(),
            # Tr.RandomContrast(),
            # Tr.RandomAdjustSharpness(),
            # Tr.RandomPosterize(),
            # Tr.RandomSolarize(),
            Tr.RandomSelect(
                Tr.RandomResize(scales, max_size=1333),
                Tr.Compose([
                    Tr.RandomResize([400, 500, 600]),
                    Tr.RandomSizeCrop(384, 600),
                    Tr.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
            Tr.RandomErasing1(),
            Tr.RandomErasing2(),
            Tr.RandomErasing3(),
        ])


    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_weak_transforms_with_record(image_set):

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_strong_transforms_with_record600(image_set):

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [int(i*600/800) for i in scales]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            Tr.RandomColorJiter(),
            Tr.RandomGrayScale(),
            Tr.RandomGaussianBlur(),
            # Tr.RandomContrast(),
            # Tr.RandomAdjustSharpness(),
            # Tr.RandomPosterize(),
            # Tr.RandomSolarize(),
            Tr.RandomSelect(
                Tr.RandomResize(scales, max_size=1000),
                Tr.Compose([
                    Tr.RandomResize([400, 500, 600]),
                    Tr.RandomSizeCrop(384, 600),
                    Tr.RandomResize(scales, max_size=1000),
                ])
            ),
            normalize,
            Tr.RandomErasing1(),
            Tr.RandomErasing2(),
            Tr.RandomErasing3(),
        ])


    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_weak_transforms_with_record600(image_set):

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if args.dataset_file == "voc_omni":
        PATHS = {
            "train": (root / "VOCdevkit/VOC20072012trainval/JPEGImages", root / "VOCdevkit/VOC20072012trainval" / args.annotation_json_label),
            "val": (root / "VOCdevkit/VOC2007test/JPEGImages", root / "VOCdevkit/VOC2007test" / 'instances_VOC_test2007.json'),
        }
    elif args.dataset_file == "voc_semi":
        PATHS = {
            "train": (root / "VOCdevkit/VOC2007trainval/JPEGImages", root / "VOCdevkit/VOC2007trainval" / args.annotation_json_label),
            "val": (root / "VOCdevkit/VOC2007test/JPEGImages", root / "VOCdevkit/VOC2007test" / 'instances_VOC_test2007.json'),
        }
    elif args.dataset_file == "bees_omni":
        PATHS = {
            "train": (root / "ML-Data", root / args.annotation_json_label),
            "val": (root / "ML-Data", root / 'instances_bees_val.json'),
        }
    elif args.dataset_file == "crowdhuman_omni":
        PATHS = {
            "train": (root / "Images", root / args.annotation_json_label),
            "val": (root / "Images", root / 'test_fullbody.json'),
        }
    elif args.dataset_file == "objects_omni":
        PATHS = {
            "train": (root / "train_objects365", root / "annotations" / args.annotation_json_label),
            "val": (root / "val_objects365", root / "annotations" / 'objects365_val_w_indicator.json'),
        }
    elif args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point":
        PATHS = {
            "train": (root / "val2014", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2017", root / "annotations" / 'instances_w_indicator_val2017.json'),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2017", root / "annotations" / 'instances_w_indicator_val2017.json'),
        }

    img_folder, ann_file = PATHS[image_set]
    if args.pixels == 600:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms600(image_set), return_masks=args.masks,
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    elif args.pixels == 800:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset


def build_semi_label(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'

    if args.dataset_file == "voc_semi":
        PATHS = {
            "train": (root / "VOCdevkit/VOC2007trainval/JPEGImages", root / "VOCdevkit/VOC2007trainval" / args.annotation_json_label),
            "val": (root / "OCdevkit/VOC2007test/JPEGImages", root / "VOCdevkit/VOC2007test" / 'instances_VOC_test2007.json'),
        }
    elif args.dataset_file == "voc_omni":
        PATHS = {
            "train": (root / "VOCdevkit/VOC20072012trainval/JPEGImages", root / "VOCdevkit/VOC20072012trainval" / args.annotation_json_label),
            "val": (root / "val2014", root / "annotations" / f'{mode}_val2014.json'),
        }
    elif args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point":
        PATHS = {
            "train": (root / "val2014", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2014", root / "annotations" / f'{mode}_val2014.json'),
        }
    elif args.dataset_file == "bees_omni":
        PATHS = {
            "train": (root / "ML-Data", root / args.annotation_json_label),
            "val": (root / "ML-Data", root / 'instances_bees_val.json'),
        }
    elif args.dataset_file == "crowdhuman_omni":
        PATHS = {
            "train": (root / "Images", root / args.annotation_json_label),
            "val": (root / "Images", root / 'test_fullbody.json'),
        }
    elif args.dataset_file == "objects_omni":
        PATHS = {
            "train": (root / "train_objects365", root / "annotations" / args.annotation_json_label),
            "val": (root / "val_objects365", root / "annotations" / 'objects365_val_w_indicator.json'),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }

    img_folder, ann_file = PATHS[image_set]
    if args.pixels == 600:
        dataset = CocoDetection_semi(img_folder, ann_file,
                                     transforms_strong=make_coco_strong_transforms_with_record600(image_set),
                                     transforms_weak=make_coco_weak_transforms_with_record600(image_set),
                                     return_masks=args.masks,
                                     cache_mode=args.cache_mode, local_rank=get_local_rank(),
                                     local_size=get_local_size())

    elif args.pixels == 800:
        dataset = CocoDetection_semi(img_folder, ann_file, transforms_strong=make_coco_strong_transforms_with_record(image_set), transforms_weak=make_coco_weak_transforms_with_record(image_set), return_masks=args.masks,
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())

    return dataset

def build_semi_unlabel(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if args.dataset_file == "voc_semi":
        PATHS = {
            "train": (root / "VOCdevkit/VOC2012trainval/JPEGImages", root / "VOCdevkit/VOC2012trainval" / args.annotation_json_unlabel),
            "val": (root / "OCdevkit/VOC2007test/JPEGImages", root / "VOCdevkit/VOC2007test" / 'instances_VOC_test2007.json'),
        }
    elif args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point":
        PATHS = {
            "train": (root / "train2014", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2014", root / "annotations" / f'{mode}_val2014.json'),
        }
    elif args.dataset_file == "voc_omni":
        PATHS = {
            "train": (root / "VOCdevkit/VOC20072012trainval/JPEGImages", root / "VOCdevkit/VOC20072012trainval" / args.annotation_json_unlabel),
            "val": (root / "val2014", root / "annotations" / f'{mode}_val2014.json'),
        }
    elif args.dataset_file == "coco_add_semi":
        PATHS = {
            "train": (root / "unlabeled2017", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }
    elif args.dataset_file == "crowdhuman_omni":
        PATHS = {
            "train": (root / "Images", root / args.annotation_json_unlabel),
            "val": (root / "Images", root / 'test_fullbody.json'),
        }
    elif args.dataset_file == "bees_omni":
        PATHS = {
            "train": (root / "ML-Data", root / args.annotation_json_unlabel),
            "val": (root / "ML-Data", root / 'instances_bees_val.json'),
        }
    elif args.dataset_file == "coco_objects_tagsU" or args.dataset_file == "coco_objects_points":
        PATHS = {
            "train": (root / "train_objects365", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }
    elif args.dataset_file == "objects_omni":
        PATHS = {
            "train": (root / "train_objects365", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val_objects365", root / "annotations" / 'objects365_val_w_indicator.json'),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }

    img_folder, ann_file = PATHS[image_set]

    if args.pixels == 600:
        dataset = CocoDetection_semi(img_folder, ann_file,
                                     transforms_strong=make_coco_strong_transforms_with_record600(image_set),
                                     transforms_weak=make_coco_weak_transforms_with_record600(image_set),
                                     return_masks=args.masks,
                                     cache_mode=args.cache_mode, local_rank=get_local_rank(),
                                     local_size=get_local_size())

    elif args.pixels == 800:
        dataset = CocoDetection_semi(img_folder, ann_file, transforms_strong=make_coco_strong_transforms_with_record(image_set), transforms_weak=make_coco_weak_transforms_with_record(image_set), return_masks=args.masks,
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())

    return dataset
