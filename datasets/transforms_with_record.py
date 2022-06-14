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
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.misc import interpolate
from PIL import ImageFilter


def erase(image, target, region):
    img_h, img_w = target['size']
    # img_c = image.shape[-3]
    i, j, h, w, v = region
    # v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
    erased_image = F.erase(image, i, j, h, w, v, inplace=False)

    target = target.copy()
    fields = ["labels", "area", "iscrowd", "boxes", "points"]

    boxes = target["boxes"]
    boxes_xy = box_cxcywh_to_xyxy(boxes)
    i = i / img_h
    j = j / img_w
    h = h / img_h
    w = w / img_w
    keep = ~((boxes_xy[:,0]>j) & (boxes_xy[:,1]>i) & (boxes_xy[:,2]<j+w) & (boxes_xy[:,3]<i+h))
    for field in fields:
        target[field] = target[field][keep]
        # !!! target["area"] is incorrect, but we don't use it
    return erased_image, target

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "points" in target:
        points = target["points"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_points = points - torch.as_tensor([j, i])
        cropped_points = torch.min(cropped_points.reshape(-1, 2), max_size)
        cropped_points = cropped_points.clamp(min=0)

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)
        target["points"] = cropped_points[keep]
        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]  # here the box is xyxy
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    if "points" in target:
        points = target["points"]  # here the box is xyxy
        points = points * torch.as_tensor([-1, 1]) + torch.as_tensor([w, 0])
        target["points"] = points

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "points" in target:
        points = target["points"]
        scaled_points = points * torch.as_tensor([ratio_width, ratio_height])
        target["points"] = scaled_points

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict, record: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        img, target = crop(img, target, region)

        record['RandomSizeCrop'] = region  # i,j,h,w, i,j are not the center, look like is the left top coordinate
        # top (int): Vertical component of the top left corner of the crop box.
        # left (int): Horizontal component of the top left corner of the crop box.
        # height (int): Height of the crop box.
        # width (int): Width of the crop box.

        return img, target, record


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, record):
        if random.random() < self.p:
            record['RandomFlip'] = True
            img, target = hflip(img, target)
            return img, target, record
        return img, target, record


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None, record=None):
        size = random.choice(self.sizes)

        if 'RandomResize_times' in record.keys():
            record['RandomResize_times'] = record['RandomResize_times'] + 1
        else:
            record['RandomResize_times'] = 1

        if 'RandomResize_scale' in record.keys():
            record['RandomResize_scale'].append(size)
        else:
            record['RandomResize_scale'] = [size]
        img, target = resize(img, target, size, self.max_size)
        return img, target, record


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target, record):
        if random.random() < self.p:
            return self.transforms1(img, target, record)
        return self.transforms2(img, target, record)


class ToTensor(object):
    def __call__(self, img, target, record):

        return F.to_tensor(img), target, record


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class RandomErasing1(object):

    def __init__(self):
        self.eraser = T.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random")

    def __call__(self, img, target, record):
        if random.random() < 0.7:
            region = T.RandomErasing.get_params(img, scale=(0.05, 0.2), ratio=(0.3, 3.3))
            img, target = erase(img, target, region)

            record['RandomErasing1'] = region
            return img, target, record

        return img, target, record


class RandomErasing2(object):

    def __init__(self):
        self.eraser = T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random")

    def __call__(self, img, target, record):
        if random.random() < 0.5:
            region = T.RandomErasing.get_params(img, scale=(0.02, 0.2), ratio=(0.1, 6))
            img, target = erase(img, target, region)

            record['RandomErasing2'] = region
            return img, target, record

        return img, target, record


class RandomErasing3(object):

    def __init__(self):
        self.eraser = T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random")

    def __call__(self, img, target, record):
        if random.random() < 0.3:
            region = T.RandomErasing.get_params(img, scale=(0.02, 0.2), ratio=(0.05, 8))
            img, target = erase(img, target, region)

            record['RandomErasing3'] = region
            return img, target, record

        return img, target, record


class RandomColorJiter(object):
    def __init__(self, p=0.8):
        self.p = p
        self.transform = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    def __call__(self, img, target, record):
        if random.random() < self.p:
            return self.transform(img), target, record
        return img, target, record

class RandomGrayScale(object):
    def __init__(self, p=0.2):
        self.p = p
        self.transform = T.Grayscale(num_output_channels=3)

    def __call__(self, img, target, record):
        if random.random() < self.p:
            return self.transform(img), target, record
        return img, target, record


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
        self.transform = T.GaussianBlur(sigma=(0.1, 2.0), kernel_size=(23,23))

    def __call__(self, img, target, record):
        if random.random() < self.p:
            return self.transform(img), target, record
            # return GaussianBlur(img), target, record
        return img, target, record


class RandomContrast(object):
    def __init__(self, p=0.5):
        self.p = p
        self.transform = T.RandomAutocontrast()

    def __call__(self, img, target, record):
        if random.random() < self.p:
            return self.transform(img), target, record
        return img, target, record


class RandomAdjustSharpness(object):
    def __init__(self, p=0.5):
        self.p = p
        self.transform = T.RandomAdjustSharpness(sharpness_factor=2)

    def __call__(self, img, target, record):
        if random.random() < self.p:
            return self.transform(img), target, record
        return img, target, record


class RandomSolarize(object):
    def __init__(self, p=0.5):
        self.p = p
        self.transform = T.RandomSolarize(threshold=192.0)

    def __call__(self, img, target, record):
        if random.random() < self.p:
            return self.transform(img), target, record
        return img, target, record


class RandomPosterize(object):
    def __init__(self, p=0.5):
        self.p = p
        self.transform = T.RandomPosterize(bits=2)

    def __call__(self, img, target, record):
        if random.random() < self.p:
            return self.transform(img), target, record
        return img, target, record


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None, record=None):
        image = F.normalize(image, mean=self.mean, std=self.std)

        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "points" in target:
            points = target["points"]
            points = points / torch.tensor([w, h], dtype=torch.float32)
            target["points"] = points

        return image, target, record


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, record):

        for t in self.transforms:
            image, target, record = t(image, target, record)
        return image, target, record

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x