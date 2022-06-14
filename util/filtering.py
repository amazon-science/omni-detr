# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
import scipy.optimize
import torch.nn as nn


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (ow, oh)

def box_calibration(cur_boxes, cur_labels, cur_idx, records_unlabel_q, records_unlabel_k, pixels):
    # cur_boxes, num * [x, y, w, h]
    if pixels == 600:
        max_pixels = 1000
    elif pixels == 800:
        max_pixels = 1333

    records_unlabel_q = records_unlabel_q[0]
    records_unlabel_k = records_unlabel_k[0]

    # we first recover the bbox coordinate for weak aug in the original image width height space
    cur_one_tensor = torch.tensor([1, 0, 0, 0])
    cur_one_tensor = cur_one_tensor.cuda()
    cur_one_tensor = cur_one_tensor.repeat(cur_boxes.shape[0], 1)
    if 'RandomFlip' in records_unlabel_k.keys() and records_unlabel_k['RandomFlip']:
        original_boxes = torch.abs(cur_one_tensor - cur_boxes)
    else:
        original_boxes = cur_boxes
    original_boxes = box_cxcywh_to_xyxy(original_boxes)
    img_w = records_unlabel_k['OriginalImageSize'][1]
    img_h = records_unlabel_k['OriginalImageSize'][0]
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
    scale_fct = scale_fct.cuda()
    scale_fct = scale_fct.repeat(cur_boxes.shape[0], 1)
    original_boxes = original_boxes * scale_fct
    cur_boxes = torch.clone(original_boxes)

    # then, we repeat the boxes generation process in the strong aug for the predicted boxes
    if 'RandomFlip' in records_unlabel_q.keys() and records_unlabel_q['RandomFlip']:
        cur_boxes = cur_boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]).cuda() + torch.as_tensor([img_w, 0, img_w, 0]).cuda()

    if records_unlabel_q['RandomResize_times'] > 1:

        # random resize
        rescaled_size1 = records_unlabel_q['RandomResize_scale'][0]
        rescaled_size1 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size1)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size1, (img_w, img_h)))
        ratio_width, ratio_height = ratios
        cur_boxes = cur_boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).cuda()
        img_w = rescaled_size1[0]
        img_h = rescaled_size1[1]

        # random size crop
        region = records_unlabel_q['RandomSizeCrop']
        i, j, h, w = region
        fields = ["labels", "area", "iscrowd"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32).cuda()
        cropped_boxes = cur_boxes - torch.as_tensor([j, i, j, i]).cuda()
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        cur_boxes = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")
        cropped_boxes = torch.clone(cur_boxes)
        cropped_boxes = cropped_boxes.reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        cur_boxes = cur_boxes[keep]
        cur_labels = cur_labels[keep]
        cur_idx = cur_idx[keep]

        img_w = w
        img_h = h

        # random resize
        rescaled_size2 = records_unlabel_q['RandomResize_scale'][1]
        rescaled_size2 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size2, max_size=max_pixels)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size2, (img_w, img_h)))
        ratio_width, ratio_height = ratios
        cur_boxes = cur_boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).cuda()
        img_w = rescaled_size2[0]
        img_h = rescaled_size2[1]
    else:
        # random resize
        rescaled_size1 = records_unlabel_q['RandomResize_scale'][0]
        rescaled_size1 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size1, max_size=max_pixels)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size1, (img_w, img_h)))
        ratio_width, ratio_height = ratios
        cur_boxes = cur_boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).cuda()
        img_w = rescaled_size1[0]
        img_h = rescaled_size1[1]

    # finally, deal with normalize part in deformable detr aug code
    cur_boxes = box_xyxy_to_cxcywh(cur_boxes)
    cur_boxes = cur_boxes / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()

    # deal with the randomerasing part
    if 'RandomErasing1' in records_unlabel_q.keys():
        region = records_unlabel_q['RandomErasing1']
        i, j, h, w, _ = region
        cur_boxes_xy = box_cxcywh_to_xyxy(cur_boxes)
        i = i / img_h
        j = j / img_w
        h = h / img_h
        w = w / img_w
        keep = ~((cur_boxes_xy[:, 0] > j) & (cur_boxes_xy[:, 1] > i) & (cur_boxes_xy[:, 2] < j + w) & (cur_boxes_xy[:, 3] < i + h))
        cur_boxes = cur_boxes[keep]
        cur_labels = cur_labels[keep]
        cur_idx = cur_idx[keep]
    if 'RandomErasing2' in records_unlabel_q.keys():
        region = records_unlabel_q['RandomErasing2']
        i, j, h, w, _ = region
        cur_boxes_xy = box_cxcywh_to_xyxy(cur_boxes)
        i = i / img_h
        j = j / img_w
        h = h / img_h
        w = w / img_w
        keep = ~((cur_boxes_xy[:, 0] > j) & (cur_boxes_xy[:, 1] > i) & (cur_boxes_xy[:, 2] < j + w) & (cur_boxes_xy[:, 3] < i + h))
        cur_boxes = cur_boxes[keep]
        cur_labels = cur_labels[keep]
        cur_idx = cur_idx[keep]
    if 'RandomErasing3' in records_unlabel_q.keys():
        region = records_unlabel_q['RandomErasing3']
        i, j, h, w, _ = region
        cur_boxes_xy = box_cxcywh_to_xyxy(cur_boxes)
        i = i / img_h
        j = j / img_w
        h = h / img_h
        w = w / img_w
        keep = ~((cur_boxes_xy[:, 0] > j) & (cur_boxes_xy[:, 1] > i) & (cur_boxes_xy[:, 2] < j + w) & (cur_boxes_xy[:, 3] < i + h))
        cur_boxes = cur_boxes[keep]
        cur_labels = cur_labels[keep]
        cur_idx = cur_idx[keep]

    updated_boxes = cur_boxes
    updated_labels = cur_labels
    assert updated_boxes.shape[0] == updated_labels.shape[0]
    return updated_boxes, updated_labels, cur_idx


def unified_filter_pseudo_labels(pseudo_unsup_outputs, targets_unlabel_q, targets_unlabel_k, records_unlabel_q, records_unlabel_k, pixels, label_type, w_p=0.5, w_t=0.5, is_binary=False, threshold=0.7):
    pseudo_unsup_logits = pseudo_unsup_outputs['pred_logits']
    softmax = nn.Softmax(dim=2)
    pseudo_unsup_prob = softmax(pseudo_unsup_logits)

    if is_binary:
        satisfied_idx_d2 = (pseudo_unsup_prob[0, :, 0] > threshold).nonzero(as_tuple=True)
        satisfied_idx_d1 = torch.zeros(satisfied_idx_d2[0].shape[0], dtype=torch.long).cuda()
        satisfied_idx = (satisfied_idx_d1, satisfied_idx_d2[0])
        pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
        satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
        satisfied_class = torch.zeros(satisfied_bbox.shape[0], dtype=torch.long).cuda()
    else:
        if label_type == 'Unsup':
            classes_scores, classes_indices = torch.max(pseudo_unsup_prob, dim=2)
            satisfied_idx = (classes_scores > threshold).nonzero(as_tuple=True)
            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
            satisfied_class = classes_indices[satisfied_idx[0], satisfied_idx[1]]
        elif label_type == 'tagsU':
            targets_gt_unlabel_q = targets_unlabel_k[0]['labels']
            targets_gt_unlabel_q = torch.unique(targets_gt_unlabel_q)

            # the first thing we do is predicting count number for each class
            # get the class: object number
            targets_gt_unlabel_q_list = targets_gt_unlabel_q.tolist()
            targets_gt_unlabel_q_new = torch.zeros(1, dtype=torch.long).cuda()
            for i_c in targets_gt_unlabel_q_list:
                classes_scores_i = pseudo_unsup_prob[0, :, i_c]
                satisfied_idx_i = (classes_scores_i > threshold).nonzero(as_tuple=True)
                if len(satisfied_idx_i[0]) > 0:
                    targets_gt_unlabel_q_new = torch.cat(
                        (targets_gt_unlabel_q_new, i_c * torch.ones(len(satisfied_idx_i[0]), dtype=torch.long).cuda()))
                else:
                    targets_gt_unlabel_q_new = torch.cat(
                        (targets_gt_unlabel_q_new, i_c * torch.ones(1, dtype=torch.long).cuda()))

            targets_gt_unlabel_q = targets_gt_unlabel_q_new[1:]

            # Compute the distance cost between predictions and tags, we use 1 - confidence score as the distance measurement
            dist_bbox_tags = torch.zeros((pseudo_unsup_prob.shape[1], targets_gt_unlabel_q.shape[0])).cuda()
            for i in range(targets_gt_unlabel_q.shape[0]):
                dist_bbox_tags[:, i] = 1 - pseudo_unsup_prob[0, :, targets_gt_unlabel_q[i]]
            dist_bbox_tags = dist_bbox_tags.cpu()

            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            indices = scipy.optimize.linear_sum_assignment(dist_bbox_tags)
            updated_idx_d1 = torch.zeros(targets_gt_unlabel_q.shape[0], dtype=torch.long).cuda()
            updated_idx_d2_1 = indices[0].tolist()
            updated_idx_d2_2 = indices[1].tolist()
            updated_idx_d2 = [x for _, x in sorted(zip(updated_idx_d2_2, updated_idx_d2_1))]
            updated_idx_d2 = torch.tensor(updated_idx_d2, dtype=torch.long).cuda()
            satisfied_idx = (updated_idx_d1, updated_idx_d2)
            satisfied_class = targets_gt_unlabel_q
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
        elif label_type == 'tagsK':
            targets_gt_unlabel_q = targets_unlabel_k[0]['labels']

            dist_bbox_tags = torch.zeros((pseudo_unsup_prob.shape[1], targets_gt_unlabel_q.shape[0])).cuda()

            for i in range(targets_gt_unlabel_q.shape[0]):
                dist_bbox_tags[:, i] = 1 - pseudo_unsup_prob[0, :, targets_gt_unlabel_q[i]]
            dist_bbox_tags = dist_bbox_tags.cpu()

            indices = scipy.optimize.linear_sum_assignment(dist_bbox_tags)

            updated_idx_d1 = torch.zeros(targets_gt_unlabel_q.shape[0], dtype=torch.long).cuda()
            updated_idx_d2_1 = indices[0].tolist()
            updated_idx_d2_2 = indices[1].tolist()
            updated_idx_d2 = [x for _, x in sorted(zip(updated_idx_d2_2, updated_idx_d2_1))]
            updated_idx_d2 = torch.tensor(updated_idx_d2, dtype=torch.long).cuda()

            satisfied_idx = (updated_idx_d1, updated_idx_d2)
            satisfied_class = targets_gt_unlabel_q
            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
        elif label_type == 'pointsU' or label_type == 'pointsK':
            targets_gt_unlabel_q = targets_unlabel_k[0]['labels']

            # Compute the distance cost between predictions and tags, we use 1 - confidence score as the distance measurement
            dist_bbox_tags = torch.zeros((pseudo_unsup_prob.shape[1], targets_gt_unlabel_q.shape[0])).cuda()

            for i in range(targets_gt_unlabel_q.shape[0]):
                dist_bbox_tags[:, i] = 1 - pseudo_unsup_prob[0, :, targets_gt_unlabel_q[i]]

            dist_bbox_tags = dist_bbox_tags.cpu()

            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            targets_gt_point_unlabel_q = targets_unlabel_k[0]['points']

            classes_scores, classes_indices = torch.max(pseudo_unsup_prob, dim=2)
            matrix_w_high_score = classes_scores[0]
            matrix_w_high_score = torch.unsqueeze(matrix_w_high_score, 1)
            matrix_w_high_score = 1 - matrix_w_high_score

            # to get an indicator matrix of which prediction a ground truth is included
            indicator_include_matrix = torch.zeros((pseudo_unsup_prob.shape[1], targets_gt_point_unlabel_q.shape[0]),
                                                   dtype=torch.long).cuda()
            for i in range(targets_gt_point_unlabel_q.shape[0]):
                point_i = targets_gt_point_unlabel_q[i, :2]
                x0 = point_i[0]
                y0 = point_i[1]
                keep_i = (pseudo_unsup_bbox[0, :, 0] - pseudo_unsup_bbox[0, :, 2] / 2 < x0) & (
                            pseudo_unsup_bbox[0, :, 0] + pseudo_unsup_bbox[0, :, 2] / 2 > x0) & (
                                     pseudo_unsup_bbox[0, :, 1] - pseudo_unsup_bbox[0, :, 3] / 2 < y0) & (
                                     pseudo_unsup_bbox[0, :, 1] + pseudo_unsup_bbox[0, :, 3] / 2 > y0)
                keep_i = keep_i.int()
                keep_i = keep_i.type(torch.cuda.LongTensor)
                if keep_i.sum() > 0:
                    indicator_include_matrix[keep_i == 1, i] = 1
            indicator_include_matrix[indicator_include_matrix < 1] = 1e3  # just give a random large number

            # Compute the distance cost between boxes
            dist_bbox_point = torch.cdist(pseudo_unsup_bbox[0][:, :2], targets_gt_point_unlabel_q[:, :2], p=2)

            if dist_bbox_point.shape[1] > 0:
                dist_bbox_point = dist_bbox_point - dist_bbox_point.min()
                dist_bbox_point = dist_bbox_point / torch.max(dist_bbox_point)
            dist_bbox_point = (dist_bbox_point + matrix_w_high_score) * indicator_include_matrix
            dist_bbox_point = dist_bbox_point.cpu()
            dist_bbox_point = w_p * dist_bbox_point + w_t * dist_bbox_tags

            indices = scipy.optimize.linear_sum_assignment(dist_bbox_point)
            updated_idx_d1 = torch.zeros(targets_gt_unlabel_q.shape[0], dtype=torch.long).cuda()
            updated_idx_d2_1 = indices[0].tolist()
            updated_idx_d2_2 = indices[1].tolist()
            updated_idx_d2 = [x for _, x in sorted(zip(updated_idx_d2_2, updated_idx_d2_1))]
            updated_idx_d2 = torch.tensor(updated_idx_d2, dtype=torch.long).cuda()

            satisfied_idx = (updated_idx_d1, updated_idx_d2)
            if w_t == 0:
                satisfied_class = classes_indices[satisfied_idx[0], satisfied_idx[1]]
            else:
                satisfied_class = targets_gt_unlabel_q
            satisfied_bbox = pseudo_unsup_bbox[satisfied_idx[0], satisfied_idx[1], :]
            satisfied_points = targets_gt_point_unlabel_q
        else:  # boxesEC or boxesU
            pseudo_unsup_bbox = pseudo_unsup_outputs['pred_boxes']
            targets_gt_box_unlabel_q = targets_unlabel_k[0]['boxes']

            classes_scores, classes_indices = torch.max(pseudo_unsup_prob, dim=2)
            matrix_w_high_score = classes_scores[0]
            matrix_w_high_score = torch.unsqueeze(matrix_w_high_score, 1)
            matrix_w_high_score = 1 - matrix_w_high_score

            # Compute the distance cost between boxes
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(pseudo_unsup_bbox[0], targets_gt_box_unlabel_q, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pseudo_unsup_bbox[0]),
                                             box_cxcywh_to_xyxy(targets_gt_box_unlabel_q))
            dist_bbox = 5 * cost_bbox + 2 * cost_giou
            if dist_bbox.shape[1] > 0:
                dist_bbox = dist_bbox - dist_bbox.min()
                dist_bbox = dist_bbox / torch.max(dist_bbox)
            dist_bbox = dist_bbox + matrix_w_high_score
            dist_bbox = dist_bbox.cpu()

            indices = scipy.optimize.linear_sum_assignment(dist_bbox)

            updated_idx_d1 = torch.zeros(targets_gt_box_unlabel_q.shape[0], dtype=torch.long).cuda()
            updated_idx_d2_1 = indices[0].tolist()
            updated_idx_d2_2 = indices[1].tolist()
            updated_idx_d2 = [x for _, x in sorted(zip(updated_idx_d2_2, updated_idx_d2_1))]
            updated_idx_d2 = torch.tensor(updated_idx_d2, dtype=torch.long).cuda()

            satisfied_idx = (updated_idx_d1, updated_idx_d2)
            satisfied_class = classes_indices[satisfied_idx[0], satisfied_idx[1]]
            satisfied_bbox = targets_gt_box_unlabel_q

    pseudo_annotations = []
    for i in range(pseudo_unsup_bbox.shape[0]):
        i_target = targets_unlabel_q[i]
        i_dict = {}
        i_idx = (satisfied_idx[0] == i).nonzero(as_tuple=True)

        if len(i_idx[0]) == 0:  # means no predicted bbox satisfies the confidence threshold
            # just assign []
            for key, value in i_target.items():
                if key == 'boxes':
                    i_dict['boxes'] = torch.empty((0), dtype=torch.int64, device='cuda')
                elif key == 'labels':
                    i_dict['labels'] = torch.empty((0), dtype=torch.int64, device='cuda')
                else:
                    i_dict[key] = i_target[key]
        else:
            i_boxes = satisfied_bbox[i_idx[0]]
            i_labels = satisfied_class[i_idx[0]]
            cur_i_idx = i_idx[0]
            i_boxes, i_labels, cur_i_idx = box_calibration(i_boxes, i_labels, cur_i_idx, records_unlabel_q,
                                                           records_unlabel_k, pixels)
            for key, value in i_target.items():
                if key == 'boxes':
                    i_dict['boxes'] = i_boxes
                elif key == 'labels':
                    # the labels also need to calibration
                    i_dict['labels'] = i_labels
                else:
                    i_dict[key] = i_target[key]
        pseudo_annotations.append(i_dict)
    return pseudo_annotations

