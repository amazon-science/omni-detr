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

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, train_one_epoch_semi, train_one_epoch_burnin
from models import build_model, build_model_semi
from collections import OrderedDict


def get_args_parser():
    parser = argparse.ArgumentParser('Omni-DETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=400, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file',
                        default='coco_omni')  # coco_omni, coco_35to80_tagsU, coco_35to80_point, coco_objects_tagsU, coco_objects_points, bees_omni, voc_semi_ voc_omni, objects_omni, crowdhuman_omni
    parser.add_argument('--data_path', default='./coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='results_tmp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--percent', default='10',
                        help='percent with fully labeled')
    parser.add_argument('--BURN_IN_STEP', default=20, type=int,
                        help='as the name means')
    parser.add_argument('--TEACHER_UPDATE_ITER', default=1, type=int,
                        help='as the name means')
    parser.add_argument('--EMA_KEEP_RATE', default=0.9996, type=float,
                        help='as the name means')
    parser.add_argument('--annotation_json_label', default='',
                        help='percent with fully labeled')
    parser.add_argument('--annotation_json_unlabel', default='',
                        help='percent with fully labeled')
    parser.add_argument('--CONFIDENCE_THRESHOLD', default=0.7, type=float,
                        help='as the name means')
    parser.add_argument('--PASSED_NUM', default='passed_num.txt',
                        help='the sample number passing the threshold each batch')
    parser.add_argument('--pixels', default=600, type=int,
                        help='as the name means')
    parser.add_argument('--w_p', default=0.5, type=float,
                        help='weight for point cost matrix during filtering')
    parser.add_argument('--w_t', default=0.5, type=float,
                        help='weight for tag cost matrix during filtering')
    parser.add_argument('--save_freq', default=5, type=int,
                        metavar='N', help='checkpoint save frequency (default: 5)')
    parser.add_argument('--eval_freq', default=1, type=int)
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset_file == "coco":
        model, criterion, postprocessors = build_model(args)
    elif args.dataset_file == "coco_omni" or args.dataset_file == 'coco_add_semi' or args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point" or args.dataset_file == 'coco_objects_tagsU' or args.dataset_file == 'coco_objects_points' or args.dataset_file == 'bees_omni' or args.dataset_file == 'voc_semi' or args.dataset_file == 'voc_omni' or args.dataset_file == 'objects_omni' or args.dataset_file == 'crowdhuman_omni':
        model_student, model_teacher, criterion, criterion_burnin, postprocessors = build_model_semi(args)
    model_student.to(device)
    model_teacher.to(device)

    model_without_ddp_student = model_student
    model_without_ddp_teacher = model_teacher
    n_parameters = sum(p.numel() for p in model_student.parameters() if p.requires_grad)
    print('number of student model params:', n_parameters)

    n_parameters = sum(p.numel() for p in model_teacher.parameters() if p.requires_grad)
    print('number of teacher model params:', n_parameters)

    if args.dataset_file == "coco":
        dataset_train = build_dataset(image_set='train', label=True, args=args)
    elif args.dataset_file == "coco_omni" or args.dataset_file == 'coco_add_semi' or args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point" or args.dataset_file == 'coco_objects_tagsU' or args.dataset_file == 'coco_objects_points' or args.dataset_file == 'bees_omni' or args.dataset_file == 'voc_semi' or args.dataset_file == 'voc_omni' or args.dataset_file == 'objects_omni' or args.dataset_file == 'crowdhuman_omni':
        dataset_train_label = build_dataset(image_set='train', label=True, args=args)
        dataset_train_unlabel = build_dataset(image_set='train', label=False, args=args)
        dataset_train_burnin = build_dataset(image_set='burnin', label=True, args=args)
    dataset_val = build_dataset(image_set='val', label=True, args=args)

    if args.dataset_file == "coco":
        if args.distributed:
            if args.cache_mode:
                sampler_train = samplers.NodeDistributedSampler(dataset_train)
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_train = samplers.DistributedSampler(dataset_train)
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    elif args.dataset_file == "coco_omni" or args.dataset_file == 'coco_add_semi' or args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point" or args.dataset_file == 'coco_objects_tagsU' or args.dataset_file == 'coco_objects_points' or args.dataset_file == 'bees_omni' or args.dataset_file == 'voc_semi' or args.dataset_file == 'voc_omni' or args.dataset_file == 'objects_omni' or args.dataset_file == 'crowdhuman_omni':
        if args.distributed:
            if args.cache_mode:
                sampler_train_label = samplers.NodeDistributedSampler(dataset_train_label)
                sampler_train_unlabel = samplers.NodeDistributedSampler(dataset_train_unlabel)
                sampler_train_burnin = samplers.NodeDistributedSampler(dataset_train_burnin)
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_train_label = samplers.DistributedSampler(dataset_train_label)
                sampler_train_unlabel = samplers.DistributedSampler(dataset_train_unlabel)
                sampler_train_burnin = samplers.DistributedSampler(dataset_train_burnin)
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train_label = torch.utils.data.RandomSampler(dataset_train_label)
            sampler_train_unlabel = torch.utils.data.RandomSampler(dataset_train_unlabel)
            sampler_train_burnin = torch.utils.data.RandomSampler(dataset_train_burnin)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.dataset_file == "coco":
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
    elif args.dataset_file == "coco_omni" or args.dataset_file == 'coco_add_semi' or args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point" or args.dataset_file == 'coco_objects_tagsU' or args.dataset_file == 'coco_objects_points' or args.dataset_file == 'bees_omni' or args.dataset_file == 'voc_semi' or args.dataset_file == 'voc_omni' or args.dataset_file == 'objects_omni' or args.dataset_file == 'crowdhuman_omni':
        batch_sampler_train_label = torch.utils.data.BatchSampler(
            sampler_train_label, args.batch_size, drop_last=True)
        batch_sampler_train_unlabel = torch.utils.data.BatchSampler(
            sampler_train_unlabel, args.batch_size, drop_last=True)
        batch_sampler_train_burnin = torch.utils.data.BatchSampler(
            sampler_train_burnin, 2, drop_last=True)

    if args.dataset_file == "coco":
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                       pin_memory=True)
    elif args.dataset_file == "coco_omni" or args.dataset_file == 'coco_add_semi' or args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point" or args.dataset_file == 'coco_objects_tagsU' or args.dataset_file == 'coco_objects_points' or args.dataset_file == 'bees_omni' or args.dataset_file == 'voc_semi' or args.dataset_file == 'voc_omni' or args.dataset_file == 'objects_omni' or args.dataset_file == 'crowdhuman_omni':
        data_loader_train_label = DataLoader(dataset_train_label, batch_sampler=batch_sampler_train_label,
                                             collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                             pin_memory=True)
        data_loader_train_unlabel = DataLoader(dataset_train_unlabel, batch_sampler=batch_sampler_train_unlabel,
                                               collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                               pin_memory=True)
        data_loader_train_burnin = DataLoader(dataset_train_burnin, batch_sampler=batch_sampler_train_burnin,
                                              collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                              pin_memory=True)
    data_loader_val = DataLoader(dataset_val, 2, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp_student.named_parameters():
        print(n)

    for n, p in model_without_ddp_teacher.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp_student.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,
                                                                                                   args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp_student.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp_student.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
        optimizer_burnin = torch.optim.SGD(param_dicts, lr=2e-4, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        optimizer_burnin = torch.optim.AdamW(param_dicts, lr=2e-4, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    lr_scheduler_burnin = torch.optim.lr_scheduler.StepLR(optimizer_burnin, args.lr_drop)

    if args.distributed:
        model_student = torch.nn.parallel.DistributedDataParallel(model_student, device_ids=[args.gpu])
        model_without_ddp_student = model_student.module
        model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu])
        model_without_ddp_teacher = model_teacher.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp_student.detr.load_state_dict(checkpoint['model_student'])
        model_without_ddp_teacher.detr.load_state_dict(checkpoint['model_teacher'])
    TEACHER_LOADED = False
    OPTIMIZOR_LOADED = False
    output_dir = Path(args.output_dir)
    if args.resume:
        print("Resume from checkpoint {}".format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'model' in checkpoint.keys():  # means load burn in model
            missing_keys, unexpected_keys = model_without_ddp_student.load_state_dict(checkpoint['model'], strict=False)
            TEACHER_LOADED = False
        elif 'model_student' in checkpoint.keys() and 'model_teacher' not in checkpoint.keys():
            missing_keys, unexpected_keys = model_without_ddp_student.load_state_dict(
                checkpoint['model_student'], strict=False)
            TEACHER_LOADED = False
        else:  # load previous trained student and teacher models
            missing_keys, unexpected_keys = model_without_ddp_student.load_state_dict(
                checkpoint['model_student'], strict=False)
            missing_keys_teacher, unexpected_keys_teacher = model_without_ddp_teacher.load_state_dict(
                checkpoint['model_teacher'], strict=False)
            unexpected_keys_teacher = [k for k in unexpected_keys_teacher if
                                       not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys_teacher) > 0:
                print('Missing Keys of the teacher: {}'.format(missing_keys_teacher))
            if len(unexpected_keys_teacher) > 0:
                print('Unexpected Keys of the teacher: {}'.format(unexpected_keys_teacher))
            TEACHER_LOADED = True

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            if 'model' in checkpoint.keys():  # means load burn in model
                optimizer_burnin.load_state_dict(checkpoint['optimizer'])
                lr_scheduler_burnin.load_state_dict(checkpoint['lr_scheduler'])
            else:
                import copy
                p_groups = copy.deepcopy(optimizer.param_groups)
                optimizer.load_state_dict(checkpoint['optimizer'])
                for pg, pg_old in zip(optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old['initial_lr']
                # print(optimizer.param_groups)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                args.override_resumed_lr_drop = True
                if args.override_resumed_lr_drop:
                    print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                    lr_scheduler.step_size = args.lr_drop
                    lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                lr_scheduler.step(lr_scheduler.last_epoch)
                OPTIMIZOR_LOADED = True
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model_student, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        if args.dataset_file == "coco":
            if args.distributed:
                sampler_train.set_epoch(epoch)
        elif args.dataset_file == "coco_omni" or args.dataset_file == 'coco_add_semi' or args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point" or args.dataset_file == 'coco_objects_tagsU' or args.dataset_file == 'coco_objects_points' or args.dataset_file == 'bees_omni' or args.dataset_file == 'voc_semi' or args.dataset_file == 'voc_omni' or args.dataset_file == 'objects_omni' or args.dataset_file == 'crowdhuman_omni':
            if args.distributed:
                sampler_train_label.set_epoch(epoch)
                sampler_train_unlabel.set_epoch(epoch)
                sampler_train_burnin.set_epoch(epoch)

        if epoch < args.BURN_IN_STEP:
            train_stats = train_one_epoch_burnin(
                model_student, criterion_burnin, data_loader_train_burnin, optimizer_burnin, device, epoch,
                args.clip_max_norm)
            lr_scheduler_burnin.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every save_freq epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_freq == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    print('-------saving model--------')

                    utils.save_on_master({
                        'model': model_without_ddp_student.state_dict(),
                        'optimizer': optimizer_burnin.state_dict(),
                        'lr_scheduler': lr_scheduler_burnin.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            if (epoch + 1) % args.eval_freq == 0:

                test_stats, coco_evaluator = evaluate(
                    model_student, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
                )

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}

                if args.output_dir and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

        else:
            if epoch >= args.BURN_IN_STEP:
                if TEACHER_LOADED == False:
                    print('!!sucessfully transfer the student model!!')
                    # update and copy the whole model
                    keep_rate = 0.00
                    student_model_dict = {
                        key: value for key, value in model_student.state_dict().items()
                    }
                    new_teacher_dict = OrderedDict()
                    for key, value in model_teacher.state_dict().items():
                        if key in student_model_dict.keys():
                            new_teacher_dict[key] = (
                                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                            )
                    model_teacher.load_state_dict(new_teacher_dict)
                    TEACHER_LOADED = True

                    if not OPTIMIZOR_LOADED:
                        import copy
                        p_groups = copy.deepcopy(optimizer.param_groups)
                        optimizer = optimizer_burnin
                        for pg, pg_old in zip(optimizer.param_groups, p_groups):
                            pg['lr'] = pg_old['lr']
                            pg['initial_lr'] = pg_old['initial_lr']
                        # print(optimizer.param_groups)
                        lr_scheduler = lr_scheduler_burnin
                        # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                        args.override_resumed_lr_drop = True
                        if args.override_resumed_lr_drop:
                            print(
                                'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                            lr_scheduler.step_size = args.lr_drop
                            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                        lr_scheduler.step(lr_scheduler.last_epoch)
                        OPTIMIZOR_LOADED = True

            # generate the pseudo label using teacher model, update the annotation of unlabeled data and train
            train_stats = train_one_epoch_semi(
                model_student, model_teacher, criterion, data_loader_train_label, data_loader_train_unlabel, optimizer,
                device, epoch,
                args.clip_max_norm, args.CONFIDENCE_THRESHOLD, args.EMA_KEEP_RATE, args.PASSED_NUM, args.dataset_file,
                args.pixels, args.w_p, args.w_t)

            lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every save_freq epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_freq == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    print('-------saving model--------')

                    utils.save_on_master({
                        'model_student': model_without_ddp_student.state_dict(),
                        'model_teacher': model_without_ddp_teacher.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            if (epoch + 1) % args.eval_freq == 0:

                test_stats_student, coco_evaluator = evaluate(
                    model_student, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
                )

                test_stats_teacher, coco_evaluator = evaluate(
                    model_teacher, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
                )

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_stu_{k}': v for k, v in test_stats_student.items()},
                             **{f'test_tea_{k}': v for k, v in test_stats_teacher.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}

                if args.output_dir and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Omni-DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

