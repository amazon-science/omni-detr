# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets


def to_cuda_semi(samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes, device):
    samples_q = samples_q.to(device, non_blocking=True)
    samples_k = samples_k.to(device, non_blocking=True)
    targets_q = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets_q]
    targets_k = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets_k]
    return samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets



class data_prefetcher_semi():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples_q, self.next_targets_q, self.next_records_q, self.next_samples_k, self.next_targets_k, self.next_records_k, self.next_indicators, self.next_labeltypes = next(self.loader)
        except StopIteration:
            self.next_samples_q = None
            self.next_targets_q = None
            self.next_samples_k = None
            self.next_targets_k = None
            self.next_records_q = None
            self.next_records_k = None
            self.next_indicators = None
            self.next_labeltypes = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples_q, self.next_targets_q, self.next_records_q, self.next_samples_k, self.next_targets_k, self.next_records_k, self.next_indicators, self.next_labeltypes = to_cuda_semi(self.next_samples_q, self.next_targets_q, self.next_records_q, self.next_samples_k, self.next_targets_k, self.next_records_k, self.next_indicators, self.next_labeltypes, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples_q = self.next_samples_q
            targets_q = self.next_targets_q
            records_q = self.next_records_q
            samples_k = self.next_samples_k
            targets_k = self.next_targets_k
            records_k = self.next_records_k
            indicators = self.next_indicators
            labeltypes = self.next_labeltypes
            if samples_q is not None:
                samples_q.record_stream(torch.cuda.current_stream())
            if samples_k is not None:
                samples_k.record_stream(torch.cuda.current_stream())
            if targets_q is not None:
                for t in targets_q:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            if targets_k is not None:
                for t in targets_k:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes = next(self.loader)
                samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes = to_cuda_semi(samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes, self.device)
            except StopIteration:
                samples_q = None
                targets_q = None
                samples_k = None
                targets_k = None
                indicators = None
                labeltypes = None
        return samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes