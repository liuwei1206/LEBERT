# -*- coding: utf-8 -*-
# @Time    : 2020/12/01 11:47
# @Author  : liuwei
# @File    : sampler.py
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
import numpy as np
import torch
import torch.nn as nn
import math
import random
random.seed(106524)

class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, do_shuffle=False):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.do_shuffle = do_shuffle

    def __iter__(self):
        start_pos = self.rank * self.num_samples
        end_pos = (self.rank + 1) * self.num_samples
        indices = list(range(start_pos, end_pos))

        assert len(indices) == self.num_samples
        if self.do_shuffle:
            random.shuffle(indices)
            return iter(indices)
        else:
            return iter(indices)

    def __len__(self):
        return self.num_samples