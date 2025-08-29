#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """Dataset class for training data

    Args:
        data_dict: dictionary containing data

    Return (__getitem__ only returns one sample):
        x (tensor): input L2C features (dim=101)
        y (tensor): ground truth target values (diagnosis, mmse, ventricle)
        m (tensor): mask indicating whether there's ground truth (same order as y)
    """

    def __init__(self, data_dict):
        self.input = data_dict["input"]
        self.truth = data_dict["truth"]
        self.mask = data_dict["mask"]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        x = self.input[idx, :-1]  # last dim is RID
        if x.dtype != np.float:
            x = x.astype(np.float)
        y = self.truth[idx, :]
        m = self.mask[idx, :]

        x = torch.tensor(x, dtype=torch.float)

        return x, y, m


class TestDataset(Dataset):
    """Dataset class for test data

    Args:
        data_dict: dictionary containing data

    Return (__getitem__ only returns one sample):
        x (tensor): input-half L2C features (dim=101)
        Note for test we don't have ground truth or the mask
    """

    def __init__(self, data_dict):
        self.input = data_dict["input"]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        x = self.input[idx, :-1]
        if x.dtype != np.float:
            x = x.astype(np.float)
        x = torch.tensor(x, dtype=torch.float)

        return x
