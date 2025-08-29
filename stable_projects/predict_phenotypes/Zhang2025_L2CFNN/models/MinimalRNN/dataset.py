#!/usr/bin/env python
# encoding: utf-8
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import numpy as np
import torch
from torch.utils.data import Dataset


def to_categorical(y, nb_classes=3):
    """Convert list of labels to one-hot vectors"""
    if len(y.shape) == 2:
        y = y.squeeze(1)

    ret_mat = np.full((len(y), nb_classes), np.nan)
    good = ~np.isnan(y)

    ret_mat[good] = 0
    ret_mat[good, y[good].astype(int)] = 1.0

    return ret_mat


class TrainDataset(Dataset):
    """Dataset class for training data, note ground truth and mask sequences
        are input shifted to the right by one position due to the autoregressive
        nature of RNN

    Args:
        data_dict: dictionary containing data

    Return (__getitem__ only returns one sample):
        i_cat (tensor): input time series for diagnosis
        i_val (tensor): input time series for other features
        truth (tensor): ground truth time series for all features
        mask (tensor): boolean mask indicating ground truth availability
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.remove_1tp_subj()
        self.subject_list = sorted(list(self.data_dict.keys()))

    def __len__(self):
        return len(self.subject_list)

    def remove_1tp_subj(self):
        self.data_dict = {
            k: v for k, v in self.data_dict.items() if v["input"].shape[0] > 1
        }

    def __getitem__(self, idx):
        rid = self.subject_list[idx]
        subj_data = self.data_dict[rid]

        x, mask, truth = (
            subj_data["input"],
            subj_data["mask"],
            subj_data["truth"],
        )
        num_tp, C = x.shape
        assert num_tp > 1, print(f"less than 1 tp at {idx}")

        cat_in = to_categorical(x[:, 1])
        i_cat = torch.tensor(cat_in, dtype=torch.float)
        i_val = torch.tensor(x[:, 2:], dtype=torch.float)
        mask = torch.tensor(mask[1:, 1:], dtype=torch.float)
        truth = torch.tensor(truth[1:, 1:], dtype=torch.float)
        return i_cat, i_val, truth, mask


def pad_seq(max_len, current_seq):
    """
    Helper function to pad the sequence to the max length within a batch
    """
    remaining_len = max_len - current_seq.shape[0]
    return torch.cat(
        [current_seq, torch.zeros((remaining_len, current_seq.shape[1]))],
        dim=0,
    )


def pad_collate_train(batch):
    """
    args:
        batch - list of (i_cat, i_val, truth, mask)

    reutrn:
        padded_src - shape (batch, max_src_len, feature_dim)
    """
    # find longest sequence
    max_batch_length = max(map(lambda x: x[0].shape[0], batch))

    # pad according to max_len
    padded_list = list([] for _ in range(4))

    # each item in the batch is a list of 4 tensors
    for current_seq in batch:
        for val_id, value in enumerate(current_seq):
            if val_id < 2:
                padded_list[val_id].append(pad_seq(max_batch_length, value))
            else:
                padded_list[val_id].append(
                    pad_seq(max_batch_length - 1, value)
                )  # -1 because gt/mask is shorter

    out = [
        torch.stack(padded, dim=0).permute(1, 0, 2) for padded in padded_list
    ]

    return out[0], out[1], out[2], out[3]


class TestDataset(Dataset):
    """Dataset class for test data, there's only input

    Args:
        data_dict: dictionary containing data

    Return (__getitem__ only returns one sample):
        i_cat (tensor): input time series for diagnosis
        i_val (tensor): input time series for other features
    """

    def __init__(self, data_dict):

        self.data_dict = data_dict
        # self.remove_1tp_subj()
        self.subject_list = sorted(list(self.data_dict.keys()))

    def __len__(self):
        return len(self.subject_list)

    def remove_1tp_subj(self):
        self.data_dict = {
            k: v for k, v in self.data_dict.items() if v["input"].shape[0] > 2
        }

    def __getitem__(self, idx):
        rid = self.subject_list[idx]
        subj_data = self.data_dict[rid]
        x = subj_data["input"]

        cat_in = to_categorical(x[:, 1])
        i_cat = torch.tensor(cat_in, dtype=torch.float)
        i_val = torch.tensor(x[:, 2:], dtype=torch.float)

        return i_cat, i_val


def pad_collate_test(batch):
    """
    args:
        batch - list of (i_cat, i_val)

    reutrn:
        padded_src - shape (batch, max_src_len, feature_dim)
    """

    # find longest sequence
    max_batch_length = max(map(lambda x: x[0].shape[0], batch))

    # pad according to max_len
    padded_list = list([] for _ in range(3))
    length_list = torch.tensor([v[0].shape[0] for v in batch])
    # each subject in the batch is a sequence of 2 tensors
    for current_seq in batch:
        for val_id, value in enumerate(current_seq):
            if val_id < 2:
                padded_list[val_id].append(pad_seq(max_batch_length, value))
            else:
                padded_list[val_id].append(value)

    out = [
        torch.stack(padded, dim=0).permute(1, 0, 2)
        for padded in padded_list[:-1]
    ]

    return out[0], out[1], length_list
