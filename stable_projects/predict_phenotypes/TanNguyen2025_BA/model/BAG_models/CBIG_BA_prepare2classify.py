'''
Written by Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script prepares brain age gap features for AD classification, and is used by
the BAG model. This script merges diagnosis information with brain age gap predictions 
for multiple datasets (e.g., ADNI, AIBL, MACC), across multiple train/val/test splits.

It generates:
1. Numpy arrays of shape [#scans x 2+] for train, val, and test sets per dataset.
2. Combined arrays from all datasets (i.e., "all_datasets") for each split.
3. Development (dev) sets by merging train + val sets.

Expected output(s):
1. *_id_dx_bag.npy               : [#scans x 5] Numpy arrays per dataset and train/val/test split,
                                   with diagnosis and features.
2. *_all_datasets_id_dx_bag.npy  : Combined arrays across all datasets per train/val/test split.
3. dev_all_datasets_id_dx_bag.npy: Train+val development set for each split.

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m model.BAG_models.CBIG_BA_prepare2classify --in_dir /not/real/path --out_dir /not/real/path \
        --task ad_classification --full_trainval_size 997;
'''

import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError("ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from CBIG_BA_config import global_config
from model.cnn.CBIG_BA_train import set_random_seeds
from model.cnn.modules.CBIG_BA_sfcn import SFCN_FC
from model.cnn.modules.CBIG_BA_helper import init_weights_bias
sys.path.append(global_config.PYMENT_DIR)
from pyment.models import RegressionSFCN
from utils.CBIG_BA_complete_step import generate_complete_flag


def generate_bag_input(matched_dir, in_base_dir, in_dir, out_base_dir, seeds,
                       task, dataset, processid_col, dx_col):
    """
    Merges diagnosis and predicted BAG features for each scan from a given dataset,
    and saves the result as Numpy arrays for train/val/test splits across seeds.

    Parameters:
        matched_dir (str): Directory containing diagnosis-balanced CSVs per dataset and task.
        in_base_dir (str): Directory with reference train/val/test CSVs.
        in_dir (str): Directory containing BAG prediction results per dataset.
        out_base_dir (str): Output directory to save *_id_dx_bag.npy files.
        seeds (int): Number of train/val/test splits (random seeds).
        task (str): Classification task ('ad_classification').
        dataset (str): Dataset name ('ADNI', 'AIBL', or 'MACC').
        processid_col (str): Column name for unique scan/process ID.
        dx_col (str): Column name for diagnosis label.
    """

    splits = ['train', 'val', 'test']

    trainval_size = max_trainval_size

    # df_balanced, df_id_features, df_balanced_id_features, df_id_dx, df_features, and df_id_dx_bag
    # comprise all scans from a dataset.

    # df_balanced contains scan ID + dx
    df_balanced = pd.read_csv('{}/{}/{}_balanced.csv'.format(
        matched_dir, task, dataset))

    # df_id_features contains scan ID + features
    df_id_features = pd.read_csv('{}/{}/ID_bag.csv'.format(in_dir, dataset))
    df_id_features = df_id_features[['id', 'sex', 'age', 'pred_age', 'bag']]
    df_id_features = df_id_features.rename(columns={'id': processid_col})

    # df_balanced_id_features contains scan ID + dx + features
    df_balanced_id_features = df_balanced.merge(df_id_features,
                                                on=processid_col,
                                                how='right')

    # df_id_dx contains scan ID + dx (with renamed dx column)
    df_id_dx = df_balanced_id_features[[processid_col, dx_col]]

    # df_features contains features
    df_features = df_balanced_id_features[['sex', 'age', 'pred_age', 'bag']]

    # df_id_dx_bag contains scan ID + dx (with renamed dx column) + features
    df_id_dx_bag = pd.concat([df_id_dx, df_features], axis=1)
    for seed in range(seeds):
        out_dir = os.path.join(out_base_dir, dataset, str(trainval_size),
                               'seed_{}'.format(seed))
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        # df_reference_split, and df_splits comprise only scans from a dataset, for a specific split
        for split in splits:
            # if full train+val size or test split, then point to full data split folder
            if (trainval_size == max_trainval_size) or (split == 'test'):
                df_reference_split = pd.read_csv(
                    '{}/{}/full/seed_{}/{}.csv'.format(in_base_dir, task, seed,
                                                       split))
            # else, point to corresponding non-full train+val size data split folder
            else:
                df_reference_split = pd.read_csv('{}/{}/{}/seed_{}/{}.csv'.format(in_base_dir, task, \
                    trainval_size, seed, split))

            # df_reference_split contains scan ID for dataset, for split
            df_reference_split = df_reference_split[
                df_reference_split['DATASET'] == dataset]

            # df_splits contains dx + features for dataset, for split, df_splits has additional scan ID column
            df_splits = df_id_dx_bag.merge(df_reference_split[[processid_col]],
                                           on=processid_col,
                                           how='right')
            np.save('{}/{}_id_dx_bag'.format(out_dir, split), df_splits)

def concat_all_datasets(out_base_dir, seeds):
    """
    Concatenates *_id_dx_bag.npy files from AIBL, ADNI, and MACC datasets
    into a single array per split and saves them in an "all_datasets" folder.

    Parameters:
        out_base_dir (str): Directory containing individual dataset *_id_dx_bag.npy files.
        seeds (int): Number of train/val/test splits (random seeds).
    """

    splits=['train','val','test']

    # generate list of train+val sizes in descending order
    trainval_size = max_trainval_size

    for seed in range(seeds):
        out_dir = os.path.join(out_base_dir, 'all_datasets', str(trainval_size), 'seed_{}'.format(seed))
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        for split in splits:
            aibl_arr = np.load('{}/AIBL/{}/seed_{}/{}_id_dx_bag.npy'.format(out_base_dir, trainval_size, \
                seed, split), allow_pickle=True)
            adni_arr = np.load('{}/ADNI/{}/seed_{}/{}_id_dx_bag.npy'.format(out_base_dir, trainval_size, \
                seed, split), allow_pickle=True)
            macc_arr = np.load('{}/MACC/{}/seed_{}/{}_id_dx_bag.npy'.format(out_base_dir, trainval_size, \
                seed, split), allow_pickle=True)
            all_datasets_arr = np.vstack((aibl_arr, adni_arr, macc_arr))
            np.save('{}/{}_all_datasets_id_dx_bag'.format(out_dir, split), all_datasets_arr)


def concat_trainval2dev(out_base_dir, seeds):
    """
    Concatenates the 'train' and 'val' arrays for each split to form a
    development set. Saves as dev_all_datasets_id_dx_bag.npy.

    Parameters:
        out_base_dir (str): Directory containing all_datasets train/val arrays.
        seeds (int): Number of train/val/test splits (random seeds).
    """
    splits = ['train', 'val']

    # generate list of train+val sizes in descending order
    trainval_size = max_trainval_size

    for seed in range(seeds):
        out_dir = os.path.join(out_base_dir, 'all_datasets',
                               str(trainval_size), 'seed_{}'.format(seed))
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        for split in splits:
            train_arr = np.load(
                '{}/all_datasets/{}/seed_{}/train_all_datasets_id_dx_bag.npy'.
                format(out_base_dir, trainval_size, seed),
                allow_pickle=True)
            val_arr = np.load('{}/all_datasets/{}/seed_{}/val_all_datasets_id_dx_bag.npy'.format(out_base_dir, \
                trainval_size, seed), allow_pickle=True)
            dev_arr = np.vstack((train_arr, val_arr))
            np.save('{}/dev_all_datasets_id_dx_bag'.format(out_dir), dev_arr)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['ad_classification', 'mci_progression'])
    parser.add_argument('--num_split', type=int, default=50)
    parser.add_argument('--full_trainval_size', type=int)
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    # Define input arguments
    args = get_args()
    datasets = global_config.DATASETS
    seeds = args.num_split
    max_trainval_size = args.full_trainval_size
    processid_col = global_config.PROCESSID_COL
    dx_col = global_config.DX_COL
    matched_dir = global_config.MATCHED_DATA_DIR
    in_base_dir = global_config.DATA_SPLIT_DIR
    in_dir = args.in_dir
    out_base_dir = os.path.join(args.out_dir)
    task = args.task

    # For each dataset, generate BAG array
    for dataset in datasets:
        generate_bag_input(matched_dir, in_base_dir, in_dir, out_base_dir, seeds, task, dataset, processid_col, dx_col)
    # Concatenate BAG arrays across datasets
    concat_all_datasets(out_base_dir, seeds)
    # Concatenate train+val splits into development split
    concat_trainval2dev(out_base_dir, seeds)

    # Generate complete flags for next step to commence
    generate_complete_flag(out_base_dir, 'BAG_prepare2classify')
