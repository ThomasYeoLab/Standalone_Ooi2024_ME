"""
Written by Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

Script to generate train, val, and test splits for multiple train+val sizes and seeds,
ensuring participant subsets are properly nested to prevent data leakage. Each smaller
train+val size is a subset of the previous larger one.

Split proportions for datasets (AIBL, ADNI, MACC) and diagnoses (e.g. NCI, AD) are preserved
across varying sizes using controlled downsampling.

Expected output(s):
1. $TANNGUYEN2025_BA_DIR/data/data_split/[task]/[size_trainval]/seed_[i]/{train,val,test}.csv

Example:
    conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR};
    python -m data_split.CBIG_BA_data_split_vary --task ad_classification \
        --num_split 50 --step 50 --full_trainval_size 997
"""

import pandas as pd
import math
import os
import argparse
import logging
import sys

TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from CBIG_BA_config import global_config


def store_percentage_within_seed(size_trainval, num_dataset_split,
                                 pct_dataset_split, datasets, splits,
                                 prev_trainval, sizes_trainval, in_out_dir,
                                 seed, full_train, full_val):
    """
    Calculate and store the proportion of participants from each dataset for a given
    train+val size, per split (train or val), relative to the full dataset.

    This function is used when iterating through decreasing train+val sizes. For each
    size, it records the proportion of samples from each dataset (e.g., AIBL, ADNI, MACC)
    based on the most recent larger train+val size. The goal is to maintain consistent
    dataset proportions across splits as the dataset size shrinks.

    If the previous train+val size is the largest (i.e., full size), proportions are
    calculated from the full dataset. Otherwise, proportions are derived from the
    previous size's split file.

    These proportions are stored in `pct_dataset_split` to guide future sampling
    decisions that preserve dataset distribution consistency.

    Args:
        size_trainval (int): Current total size of the training and validation datasets.
        num_dataset_split (dict): Dict storing the number of participants from each dataset,
                                  for each split, at various train+val sizes.
        pct_dataset_split (dict): Dict storing the proportion of participants from each dataset,
                                  for each split, at various train+val sizes.
        datasets (list): List of dataset names (e.g., ['AIBL', 'ADNI', 'MACC']).
        splits (list): List of data splits, usually ['train', 'val'].
        prev_trainval (int): The next larger train+val size used to derive proportions.
        sizes_trainval (list): List of all train+val sizes, typically in descending order.
        in_out_dir (str): Directory path containing the CSV files for each split and seed.
        seed (int): Random seed for reproducibility.
        full_train (int): Total number of training participants in the full dataset.
        full_val (int): Total number of validation participants in the full dataset.

    Returns:
        dict: Updated `pct_dataset_split` containing proportions per dataset and split
              for the current `size_trainval`.
    """

    num_dataset_split[size_trainval], pct_dataset_split[size_trainval] = {}, {}
    for dataset in datasets:
        num_dataset_split[size_trainval][dataset], pct_dataset_split[
            size_trainval][dataset] = {}, {}
        for split in splits:
            if prev_trainval == sizes_trainval[0]:
                # if previous train+val size was full size, then use full csv
                df_split = pd.read_csv('{}/full/seed_{}/{}.csv'.format(
                    in_out_dir, seed, split))
            else:
                # else use previous train+val size csv
                df_split = pd.read_csv('{}/{}/seed_{}/{}.csv'.format(
                    in_out_dir, prev_trainval, seed, split))

            # computing absolute participant number for train+val size, for dataset, for split
            num_dataset_split[size_trainval][dataset][split] = len(
                df_split[df_split['DATASET'] == dataset])
            num_split = {'train': full_train, 'val': full_val}.get(split)

            # compute percentage for train+val size, for dataset, for split
            pct_dataset_split[size_trainval][dataset][split] = (
                num_dataset_split[size_trainval][dataset][split] / num_split)
    return pct_dataset_split


def assign_floor_dropped(size_trainval, datasets, splits, pct_dataset_split,
                         to_drop_train, to_drop_val):
    """
    Compute the floor (⌊x⌋) number of participants to drop for each dataset and split
    (train/val), based on the proportion stored for the current train+val size.

    For each dataset and split, this function calculates how many participants should be 
    dropped, using the percentage derived from the previously larger train+val size. 
    The value is floored to avoid over-dropping (i.e., ⌊x⌋ is used instead of rounding).

    For example, if the drop percentage for AIBL in the train split is 35.253% and there 
    are 80 AIBL participants in training, then 0.35253 * 80 = 28.2024, which becomes 28 
    participants dropped (floor value).

    Args:
        size_trainval (int): Current train+val size being processed.
        datasets (list): List of dataset names (e.g., ['AIBL', 'ADNI', 'MACC']).
        splits (list): List of data splits, typically ['train', 'val'].
        pct_dataset_split (dict): Proportion of each dataset in each split, for the current size.
        to_drop_train (int): Total number of participants to drop from training.
        to_drop_val (int): Total number of participants to drop from validation.

    Returns:
        dropped_train (int): Total participants dropped from training set (floored).
        dropped_val (int): Total participants dropped from validation set (floored).
        dataset_split_drop (dict): Nested dict indicating how many were dropped
                                   per dataset and per split.
    """

    dataset_split_drop = {}
    for dataset in datasets:
        dataset_split_drop[dataset] = {}
        for split in splits:
            to_drop_split = {
                'train': to_drop_train,
                'val': to_drop_val
            }.get(split)

            # compute floor number of participants to drop for each
            # train+val size, for each dataset, for each split.
            dataset_split_drop[dataset][split] = math.floor(
                pct_dataset_split[size_trainval][dataset][split] *
                to_drop_split)

    dropped_train = sum(dataset_split_drop[dataset]['train']
                        for dataset in datasets)
    dropped_val = sum(dataset_split_drop[dataset]['val']
                      for dataset in datasets)

    return dropped_train, dropped_val, dataset_split_drop


def assign_remainder_dropped(size_trainval, splits, dropped_train, dropped_val,
                             to_drop_train, to_drop_val, to_drop_trainval,
                             dataset_split_drop, datasets, prev_trainval,
                             sizes_trainval, in_out_dir, seed, classes,
                             dx_name):
    """
    Handle the remainder participants that need to be dropped after flooring values.

    After using floor division to calculate how many participants to drop from each 
    dataset and split (via `assign_floor_dropped`), this function ensures that 
    the total number of dropped participants exactly matches the intended train+val size.

    The remainder is preferentially dropped from the ADNI validation split to preserve 
    class and dataset balance. This function also prepares and returns structured 
    information on dataset compositions before sampling the next subset.

    Args:
        size_trainval (int): Current train+val size being processed.
        splits (list): List of split names (e.g., ['train', 'val']).
        dropped_train (int): Number of participants already dropped from training.
        dropped_val (int): Number already dropped from validation.
        to_drop_train (int): Intended number to drop from training.
        to_drop_val (int): Intended number to drop from validation.
        to_drop_trainval (int): Total intended drop count from train+val.
        dataset_split_drop (dict): Dictionary with floor-dropped counts.
        datasets (list): List of dataset names (e.g., ['ADNI', 'AIBL', 'MACC']).
        prev_trainval (int): Previous (larger) train+val size.
        sizes_trainval (list): All train+val sizes (descending).
        in_out_dir (str): Base directory for CSV I/O.
        seed (int): Random seed used for consistency.
        classes (list): List of class labels (e.g., ['CN', 'AD']).
        dx_name (str): Column name for diagnosis class.

    Returns:
        dropped_train (int): Updated total dropped from training.
        dropped_val (int): Updated total dropped from validation.
        dataset_split_drop (dict): Final drop counts per dataset and split.
        previous_dataset_split (dict): DataFrames of previous size per dataset/split.
        previous_dataset_split_dx (dict): Nested DataFrames split by class.
        current_dataset_split (dict): Initialized empty structure for current subset.
    """

    for split in splits:
        split_drop, to_drop_split = {
            'train': (dropped_train, to_drop_train),
            'val': (dropped_val, to_drop_val)
        }.get(split)

        # for each split, check if there is a remainder
        if split_drop < to_drop_split:
            # if so, compute remainder,
            split_drop_add = math.floor(to_drop_split - split_drop)
            # to drop from ADNI dataset, for that split
            dataset_split_drop['ADNI'][split] += split_drop_add
        dropped_train = sum(dataset_split_drop[dataset]['train']
                            for dataset in datasets)
        dropped_val = sum(dataset_split_drop[dataset]['val']
                          for dataset in datasets)
    dropped_trainval = dropped_train + dropped_val

    # double-check if for train+val, if there's a remainder
    if dropped_trainval < to_drop_trainval:
        # if so, compute remainder,
        add_dropped_trainval = math.floor(to_drop_trainval - dropped_trainval)
        # to drop from ADNI dataset, from the val split
        dataset_split_drop['ADNI']['val'] += add_dropped_trainval
        dropped_train = sum(dataset_split_drop[dataset]['train']
                            for dataset in datasets)
        dropped_val = sum(dataset_split_drop[dataset]['val']
                          for dataset in datasets)
        dropped_trainval = dropped_train + dropped_val
    elif dropped_trainval > to_drop_trainval:
        print(
            'ERROR: Dropping more participants than intended for seed {}, for size_trainval {}'
            .format(seed, size_trainval))

    previous_dataset_split, previous_dataset_split_dx, current_dataset_split = {}, {}, {}
    for dataset in datasets:
        previous_dataset_split[dataset], previous_dataset_split_dx[
            dataset], current_dataset_split[dataset] = {}, {}, {}
        for split in splits:
            if prev_trainval == sizes_trainval[0]:
                df_split = pd.read_csv('{}/full/seed_{}/{}.csv'.format(
                    in_out_dir, seed, split))
            else:
                df_split = pd.read_csv('{}/{}/seed_{}/{}.csv'.format(
                    in_out_dir, prev_trainval, seed, split))
            # for each dataset, for each split, store previous train+val size's dataframe
            previous_dataset_split[dataset][split] = df_split[
                df_split['DATASET'] == dataset]
            previous_dataset_split_dx[dataset][split] = {}
            for dx in classes:
                # for each dataset, for each split, for each diagnosis, store previous train+val size's dataframe
                previous_dataset_split_dx[dataset][split][
                    dx] = previous_dataset_split[dataset][split].loc[
                        previous_dataset_split[dataset][split][dx_name] == dx]

    return dropped_train, dropped_val, dataset_split_drop, \
        previous_dataset_split, previous_dataset_split_dx, current_dataset_split


def sample_participants_from_last_largest_trainval_size(
        splits,
        classes,
        dataset_split_drop,
        prev_class,
        previous_dataset_split_dx,
        datasets,
        current_split_dataset_dx,
        current_dataset_split,
        current_split,
        seed,
        size_out_dir,
        logger=None):
    """
    Randomly sample participants for the current train+val size from the most recent
    larger train+val size (within the same seed), ensuring participant ID containment.
    The rationale is to ensure that the participant IDs from each smaller train+val
    size is a complete subset of the larger size, to prevent leakage of
    participant IDs across different train+val sizes.

    This function performs the final participant selection after determining how many 
    to drop from each dataset and class. It ensures class balance is maintained,
    alternating (across classes) drop priority when odd numbers are involved.

    The sampled participants become the official subset for the current train+val size,
    written to CSV for future use.

    Args:
        splits (list): List of split names (e.g., ['train', 'val']).
        classes (list): Binary classes (e.g., ['CN', 'AD']).
        dataset_split_drop (dict): Number of participants to drop per dataset/split.
        prev_class (str): Last class that had an extra participant dropped.
        previous_dataset_split_dx (dict): Previous data split by dataset/split/class.
        datasets (list): Dataset names (e.g., ['ADNI', 'AIBL', 'MACC']).
        current_split_dataset_dx (dict): Output structure for sampled data per class.
        current_dataset_split (dict): Output structure for sampled data per dataset.
        current_split (dict): Output structure for full sampled data per split.
        seed (int): Random seed for reproducibility.
        size_out_dir (str): Output directory for CSV files.
        logger (Logger, optional): Logger instance to record info/debug.

    Returns:
        prev_class (str): Updated last-dropped class for alternating logic.
    """

    split_dataset_dx_drop = {}
    for split in splits:
        split_dataset_dx_drop[split], current_split_dataset_dx[split] = {}, {}
        current_split[split] = pd.DataFrame()

        for dataset in datasets:
            # initialize dictionary to store number of participants to drop
            # for each diagnosis class (e.g., NCI, AD),
            # per dataset and per split
            split_dataset_dx_drop[split][dataset], current_split_dataset_dx[
                split][dataset] = {}, {}

            # check if the number-to-be-dropped is an even number, then
            if dataset_split_drop[dataset][split] % 2 == 0:
                # if so, the number of participants to be dropped from
                # each class is exactly half to retain class balance
                split_dataset_dx_drop[split][dataset][
                    classes[0]] = dataset_split_drop[dataset][split] / 2
                split_dataset_dx_drop[split][dataset][
                    classes[1]] = dataset_split_drop[dataset][split] / 2
            else:
                # else if odd number, one class needs to drop an additional participant.
                # Therefore, alternate between the binary classes to drop
                # (i.e. if last train+val size dropped AD class, current train+val size drops NCI class)
                # This ensures classes are evenly represented as sizes shrink.
                if prev_class == classes[1]:
                    choice = classes[0]
                else:
                    choice = classes[1]
                split_dataset_dx_drop[split][dataset][choice] = math.floor(
                    dataset_split_drop[dataset][split] / 2) + 1
                split_dataset_dx_drop[split][dataset][classes[
                    1 - classes.index(choice)]] = math.floor(
                        dataset_split_drop[dataset][split] / 2)

            for dx in classes:
                # randomly sample current train+val size dataframe from most
                # recent previous larger train+val size dataframe
                current_split_dataset_dx[split][dataset][
                    dx] = previous_dataset_split_dx[dataset][split][dx].sample(
                        n=int(
                            len(previous_dataset_split_dx[dataset][split][dx])
                            - split_dataset_dx_drop[split][dataset][dx]),
                        random_state=seed)

            current_dataset_split[dataset][split] = pd.concat([
                current_split_dataset_dx[split][dataset][classes[0]],
                current_split_dataset_dx[split][dataset][classes[1]]
            ])
            current_split[split] = pd.concat(
                [current_split[split], current_dataset_split[dataset][split]])

        current_split[split].to_csv('{}/{}.csv'.format(size_out_dir, split),
                                    index=False)

    # generating logger content
    log_train, log_val = len(current_split['train']), len(current_split['val'])
    log_trainval = log_train + log_val
    logger.info('Actual train+val size = {}'.format(log_trainval))
    logger.info('Expected train:val ratio = 4.0:1')
    logger.info('Actual train:val ratio = {: .1f}:1\n'.format(log_train /
                                                              log_val))

    # store previous dropped class, to drop alternate binary class in next train+val size
    prev_class = choice
    return prev_class


def vary_train_val_size_per_seed(in_out_dir, step, seed, task):
    """
    Generate all train and val splits across varying sizes for a given seed
    (i.e. a single train-val-test split).

    Ensures participant consistency across sizes, manages balanced downsampling, 
    and logs stats about each size.

    Args:
        in_out_dir (str): Directory to read/write split files.
        step (int): Step size to decrement train+val size.
        seed (int): Random seed index.
        task (str): Task type (e.g., 'ad_classification').
    """

    # establishing convenience variables
    classes = global_config.TASK_DX[task]
    datasets = global_config.DATASETS
    splits = ['train', 'val']
    dx_name = global_config.DX_COL

    # compute full train, val, and train+val sizes
    full_train = len(
        pd.read_csv('{}/full/seed_{}/train.csv'.format(in_out_dir, seed)))
    full_val = len(
        pd.read_csv('{}/full/seed_{}/val.csv'.format(in_out_dir, seed)))
    full_trainval = full_train + full_val

    # generate list of train+val sizes in descending order
    sizes_trainval = list(range(step, full_trainval, step))
    if full_trainval not in sizes_trainval:
        sizes_trainval.append(full_trainval)
    sizes_trainval.reverse()

    # for each seed, when starting from full size, initialize prev_* variables
    prev_trainval, prev_class = sizes_trainval[0], classes[0]

    # initialize dictionaries used for storing number and percentage within
    # each train+val size, within each dataset, within each split
    num_dataset_split, pct_dataset_split = {}, {}

    # loop through each train+val size to sample participants from the
    # previous larger size (ensures each smaller train+val size's participants
    # is a complete subset of the larger size)
    for size_trainval in sizes_trainval:
        # create out dir to store output split csv for each train+val size, for each seed
        size_out_dir = '{}/{}/seed_{}'.format(in_out_dir, size_trainval, seed)

        # create log file
        log_file = os.path.join(size_out_dir, 'seed_{}.log'.format(seed))
        #assert not os.path.exists(log_file), 'Log file already exists!'
        os.makedirs(size_out_dir, exist_ok=True)

        # create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info(f'Expected train+val size: {size_trainval}')

        # store percentages to represent proportions of current train+val size's dataset's
        # split's participant numbers to full train+val size's dataset's split's participant numbers.
        pct_dataset_split = store_percentage_within_seed(
            size_trainval, num_dataset_split, pct_dataset_split, datasets,
            splits, prev_trainval, sizes_trainval, in_out_dir, seed,
            full_train, full_val)

        # compute percentage to represent proportion of train:train+val, and val:train+val.
        pct_train = (full_train / full_trainval)
        pct_val = (full_val / full_trainval)

        # compute exact (possibly decimal) number of train and val participants to drop
        to_drop_trainval = prev_trainval - size_trainval
        to_drop_train = pct_train * to_drop_trainval
        to_drop_val = pct_val * to_drop_trainval

        # since exact number of participants to drop could be decimal, drop floor number first.
        dropped_train, dropped_val, dataset_split_drop = assign_floor_dropped(
            size_trainval, datasets, splits, pct_dataset_split, to_drop_train,
            to_drop_val)

        # after dropping floor number, there could be a remainder, proceed to drop the remainder
        # to obtain expected train+val size
        dropped_train, dropped_val, dataset_split_drop, previous_dataset_split, \
            previous_dataset_split_dx, current_dataset_split = assign_remainder_dropped(
            size_trainval, splits, dropped_train, dropped_val, to_drop_train,
            to_drop_val, to_drop_trainval, dataset_split_drop, datasets,
            prev_trainval, sizes_trainval, in_out_dir, seed, classes, dx_name)

        current_split_dataset_dx, current_split = {}, {}
        # check if current train+val size is not the full size,
        if size_trainval != sizes_trainval[0]:
            # if so, randomly sample participants for current split, from most
            # recent previous larger split
            prev_class = sample_participants_from_last_largest_trainval_size(
                splits,
                classes,
                dataset_split_drop,
                prev_class,
                previous_dataset_split_dx,
                datasets,
                current_split_dataset_dx,
                current_dataset_split,
                current_split,
                seed,
                size_out_dir,
                logger=logger)
        else:
            # else, assign current split as full split, and output current full split
            current_split = {}
            for split in splits:
                current_split[split] = pd.DataFrame()
                for dataset in datasets:
                    current_dataset_split[dataset][
                        split] = previous_dataset_split[dataset][split]
                    current_split[split] = pd.concat([
                        current_split[split],
                        current_dataset_split[dataset][split]
                    ])
                    current_split[split].to_csv('{}/{}.csv'.format(
                        size_out_dir, split),
                                                index=False)

                # Divide into NCI-only data split
                # (to finetune BAG-finetune for predicting brain age)
                if task == 'ad_classification':
                    current_split_nci = current_split[split][
                        current_split[split]['DX'] == 'NCI']
                    current_split_nci.to_csv('{}/{}_nci.csv'.format(
                        size_out_dir, split),
                                             index=False)

        prev_trainval = size_trainval
        full_train = sum(
            len(current_dataset_split[dataset]['train'])
            for dataset in datasets)
        full_val = sum(
            len(current_dataset_split[dataset]['val']) for dataset in datasets)
        full_trainval = full_train + full_val

        logger.removeHandler(fh)


def vary_train_val_size(in_out_dir, step, seeds, task):
    """
    Wrapper function to generate train and val splits for all train+val sizes, for all desired seeds.
    """

    for seed in range(seeds):
        vary_train_val_size_per_seed(in_out_dir, step, seed, task)


def create_test_splits(in_out_dir, step, seeds, full_trainval, task):
    """
    Copy or symlink test splits to match each train+val size for each seed.

    Also creates NCI-only test splits if the task is AD classification.
    (this is for BAG-finetune to finetune on only cognitively normal
    participants to predict brain age)

    Args:
        in_out_dir (str)
        step (int)
        seeds (int)
        full_trainval (int)
        task (str)
    """
    # generate list of train+val sizes in descending order
    sizes_trainval = list(range(step, full_trainval, step))
    if full_trainval not in sizes_trainval:
        sizes_trainval.append(full_trainval)
    sizes_trainval.reverse()

    for size_trainval in sizes_trainval:
        for seed in range(seeds):
            # Create symlinks
            src = os.path.join(in_out_dir, 'full', 'seed_{}'.format(seed),
                               'test.csv')
            dst_dir = os.path.join(in_out_dir, str(size_trainval),
                                   'seed_{}'.format(seed))
            dst = os.path.join(dst_dir, 'test.csv')
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)

            # To evaluate BAG-finetune brain age prediction performance only on NCI participants
            if task == 'ad_classification':
                df_test = pd.read_csv(src)
                df_test_nci = df_test[df_test['DX'] == 'NCI']
                df_test_nci.to_csv(f'{dst_dir}/test_nci.csv', index=False)


def get_args():
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace with the following attributes:
            - task (str)
            - num_split (int)
            - step (int)
            - full_trainval_size (int)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        choices=['ad_classification', 'mci_progression'])
    parser.add_argument('--num_split', type=int, default=50)
    parser.add_argument('--step', type=int, default=50)
    parser.add_argument('--full_trainval_size', type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    in_out_dir = os.path.join(global_config.DATA_SPLIT_DIR, args.task)
    vary_train_val_size(in_out_dir=in_out_dir,
                        seeds=args.num_split,
                        step=args.step,
                        task=args.task)

    full_trainval = args.full_trainval_size
    create_test_splits(in_out_dir=in_out_dir,
                       step=args.num_split,
                       seeds=args.num_split,
                       full_trainval=args.full_trainval_size,
                       task=args.task)
