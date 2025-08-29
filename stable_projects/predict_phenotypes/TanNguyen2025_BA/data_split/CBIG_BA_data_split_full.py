"""
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script performs train-validation-test data splitting by scanner model and diagnosis label
for the largest sample size (for each classification task).
This script balances classes within each scanner model as much as possible.

Scans are selected from baseline visits only and limited to AIBL, ADNI, and MACC datasets.
When necessary, excess samples are removed to ensure class balance
(e.g., #NCI = #AD or #sMCI = #pMCI).

Expected output(s):
1. $TANNGUYEN2025_BA_DIR/data/data_split/[task]/full/seed_[i]/{train,val,test}.csv
2. Log file: seed_[i].log in the same output folder

Example:
    conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR};
    python -m data_split.CBIG_BA_data_split_full --task ad_classification --num_split 50
"""

import os
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from CBIG_BA_config import global_config


def train_val_test_split(df,
                         id_col,
                         scanner_col,
                         dx_col,
                         dx_labels,
                         dev_test_ratio=0.2,
                         train_val_ratio=0.2,
                         seed=0,
                         logger=None):
    """
    Perform train-validation-test split on a dataframe grouped by scanner model.

    The function splits the input data into train, val, and test sets, preserving
    class labels (stratified) within each scanner model group. Handles edge cases
    where scanner groups are too small to allow standard splitting.

    Args:
        df (pd.DataFrame): Input dataframe containing subject info.
        id_col (str): Column name for subject ID (must be unique).
        scanner_col (str): Column name for scanner model.
        dx_col (str): Column name for diagnosis label.
        dx_labels (list[str]): List of diagnosis labels to stratify on (e.g., ['AD', 'NCI']).
        dev_test_ratio (float): Proportion of data held out for test set.
        train_val_ratio (float): Proportion of dev set held out for validation.
        seed (int): Random seed for reproducibility.
        logger (logging.Logger): Logger object for tracking split progress.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for train, val, and test sets.
    """
    # Check if RID unique
    rid = df[id_col].to_list()
    # Ensure subject IDs are unique to prevent data leakage across splits
    assert len(rid) == len(set(rid))

    # Perform train-val-test splits by scanner model with 2 steps:
    ## Step 1: split all data to dev, test using dev_test_ratio
    ## Step 2: split dev to train, val using train_val_ratio
    splits = {'train': [], 'val': [], 'test': []}
    scanner_models = df[scanner_col].unique().tolist()
    for model in scanner_models:
        df_model = df.loc[df[scanner_col] == model, :].copy()

        # Handle edge case: scanner model has only 4 samples (2 per class)
        if len(df_model[dx_col].to_list()) == 4:
            logger.info('{} only have 4 samples'.format(model))

            # If there are only 2 pairs of NCI/AD or sMCI/pMCI (4 samples) ->
            # not enough to split to train/val/test with stratify label -> no stratify
            ## Put 2 (1 nci/smci, 1 ad/pmci) to train and the other 2 put random to val/test
            train_dx_all = []
            remain_dx_all = []
            for diagnosis in dx_labels:
                df_model_dx = df_model.loc[df[dx_col] == diagnosis].copy()
                # Randomly select one sample for training from this class
                train_dx = df_model_dx.sample(n=1, random_state=seed)
                remain_dx = df_model_dx[~df_model_dx.index.isin(train_dx.index
                                                                )]

                train_dx_all.append(train_dx)
                remain_dx_all.append(remain_dx)

            train = pd.concat(train_dx_all).reset_index(drop=True)
            remain = pd.concat(remain_dx_all).reset_index(drop=True)

            # Randomly assign one remaining sample to validation set
            val = remain.sample(n=1, random_state=seed)
            test = remain[~remain.index.isin(val.index)]

        # Handle edge case: scanner model has only 2 samples (1 per class);
        # split into train and test only
        elif len(df_model[dx_col].to_list()) == 2:
            logger.info('{} only have 2 samples'.format(model))

            # If there is 1 pair (2 samples) => put 1 in train set and 1 in test set
            train, test = train_test_split(df_model,
                                           test_size=1,
                                           random_state=seed)
            val = pd.DataFrame()

        else:
            train_dx, val_dx, test_dx = [], [], []
            for diagnosis in dx_labels:
                df_model_dx = df_model.loc[df_model[dx_col] ==
                                           diagnosis].copy()
                # Split class samples into development and test sets with fixed ratio
                df_model_dx_dev, df_model_dx_test = train_test_split(
                    df_model_dx, test_size=dev_test_ratio, random_state=seed)
                # Further split development set into training and validation sets
                df_model_dx_train, df_model_dx_val = train_test_split(
                    df_model_dx_dev,
                    test_size=train_val_ratio,
                    random_state=seed)

                train_dx.append(df_model_dx_train)
                val_dx.append(df_model_dx_val)
                test_dx.append(df_model_dx_test)

            train = pd.concat(train_dx)
            val = pd.concat(val_dx)
            test = pd.concat(test_dx)

        splits['train'].append(train)
        splits['val'].append(val)
        splits['test'].append(test)
        logger.info('{}: train={}, val={}, test={}'.format(
            model, train.shape[0], val.shape[0], test.shape[0]))

    train_df = pd.concat(splits['train'])
    val_df = pd.concat(splits['val'])
    test_df = pd.concat(splits['test'])

    return train_df, val_df, test_df


def data_split_1_seed(matched_dir, output_dir, task, seed_split):
    """
    Perform a single train-validation-test split for one seed (e.g.,
    1 random train-val-test split) across all datasets.

    Loads balanced CSVs for AIBL, ADNI, and MACC datasets, performs scanner-aware
    class-balanced splitting, and saves resulting train/val/test CSVs to output folder.

    Args:
        matched_dir (str): Path to directory containing balanced dataset CSVs.
        output_dir (str): Directory to save train/val/test split CSVs.
        task (str): Classification task ('ad_classification' or 'mci_progression').
        seed_split (int): Random seed index used for the current split.
    """
    # General variables
    dev_test = 0.2
    train_val = 0.2
    id_col = global_config.ID_COL
    scanner_col = global_config.SCANNER_COL
    dx_col = global_config.DX_COL
    class_labels = global_config.TASK_DX[task]

    # Paths
    input_csv = {
        'AIBL': os.path.join(matched_dir, 'AIBL_balanced.csv'),
        'ADNI': os.path.join(matched_dir, 'ADNI_balanced.csv'),
        'MACC': os.path.join(matched_dir, 'MACC_balanced.csv')
    }
    log_file = os.path.join(output_dir, 'seed_{}.log'.format(seed_split))
    #assert not os.path.exists(log_file), 'Log file already exists!'
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Repeated train-validation-test split
    train_all, val_all, test_all = [], [], []
    for dataset in global_config.DATASETS:
        logger.info(f"{dataset.upper()}")
        df_balanced = pd.read_csv(input_csv[dataset])

        # Train val test split
        train, val, test = train_val_test_split(df_balanced,
                                                id_col,
                                                scanner_col,
                                                dx_col,
                                                class_labels,
                                                dev_test_ratio=dev_test,
                                                train_val_ratio=train_val,
                                                seed=seed_split,
                                                logger=logger)
        logger.info('-----{}: #TRAIN={}, #VAL={}, #TEST={}-----\n'.format(
            dataset.upper(), train.shape[0], val.shape[0], test.shape[0]))

        # Add dataset name column to track origin after
        # merging across datasets
        for df in [train, val, test]:
            df.insert(0, 'DATASET', dataset.upper())

        train_all.append(train)
        val_all.append(val)
        test_all.append(test)

    # Combine train, validation, and test across datasets
    train_all = pd.concat(train_all, ignore_index=True)
    val_all = pd.concat(val_all, ignore_index=True)
    test_all = pd.concat(test_all, ignore_index=True)

    # Save to csv
    train_all.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_all.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_all.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    logger.removeHandler(fh)


def data_split(args):
    """
    Wrapper function to run data splitting for a given task across multiple
    random seeds (i.e., train-val-test splits).

    For each seed, creates a new subfolder under the task's output path and calls
    `data_split_1_seed()`.

    Args:
        args (argparse.Namespace): Command-line arguments with `task` and `num_split`.

    Returns:
        None
    """
    print(f'Data split for {args.task}...'.upper())
    for i in range(args.num_split):
        data_split_1_seed(matched_dir=os.path.join(
            global_config.MATCHED_DATA_DIR, args.task),
                          output_dir=os.path.join(global_config.DATA_SPLIT_DIR,
                                                  args.task, 'full',
                                                  f'seed_{i}'),
                          task=args.task,
                          seed_split=i)


def get_args():
    """
    Parse command-line arguments for data splitting.

    Returns:
        argparse.Namespace: Parsed arguments including task name and number of splits.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        choices=['ad_classification', 'mci_progression'])
    parser.add_argument('--num_split', type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    data_split(get_args())
