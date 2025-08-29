"""
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

Script for verifying the correctness of results for AD classification
and MCI progression tasks. It compares new results against reference results for:
1. Age & sex-matched data
2. Train-validation-test splitting
3. Model outputs for all AD classification & MCI progression prediction models:
    1) Direct (AD classification)
    2) BAG
    3) BAG-finetune
    4) Brainage64D
    5) Brainage64D-finetune
    6) Direct (MCI progression prediction)
    7) Direct-AD
    8) Brainage64D-finetune-AD

By default, this script checks the results for:
    1) all 3 steps above (i.e., matched data to model outputs)
    2) all train-val-test splits
    3) all sample sizes
    4) all tasks
    5) all models

Example:
    conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR};
    python -m replication.scripts.CBIG_BA_4_check_reference_results;
"""

import os
import argparse
import numpy as np
import pandas as pd
from CBIG_BA_config import global_config
from model.utils.CBIG_BA_io import read_json, txt2list
from scipy.io import loadmat
from utils.CBIG_BA_complete_step import generate_complete_flag


def check_matching(ref_dir, new_dir, output_dir, verbose=False):
    """
    Compare age & sex-matched data CSV files
    between reference and new directories.

    Args:
        ref_dir (str): Path to the reference matched data directory.
        new_dir (str): Path to the new matched data directory.
        verbose (bool): Whether to print detailed result logs.

    Returns:
        bool: True if all matching files are identical, False otherwise.
    """
    # Define what we want to check
    check_comp = [
        f"{i}_{j}" for i in global_config.DATASETS
        for j in ['balanced', 'excess']
    ]
    check_dict = {k: False for k in check_comp}

    # Check and compare results
    for i in check_comp:
        ref_df = pd.read_csv(os.path.join(ref_dir, f"{i}.csv"))
        new_df = pd.read_csv(os.path.join(new_dir, f"{i}.csv"))

        check_dict[i] = ref_df.equals(new_df)

    # Combine all test results and save log for pretty print
    is_pass = all(check_dict.values())
    if is_pass and verbose:
        print("Check matching all passed!")
    elif not is_pass and verbose:
        print(
            f"Check matching failed at: {[k for k, v in check_dict.items() if not v]}"
        )
    else:
        pass

    return is_pass


def check_splits(ref_dir,
                 new_dir,
                 split_size,
                 split_list,
                 output_dir,
                 verbose=False):
    """
    Compare data split CSV files (train, val, test)
    between reference and new directories.

    Args:
        ref_dir (str): Path to the reference data split directory.
        new_dir (str): Path to the new data split directory.
        split_size (list): List of development sizes to check.
        split_list (list): List of random train-val-test splits
                           to check.
        verbose (bool): Whether to print detailed result logs.

    Returns:
        bool: True if all split files are identical, False otherwise.
    """
    # Define what we want to check
    check_comp = [f'{i}-seed_{j}' for i in split_size for j in split_list]
    check_dict = {k: False for k in check_comp}

    # Check and compare results
    for i in check_comp:
        size, split = i.split('-')
        i_subset = []
        for subset in ['train', 'val', 'test']:
            ref_df = pd.read_csv(
                os.path.join(ref_dir, size, split, f"{subset}.csv"))
            new_df = pd.read_csv(
                os.path.join(new_dir, size, split, f"{subset}.csv"))

            i_subset.append(ref_df.equals(new_df))
        check_dict[i] = all(i_subset)

    # Combine all test results and save log for pretty print
    is_pass = all(check_dict.values())
    if is_pass and verbose:
        print("Check data split all passed!")
    elif not is_pass and verbose:
        print(
            f"Check data split failed at: {[k for k, v in check_dict.items() if not v]}"
        )
    else:
        pass

    return is_pass


def check_cnn(ref_dir, new_dir, verbose=False, BAG_finetune_flag=False):
    """
    Compare CNN model training and test results between reference and new runs.

    Args:
        ref_dir (str): Reference results directory for the model.
        new_dir (str): New results directory for the model.
        verbose (bool): Whether to print detailed logs.
        BAG_finetune_flag (bool): If True, compares MAE instead of AUC.

    Returns:
        bool: True if all components (params, training curves, test scores) match, False otherwise.
    """
    if BAG_finetune_flag:
        score = 'mae'
    else:
        score = 'auc'

    # Define what we want to check
    check_dict = {
        k: False
        for k in [
            'check_best', f'train_{score}_compare', f'val_{score}_compare',
            f'test_{score}_compare'
        ]
    }

    # Check best results
    check_list = [f'best_{score}', 'best_epoch', f'test_{score}']
    ref_dict = read_json(os.path.join(ref_dir, 'params_test.json'), check_list)
    new_dict = read_json(os.path.join(new_dir, 'params_test.json'), check_list)
    wrong_list = [i for i in check_list if ref_dict[i] != new_dict[i]]
    check_dict['check_best'] = len(wrong_list) == 0

    # Check train
    check_dict[f'train_{score}_compare'] = np.allclose(
        txt2list(os.path.join(ref_dir, f'train_{score}.txt')),
        txt2list(os.path.join(new_dir, f'train_{score}.txt')))
    check_dict[f'val_{score}_compare'] = np.allclose(
        txt2list(os.path.join(ref_dir, f'val_{score}.txt')),
        txt2list(os.path.join(new_dir, f'val_{score}.txt')))

    # Check prediction
    ref_test = pd.read_csv(os.path.join(ref_dir, 'test_predict_score.csv'))
    new_test = pd.read_csv(os.path.join(new_dir, 'test_predict_score.csv'))
    check_dict[f'test_{score}_compare'] = ref_test.equals(new_test)

    # Combine all test results and save log for pretty print
    is_pass = all(check_dict.values())
    if is_pass and verbose:
        print("Check CNN all passed!")
    elif not is_pass and verbose:
        print(
            f"Check CNN failed at: {[k for k, v in check_dict.items() if not v]}"
        )
    else:
        pass

    return is_pass


def check_logreg(ref_dir, new_dir, verbose=False):
    """
    Compare logistic regression model results between reference and new runs.

    Args:
        ref_dir (str): Reference results directory.
        new_dir (str): New results directory.
        verbose (bool): Whether to print detailed logs.

    Returns:
        bool: True if parameter and prediction results match, False otherwise.
    """
    # Define what we want to check
    check_dict = {k: False for k in ['check_best', 'test_auc_compare']}

    # Check best results
    check_list = ['best_reg', 'best_auc', 'test_auc']
    ref_dict = read_json(os.path.join(ref_dir, 'params_test.json'), check_list)
    new_dict = read_json(os.path.join(new_dir, 'params_test.json'), check_list)
    wrong_list = [i for i in check_list if ref_dict[i] != new_dict[i]]
    check_dict['check_best'] = len(wrong_list) == 0

    # Check prediction
    ref_test = pd.read_csv(os.path.join(ref_dir, 'test_predict_score.csv'))
    new_test = pd.read_csv(os.path.join(new_dir, 'test_predict_score.csv'))
    check_dict['test_auc_compare'] = ref_test.equals(new_test)

    # Combine all test results and save log for pretty print
    is_pass = all(check_dict.values())
    if is_pass and verbose:
        print("Check logistic regression all passed!")
    elif not is_pass and verbose:
        print(
            f"Check logistic regression failed at: {[k for k, v in check_dict.items() if not v]}"
        )
    else:
        pass

    return is_pass


def check_BAG_classifier(ref_dir, new_dir, verbose=False):
    """
    Compare BAG classifier results (.mat format) between reference and new runs.

    Args:
        ref_dir (str): Reference results directory.
        new_dir (str): New results directory.
        verbose (bool): Whether to print detailed logs.

    Returns:
        bool: True if .mat test prediction results match, False otherwise.
    """
    # Define what we want to check
    check_dict = {k: False for k in ['test_auc_compare']}

    # Check prediction
    ref_mat = loadmat(os.path.join(ref_dir, 'test_auc_50seeds.mat'))
    new_mat = loadmat(os.path.join(new_dir, 'test_auc_50seeds.mat'))
    # Compare the loaded .mat content (assumes keys are the same)
    check_dict['test_auc_compare'] = all(
        np.array_equal(ref_mat[k], new_mat[k]) for k in ref_mat
        if not k.startswith('__')  # ignore MATLAB metadata keys
    )

    # Combine all test results and save log for pretty print
    is_pass = all(check_dict.values())
    if is_pass and verbose:
        print("Check BAG classifier all passed!")
    elif not is_pass and verbose:
        print(
            f"Check BAG classifier failed at: {[k for k, v in check_dict.items() if not v]}"
        )
    else:
        pass

    return is_pass


def check_model(ref_dir,
                new_dir,
                model_list,
                split_size,
                split_list,
                output_dir,
                verbose=False,
                task='ad_classification'):
    """
    Main function to check model results across all configurations and types.

    Args:
        ref_dir (str): Base directory containing reference model outputs.
        new_dir (str): Base directory containing new model outputs.
        model_list (list): List of model names to validate.
        split_size (list): List of dataset sizes to validate.
        split_list (list): List of random seed IDs to validate.
        verbose (bool): Whether to print detailed logs.
        task (str): Task name ('ad_classification' or 'mci_progression').

    Returns:
        bool: True if all model results match, False otherwise.
    """

    # Define what we want to check
    non_vary_size_model_list = [
        m for m in model_list if m not in ['BAG', 'BAG_finetune']
        and not (task == 'mci_progression' and m == 'direct')
    ]
    check_comp = [
        f'{i}-{j}-seed_{k}' for i in non_vary_size_model_list
        for j in split_size for k in split_list
    ]
    check_dict = {k: False for k in check_comp}

    # Check and compare results
    for i in check_comp:
        model, size, split = i.split('-')

        # Handle non-BAG models
        if model in ['direct', 'brainage64D_finetune']:
            # Handle MCI progression direct separately
            if (model == 'direct') and (task == 'mci_progression'):
                continue

            check_dict[i] = check_cnn(
                ref_dir=os.path.join(ref_dir, model, size, split,
                                     'best_model'),
                new_dir=os.path.join(new_dir, model, size, split,
                                     'best_model'))
        elif model in ['direct_ad', 'brainage64D', 'brainage64D_finetune_ad']:
            if model == 'brainage64D':
                ref_path = os.path.join(ref_dir, model, 'log_reg', size, split)
                new_path = os.path.join(new_dir, model, 'log_reg', size, split)
            else:
                ref_path = os.path.join(ref_dir, model, size, split)
                new_path = os.path.join(new_dir, model, size, split)

            check_dict[i] = check_logreg(ref_dir=ref_path, new_dir=new_path)

    # Handle BAG models separately
    # (because only 1x sample size of 997)
    BAG_model_list = [m for m in model_list if m in ['BAG', 'BAG_finetune']]
    for model in BAG_model_list:
        if model in model_list:
            for split in split_list:
                check_BAG_classifier(
                    ref_dir=os.path.join(ref_dir, model, 'BAG_classifier'),
                    new_dir=os.path.join(new_dir, model, 'BAG_classifier'))

    # Handle MCI progression direct separately
    # (because only 1x sample size of 448)
    if ('direct' in model_list) and (task == 'mci_progression'):
        model = 'direct'
        for split in split_list:
            key = f"{model}-448-seed_{split}"
            check_dict[key] = check_cnn(
                ref_dir=os.path.join(ref_dir, model, '448',
                                     f'seed_{str(split)}', 'best_model'),
                new_dir=os.path.join(new_dir, model, '448',
                                     f'seed_{str(split)}', 'best_model'))

    # Combine all test results and save log for pretty print
    is_pass = all(check_dict.values())
    if is_pass and verbose:
        print(f"Check models all passed!")
        # Generate complete flags for next step to commence
        generate_complete_flag(output_dir, f'4_check_reference_results_{task}')
    elif not is_pass and verbose:
        print(
            f"Check models failed at: {[k for k, v in check_dict.items() if not v]}"
        )
    else:
        pass

    return is_pass


def check_reference_results(args):
    """
    Wrapper to check matched data, split files, and model outputs based on user-specified args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments from `get_args()`.
    """

    # Shared variables
    task_models = {
        'ad_classification': [
            'direct', 'BAG', 'BAG_finetune', 'brainage64D',
            'brainage64D_finetune'
        ],
        'mci_progression': ['direct', 'direct_ad', 'brainage64D_finetune_ad']
    }
    split_set = {
        'ad_classification':
        ['50'] + [str(i) for i in range(100, 901, 100)] + ['997'],
        'mci_progression': [str(i) for i in range(50, 401, 50)] + ['448']
    }
    task2check = [args.task] if args.task != 'all' else [
        'ad_classification', 'mci_progression'
    ]
    VERBOSE = args.verbose == 'y'

    # Results folders to compare
    REFERENCE_DIR = os.path.join(global_config.REPLICATION_DATA, 'ref_results')
    # replication
    DATA_DIR = os.path.join(global_config.ROOT_DIR, 'data')
    OUTPUT_DIR = os.path.join(global_config.ROOT_DIR, 'replication', 'output')

    # Print what are we checking
    print("[INFO]: Tasks:", task2check)
    print("[INFO]: Split sets:", split_set)
    print("[INFO]: Split ids:", args.split)

    # MAIN WORK
    for task in task2check:
        print(f"\nTask: {task}".upper())
        if args.module == "matching":
            check_matching(ref_dir=os.path.join(REFERENCE_DIR, 'matched',
                                                task),
                           new_dir=os.path.join(DATA_DIR, 'matched', task),
                           output_dir=OUTPUT_DIR,
                           verbose=VERBOSE)

        elif args.module == "data_split":
            check_splits(ref_dir=os.path.join(REFERENCE_DIR, 'data_split',
                                              task),
                         new_dir=os.path.join(DATA_DIR, 'data_split', task),
                         split_size=split_set[task],
                         split_list=args.split,
                         output_dir=OUTPUT_DIR,
                         verbose=VERBOSE)

        elif args.module == "models":
            check_model(ref_dir=os.path.join(REFERENCE_DIR, 'output', task),
                        new_dir=os.path.join(OUTPUT_DIR, task),
                        model_list=task_models[task],
                        split_size=split_set[task],
                        split_list=args.split,
                        output_dir=OUTPUT_DIR,
                        verbose=VERBOSE,
                        task=task)

        else:  # Check everything
            check_matching(
                ref_dir=os.path.join(REFERENCE_DIR, 'matched', task),
                new_dir=os.path.join(DATA_DIR, 'matched', task),
                output_dir=OUTPUT_DIR,
                verbose=VERBOSE,
            )

            check_splits(ref_dir=os.path.join(REFERENCE_DIR, 'data_split',
                                              task),
                         new_dir=os.path.join(DATA_DIR, 'data_split', task),
                         split_size=split_set[task],
                         split_list=args.split,
                         output_dir=OUTPUT_DIR,
                         verbose=VERBOSE)

            check_model(ref_dir=os.path.join(REFERENCE_DIR, 'output', task),
                        new_dir=os.path.join(OUTPUT_DIR, task),
                        model_list=task_models[task],
                        split_size=split_set[task],
                        split_list=args.split,
                        output_dir=OUTPUT_DIR,
                        verbose=VERBOSE,
                        task=task)


def get_args():
    """
    Parse and return command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--module',
                        choices=['matching', 'data_split', 'models', 'all'],
                        help='Module to check',
                        default='all')
    parser.add_argument(
        '--task',
        choices=['ad_classification', 'mci_progression', 'all'],
        help="Task to check reference results (default: all)",
        default='all')
    parser.add_argument('--split',
                        nargs="+",
                        type=int,
                        help="List of data split to check",
                        default=list(range(50)))
    parser.add_argument('--verbose', choices=['y', 'n'], default='y')
    return parser.parse_args()


if __name__ == "__main__":
    check_reference_results(get_args())
