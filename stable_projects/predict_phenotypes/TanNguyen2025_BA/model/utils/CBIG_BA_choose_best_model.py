"""
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script identifies the best model (based on validation AUC) among different train-val-test splits (seeds)
and creates a symbolic link named 'best_model' pointing to the best-performing checkpoint.

Summary statistics (validation AUC and optionally test AUC) are saved in a log file and CSV for plotting.
If specified, a `.mat` file with test AUCs is also saved.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.io import savemat
from datetime import datetime
import sys

# Load path to utility scripts
TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)

# Import custom complete flag generator
from utils.CBIG_BA_complete_step import generate_complete_flag

def main(args):
    # Flag to determine whether to summarize test AUCs
    summary_test = args.summary_test == 'y'

    # Create output summary directory
    output_dir = os.path.join(args.base_dir, '_summary')
    os.makedirs(output_dir, exist_ok=True)

    # Format current date as a timestamp
    start_time_fmt = datetime.now().strftime("%Y%m%d")

    # Exclude unrelated directories when scanning model checkpoints
    exclude_dir = [
        'log', 'seed_SEEDID', 'best_model', 'complete_train', 'complete_test'
    ]

    best_model_all_splits = []

    # Loop over all seeds to identify best model for each
    for seed_id in range(args.seed_max):
        seed_dir = os.path.join(args.base_dir, f'seed_{seed_id}')
        model_list = sorted(
            [i for i in os.listdir(seed_dir) if i not in exclude_dir])

        # Track best model details
        best_auc, best_id = -1, None

        # Loop through each model directory to find best AUC
        for model_id in model_list:
            param_json = os.path.join(
                seed_dir, model_id,
                'params_test.json' if summary_test else 'params.json')
            with open(param_json, 'r') as f:
                data = json.load(f)

            if data['best_auc'] > best_auc:
                best_auc = data['best_auc']
                best_id = model_id
                best_lr = data['init_lr']
                if summary_test:
                    test_auc = data['test_auc']

        # Store best model info for this seed
        if summary_test:
            best_model_all_splits.append(
                [seed_id, best_id, best_auc, best_lr, test_auc])
        else:
            best_model_all_splits.append([seed_id, best_id, best_auc, best_lr])

        # Create symlink pointing to best model (if enabled)
        if args.create_link == 'y':
            os.system(
                f"ln -sfT {os.path.join(seed_dir, best_id)} {os.path.join(seed_dir, 'best_model')}"
            )

    # Save best model info into a DataFrame
    if summary_test:
        df = pd.DataFrame(best_model_all_splits,
                          columns=[
                              'SEED_ID', 'BEST_ID', 'BEST_VAL_AUC', 'BEST_LR',
                              'TEST_AUC'
                          ])
    else:
        df = pd.DataFrame(
            best_model_all_splits,
            columns=['SEED_ID', 'BEST_ID', 'BEST_VAL_AUC', 'BEST_LR'])

    # Write AUC stats to log file
    log_file = os.path.join(
        output_dir, f'results_first{args.seed_max}seeds_{start_time_fmt}.log')
    with open(log_file, 'w') as f:
        f.write('VALIDATION AUC = {:.4f} +/- {:.4f}\n'.format(
            np.mean(df['BEST_VAL_AUC']), np.std(df['BEST_VAL_AUC'], ddof=1)))
        if summary_test:
            f.write('TEST AUC = {:.4f} +/- {:.4f}\n'.format(
                np.mean(df['TEST_AUC']), np.std(df['TEST_AUC'], ddof=1)))

    # Save detailed summary for plotting and further analysis
    output_csv = os.path.join(
        output_dir, f'summary_first{args.seed_max}seeds_{start_time_fmt}.csv')
    output_mat = os.path.join(
        output_dir,
        f'direct_test_auc_first{args.seed_max}seeds_{start_time_fmt}.mat')

    # Avoid overwriting existing results
    #if os.path.exists(output_csv) or os.path.exists(output_mat):
        #raise FileExistsError(f"{output_csv} or {output_mat} already exists!")
    #else:
    df.to_csv(output_csv, index=False)
    if summary_test:
        savemat(output_mat, {'test_auc': df['TEST_AUC'].to_list()})

    # Mark each seed as completed for best_model selection
    for seed_id in range(args.seed_max):
        seed_dir = os.path.join(args.base_dir, f'seed_{seed_id}')
        generate_complete_flag(seed_dir, 'best_model')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--base_dir',
        type=str,
        help=
        'Base directory containing per-seed folders. Each folder must contain a params.json.'
    )
    parser.add_argument('--seed_max',
                        type=int,
                        default=50,
                        help='Total number of seeds to process.')
    parser.add_argument(
        '--create_link',
        type=str,
        choices=['y', 'n'],
        default='y',
        help=
        'Whether to create a symbolic link (named "best_model") to the best model per seed.'
    )
    parser.add_argument(
        '--summary_test',
        type=str,
        choices=['y', 'n'],
        default='y',
        help=
        'Whether to summarize and report test AUC in addition to validation AUC.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
