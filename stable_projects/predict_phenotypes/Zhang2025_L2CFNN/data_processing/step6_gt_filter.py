#!/usr/bin/env python
# Filter out ground truth subjects who are actually used
# during evaluation based on the input data.
# This is because sometimes the ground truth is available,
# but the input timepoints are not (none of the required features is present)
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


import argparse
from os import makedirs, path

import numpy as np
import pandas as pd

from config import global_config


def remove_missing(
    data_df, baseline_df, data_id_col="RID", baseline_id_col="patients"
):
    """
    Filters data_df to include only IDs present in baseline_df.

    Args:
        data_df (pd.DataFrame): DataFrame containing data (e.g., test/val set).
                                Must have a column specified by data_id_col.
        baseline_df (pd.DataFrame): DataFrame containing baseline IDs.
                                    Must have a column specified by baseline_id_col.
        data_id_col (str): Name of the ID column in data_df. Defaults to 'RID'.
        baseline_id_col (str): Name of the ID column in baseline_df. Defaults to 'patients'.


    Returns:
        pd.DataFrame: A filtered version of data_df.
    """
    data_rids = data_df[data_id_col].unique()
    present_rids = baseline_df[baseline_id_col].unique()
    non_missing_rids = data_rids[np.isin(data_rids, present_rids)]
    # Use f-string for query for clarity if column name is dynamic
    filtered_df = data_df[data_df[data_id_col].isin(non_missing_rids)]
    return filtered_df


def main(args):
    """Main execution function"""
    site = args.site

    # Generate the save directory
    save_path = path.join(global_config.clean_gt_path, f"{site}")
    makedirs(save_path, exist_ok=True)

    if args.need_split:  # ADNI or fakeADNI has 20-splits
        for f in range(20):
            test_gt = pd.read_csv(
                path.join(global_config.split_path, f"{site}/fold{f}_test.csv")
            )
            val_gt = pd.read_csv(
                path.join(global_config.split_path, f"{site}/fold{f}_val.csv")
            )
            test_baseline = pd.read_csv(
                path.join(
                    global_config.fold_gen_path,
                    f"{site}/fold{f}_baseline_test.csv",
                )
            )
            val_baseline = pd.read_csv(
                path.join(
                    global_config.fold_gen_path,
                    f"{site}/fold{f}_baseline_val.csv",
                )
            )

            test_save = remove_missing(test_gt, test_baseline)
            val_save = remove_missing(val_gt, val_baseline)
            test_save.to_csv(
                path.join(save_path, f"fold{f}_test.csv"), index=False
            )
            val_save.to_csv(
                path.join(save_path, f"fold{f}_val.csv"), index=False
            )

    else:
        test_gt = pd.read_csv(
            path.join(global_config.split_path, f"{site}/test_gt.csv")
        )
        test_baseline = pd.read_csv(
            path.join(global_config.fold_gen_path, f"{site}/baseline_test.csv")
        )

        test_save = remove_missing(test_gt, test_baseline)
        test_save.to_csv(path.join(save_path, f"test_gt.csv"), index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--need_split", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
