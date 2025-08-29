#!/usr/bin/env python
# 1. Split main data csv and patient baselines into each folds according to
# folds generated in step 1.
# 2. Take note that we only save rows with train/val mask==1. This means that
# we only save input half of val and the entire train. This ensures that we don't
# have data leakage.
# 3. Generate submission templates for output halves of val fold and test.
# Note: `submission_df` refers to the template for evaluation-ready prediction dataframe.
# The term originates from TADPOLE-style outputs but does not imply leaderboard submission.
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


import argparse
from os import makedirs, path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config import global_config


def make_date_col(starts, duration, site):
    """
    Return a list of dates
    The start date of each list of dates is specified by *starts*
    """
    if site != "OASIS":
        date_range = [relativedelta(months=i) for i in range(1, duration + 1)]
    else:
        date_range = range(1, duration + 1)

    ret = []
    for start in starts:
        ret.append([start + d for d in date_range])

    return ret


def gen_submission(baseline, frame, site, forecast_horizon):
    """
    Generate a standardized dataframe of predictions aligned with the evaluation template.
    This is referred to as `submission_df` for historical consistency with TADPOLE.
    No external submission is performed in this project.
    The starting date of submission_df is the last timepoint of input half.
    We predict `forecast_horizon` months into the future.
    """
    submission = pd.DataFrame(
        columns=[
            "RID",
            "Forecast Month",
            "Forecast Date",
            "CN relative probability",
            "MCI relative probability",
            "AD relative probability",
            "MMSE",
            "MMSE 50% CI lower",
            "MMSE 50% CI upper",
            "Ventricles_ICV",
            "Ventricles_ICV 50% CI lower",
            "Ventricles_ICV 50% CI upper",
        ]
    )

    # extract test patients from baseline (either test_baseline or val_baseline)
    ptid = baseline.patients.unique()
    tadpole_test = frame[frame["RID"].isin(ptid)]

    # convert EXAMDATE from string to datetime
    tadpole_date = tadpole_test.copy(deep=True)
    if site != "OASIS":
        tadpole_date["EXAMDATE"] = pd.to_datetime(
            tadpole_date["EXAMDATE"], format="%Y-%m-%d"
        )

    # get the starting and end date for each test patient
    a = tadpole_date.groupby("RID").agg({"EXAMDATE": ["min", "max"]})
    # starting from the max date available (only until first half),
    # we predict into the future for X months
    starts = a["EXAMDATE"]["max"]
    future_dates = make_date_col(starts, forecast_horizon, site)

    submission_save = submission.copy(deep=True)
    for sub_id, sub in enumerate(a.index):
        insert_row = {
            "RID": sub,
            "Forecast Month": range(1, forecast_horizon + 1),
            "Forecast Date": future_dates[sub_id],
        }
        submission_save = pd.concat(
            [submission_save, pd.DataFrame(insert_row)]
        )

    return submission_save


def gen_test(frame, baseline, mask, save_dir, site, forecast_horizon):
    """
    Generate the test input for external datasets (entire dataset as test,
    no train-val-test split like ADNI)
    """

    # merge the test mask with frame
    new = frame.merge(mask, on=["RID", "EXAMDATE"], how="left")
    assert frame.shape[0] == new.shape[0], "Check duplicate rows in mask.csv"

    frame_save = new[(new.test == 1)]
    test_subj = frame_save[frame_save.test == 1].RID.unique()
    base_test_save = baseline[baseline["patients"].isin(test_subj)]
    submission_test = gen_submission(
        base_test_save, frame_save, site, forecast_horizon
    )
    frame_save.to_csv(path.join(save_dir, f"basic_test.csv"), index=False)
    base_test_save.to_csv(
        path.join(save_dir, f"baseline_test.csv"), index=False
    )
    submission_test.to_csv(
        path.join(save_dir, f"submission_test.csv"), index=False
    )


def gen_train_val(frame, baseline, mask, save_dir, args, fold):
    """
    Generate fold-wise input data for train-val-test (for ADNI)
    For training, all timepoints are considered as input.
    For val & test, only the first half timepoints are included to avoid data leakage.
    """

    # merge the test mask with frame
    new = frame.merge(mask, on=["RID", "EXAMDATE"], how="left")
    assert frame.shape[0] == new.shape[0], "Check duplicate rows in mask.csv"

    if args.independent_test:
        frame_save = new[(new.train == 1) | (new.val == 1)]
    else:
        frame_save = new[(new.train == 1) | (new.val == 1) | (new.test == 1)]

    # get the baselines for these selected subjects
    # need to separate train and test baselines

    val_subj = frame_save[frame_save.val == 1].RID.unique()
    train_subj = frame_save[frame_save.train == 1].RID.unique()
    base_train_save = baseline[baseline["patients"].isin(train_subj)]
    base_val_save = baseline[baseline["patients"].isin(val_subj)]
    submission_val = gen_submission(
        base_val_save, frame_save, args.site, args.forecast_horizon
    )
    frame_save.to_csv(
        path.join(save_dir, f"fold{fold}_basic.csv"), index=False
    )
    base_train_save.to_csv(
        path.join(save_dir, f"fold{fold}_baseline_train.csv"), index=False
    )
    base_val_save.to_csv(
        path.join(save_dir, f"fold{fold}_baseline_val.csv"), index=False
    )
    submission_val.to_csv(
        path.join(save_dir, f"fold{fold}_submission_val.csv"), index=False
    )

    if not args.independent_test:
        test_subj = frame_save[frame_save.test == 1].RID.unique()
        base_test_save = baseline[baseline["patients"].isin(test_subj)]
        submission_test = gen_submission(
            base_test_save, frame_save, args.site, args.forecast_horizon
        )
        base_test_save.to_csv(
            path.join(save_dir, f"fold{fold}_baseline_test.csv"), index=False
        )
        submission_test.to_csv(
            path.join(save_dir, f"fold{fold}_submission_test.csv"), index=False
        )


def main(args):
    """Main execution function."""

    # Set random seed
    np.random.seed(args.seed)

    # load spreadsheets
    frame = pd.read_csv(
        path.join(global_config.raw_data_path, f"{args.site}.csv"),
        low_memory=False,
    )
    baseline = pd.read_csv(
        path.join(
            global_config.baseline_path, f"{args.site}/patient_baselines.csv"
        ),
        low_memory=False,
    )

    # Generate the save directory
    save_path = path.join(global_config.fold_gen_path, f"{args.site}")
    makedirs(save_path, exist_ok=True)

    if args.independent_test:  # External test datasets
        test_mask = pd.read_csv(
            path.join(global_config.split_path, f"{args.site}/test_mask.csv"),
            low_memory=False,
        )
        test_mask = test_mask[~test_mask.duplicated(["RID", "EXAMDATE"])]
        gen_test(
            frame,
            baseline,
            test_mask,
            save_path,
            args.site,
            args.forecast_horizon,
        )

    if args.need_split:  # Train-val-test splits for ADNI
        for fold in range(20):
            val_mask = pd.read_csv(
                path.join(
                    global_config.split_path,
                    f"{args.site}/fold{fold}_mask.csv",
                ),
                low_memory=False,
            )
            val_mask = val_mask[~val_mask.duplicated(["RID", "EXAMDATE"])]
            gen_train_val(frame, baseline, val_mask, save_path, args, fold)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--need_split", action="store_true")
    parser.add_argument("--independent_test", action="store_true")
    parser.add_argument("--forecast_horizon", type=int, default=120)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
