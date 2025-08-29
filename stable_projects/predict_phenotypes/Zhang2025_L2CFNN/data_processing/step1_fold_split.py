#!/usr/bin/env python
# Split each dataset into 20 folds with 80% training and 20% validation.
# For training set, keep everything; for val set, split into in/out halves.
# Meanwhile, create the test version of the dataset. It should include
# all patients with > 2 tps, split into input and output half
# Input will be indicated with a mask, output half ground truth (for both
# val and test sets) will be saved as csv files.
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
from os import makedirs, path

import data_processing.misc as misc
import numpy as np
import pandas as pd

from config import global_config


def split_by_median_date(data, subjects):
    """
    Split timepoints in two halves, use first half to predict second half
    Even if there's no data or only 1 tp, we still keep it, but in mask it'll
    always be 0, and won't be used.
    2024.10.3 note: thanks to Minh's way of splitting, he used EXAMDATE and compare
    with median_date, which effectively filtered all NaN EXAMDATE so they won't be used
    in either traning or testing. Otherwise this will affect the sort in Frog_feature generation.
    Also this keeps the original order of the mask. The lesson here is don't sort the data
    then simplly use head() function to take the first half, instead use date comparison!
    Args:
        data (Pandas data frame): input data
        subjects: list of subjects
    Return:
        first_half (ndarray): boolean mask, rows used as input
        second_half (ndarray): boolean mask, rows to predict
    """
    first_half = np.zeros(data.shape[0], int)
    second_half = np.zeros(data.shape[0], int)
    for rid in subjects:
        subj_mask = (data.RID == rid) & data.has_data
        median_date = np.sort(data.EXAMDATE[subj_mask])[subj_mask.sum() // 2]
        first_half[subj_mask & (data.EXAMDATE < median_date)] = 1
        second_half[subj_mask & (data.EXAMDATE >= median_date)] = 1
    return first_half, second_half


def gen_mask_frame(data, train, val, test, args):
    """
    Create a frame with 3 masks:
        train: timepoints used for training model
        val: timepoints used for validation
        test: timepoints used for testing model
    """
    col = ["RID", "EXAMDATE"]
    ret = pd.DataFrame(data[col], index=range(train.shape[0]))
    ret["train"] = train
    if args.independent_test:
        ret["val"] = val + test
    else:
        ret["val"] = val
        ret["test"] = test

    return ret


def gen_mask_test(data, test):
    """
    Create mask frame for test indicating input/output half
    The mask contains all the subjects & timepoints, either 0/1
    """
    col = ["RID", "EXAMDATE"]
    ret = pd.DataFrame(data[col], index=range(test.shape[0]))
    ret["test"] = test
    return ret


def gen_ref_frame(data, test_timepoint_mask, args):
    """Create reference frame which is used to evalute models' prediction"""
    cog_score = "ADAS13" if args.use_adas else "MMSE"
    columns = [
        "RID",
        "CognitiveAssessmentDate",
        "Diagnosis",
        cog_score,
        "ScanDate",
    ]
    ret = pd.DataFrame(
        np.nan, index=range(len(test_timepoint_mask)), columns=columns
    )

    ret[columns] = data[["RID", "EXAMDATE", "DX", cog_score, "SCANDATE"]]
    ret["Ventricles"] = data["Ventricles"] / data["ICV"]
    ret = ret[test_timepoint_mask == 1]

    # map diagnosis from numeric categories back to labels
    # we've changed the raw data csv, eaiser for here and later

    mapping = {0: "CN", 1: "MCI", 2: "AD"}

    ret.replace({"Diagnosis": mapping}, inplace=True)
    ret.reset_index(drop=True, inplace=True)

    return ret


def gen_fold(data, args):
    """Generate *nb_folds* cross-validation folds from *data"""
    subjects = np.unique(data.RID)
    has_2tp = np.array([np.sum(data.RID == rid) >= 2 for rid in subjects])

    # Generate the save directory
    save_path = path.join(global_config.split_path, f"{args.site}")
    makedirs(save_path, exist_ok=True)

    # generate reference for test copy if leave-one-out setting
    if args.independent_test:
        test_subj = subjects[has_2tp]
        test_in_timepoints, test_out_timepoints = split_by_median_date(
            data, test_subj
        )
        test_mask = gen_mask_test(data, test_in_timepoints)
        test_gt = gen_ref_frame(data, test_out_timepoints, args)

        test_mask.to_csv(path.join(save_path, "test_mask.csv"), index=False)
        test_gt.to_csv(path.join(save_path, "test_gt.csv"), index=False)

    # only generate 20-fold split if we need it (i.e., for ADNI training)
    if args.need_split:
        # we only take subjects with at least 2 timepoints
        potential_targets = np.random.permutation(subjects[has_2tp])
        folds = np.array_split(potential_targets, args.nb_folds)

        for test_fold in range(args.nb_folds):
            val_fold = (test_fold + 1) % args.nb_folds
            train_folds = [
                i
                for i in range(args.nb_folds)
                if (i != test_fold and i != val_fold)
            ]

            train_subj = np.concatenate(
                [folds[i] for i in train_folds], axis=0
            )
            val_subj = folds[val_fold]
            test_subj = folds[test_fold]

            # all timepoints are used for training
            train_timepoints = (
                np.in1d(data.RID, train_subj) & data.has_data
            ).astype(int)

            # split into input/output halves for val and test
            val_in_timepoints, val_out_timepoints = split_by_median_date(
                data, val_subj
            )
            test_in_timepoints, test_out_timepoints = split_by_median_date(
                data, test_subj
            )

            # depending on leave_one_out, mask and GT generation will differ.
            mask_frame = gen_mask_frame(
                data,
                train_timepoints,
                val_in_timepoints,
                test_in_timepoints,
                args,
            )

            mask_frame.to_csv(
                path.join(save_path, f"fold{test_fold}_mask.csv"), index=False
            )

            if (
                args.independent_test
            ):  # both val & test timepoints used for val
                val_frame = gen_ref_frame(
                    data, val_out_timepoints + test_out_timepoints, args
                )
                val_frame.to_csv(
                    path.join(save_path, f"fold{test_fold}_val.csv"),
                    index=False,
                )
            else:
                val_frame = gen_ref_frame(data, val_out_timepoints, args)
                val_frame.to_csv(
                    path.join(save_path, f"fold{test_fold}_val.csv"),
                    index=False,
                )
                test_frame = gen_ref_frame(data, test_out_timepoints, args)
                test_frame.to_csv(
                    path.join(save_path, f"fold{test_fold}_test.csv"),
                    index=False,
                )


def main(args):
    """Main execution function."""

    # Set random seed, load feature columns
    np.random.seed(args.seed)
    feature_dict = global_config.feature_dict
    feature_set = "MinFull" if args.use_full else "PARTIAL"
    features = feature_dict[feature_set]

    # load spreadsheet using proper converters
    frame_path = path.join(global_config.raw_data_path, f"{args.site}.csv")
    if args.site == "OASIS":  # OASIS doesn't have EXAMDATE in %Y-%m-%d format
        frame = pd.read_csv(frame_path, low_memory=False)
    else:
        frame = pd.read_csv(
            frame_path,
            converters=misc.CONVERTERS,
            low_memory=False,
        )

    available_features = [feat for feat in features if feat in frame.columns]
    frame["has_data"] = (
        ~frame[available_features].isnull().apply(np.all, axis=1)
    )
    gen_fold(frame, args)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nb_folds", type=int, default=20)
    parser.add_argument("--independent_test", action="store_true")
    parser.add_argument("--need_split", action="store_true")
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--use_adas", action="store_true")
    parser.add_argument("--site", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
