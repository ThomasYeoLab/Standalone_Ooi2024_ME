#!/usr/bin/env python
# Generate Frog features by squashing longitudinal data
# into cross-sectional (L2C transformation).
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# Standard library
import argparse
from os import makedirs, path

# Third-party libraries
import numpy as np
import pandas as pd

# Local application/library imports
from config import global_config
from data_processing.misc import (
    create_dx_features,
    create_long_features,
    gen_feature_names,
)


def assemble_feature_vector(
    pat_bl,
    pat,
    current_idx,
    age_col,
    data_df,
    pat_index,
    history_idx,
    features,
    use_adas,
):
    """
    Assemble the L2C features for each (current_idx, history_idx)
    pair within the for loop for training
    """
    feature_vector = [
        pat_bl["patients"],
        pat.iloc[0, pat.columns.get_loc("RID")],
        pat_bl["APOE"],
        pat_bl["isMale"],
        pat_bl["educ"],
        pat_bl["married"],
    ]

    feature_vector.append(
        pat.iloc[current_idx, pat.columns.get_loc(age_col)]
    )  # curr_age
    feature_vector.append(
        pat.iloc[current_idx, pat.columns.get_loc("Month_bl")]
    )
    feature_vector.append(
        pat.iloc[current_idx, pat.columns.get_loc("DX")]
    )  # curr_dx_numeric
    if use_adas:
        feature_vector.append(
            pat.iloc[current_idx, pat.columns.get_loc("ADAS13")]
        )
    else:
        feature_vector.append(
            pat.iloc[current_idx, pat.columns.get_loc("MMSE")]
        )
    feature_vector.append(
        pat.iloc[current_idx, pat.columns.get_loc("Ventricles")]
    )
    feature_vector.append(pat.iloc[current_idx, pat.columns.get_loc("ICV")])

    feature_vector += create_dx_features(
        data_df["DX"], data_df["Month_bl"], pat_index, history_idx, current_idx
    )  # curr_dx_numeric
    for feat in features:
        if feat in [
            "Ventricles",
            "Fusiform",
            "WholeBrain",
            "Hippocampus",
            "MidTemp",
        ]:
            feature_vector += create_long_features(
                data_df[feat] / data_df["ICV"],
                data_df["Month_bl"],
                pat_index,
                history_idx,
                current_idx,
            )
        else:
            feature_vector += create_long_features(
                data_df[feat],
                data_df["Month_bl"],
                pat_index,
                history_idx,
                current_idx,
            )
    return feature_vector


# ------------------------------------ Create Training Set ------------------------------------
def gen_L2C_train(
    data_df, base_train, long_save_path, features, use_adas, site
):
    """L2C transformation for trining set (all timepoints used)

    Args:
        data_df (dataframe): train_basic, contains all timepoints of training subjects
                and first half timepoints of val/test subjects
        base_train (dataframe): baseline features for training subjects only
        long_save_path (csv path): save path for generated dataframe
        features (list): features to include in L2C transformation
        use_adas (bool): whether predict ADAS13 (True) or MMSE (False)
        site (str): training site

    Returns:
        None
    """
    train_long = []

    for ii, pat_bl in base_train.iterrows():
        pat = data_df[data_df.RID == pat_bl.patients]
        pat = pat.sort_values(
            by="Month_bl", ascending=True, na_position="last"
        )
        pat_index = pat.index

        age_col = "curr_age" if site == "ADNI" else "AGE"

        if pat.shape[0] < 2:
            continue

        for current_idx in range(1, pat.shape[0]):
            for history_idx in range(1, current_idx + 1):
                feature_vector = assemble_feature_vector(
                    pat_bl,
                    pat,
                    current_idx,
                    age_col,
                    data_df,
                    pat_index,
                    history_idx,
                    features,
                    use_adas,
                )
                train_long.append(feature_vector)

    cog_score = "curr_adas13" if use_adas else "curr_mmse"
    colnames = [
        "ptid",
        "rid",
        "apoe",
        "male",
        "educ",
        "married",
        "curr_age",
        "month_bl",
        "curr_dx",
        cog_score,
        "curr_ventricles",
        "curr_icv",
    ] + gen_feature_names("dx")
    for feat in features:
        colnames += gen_feature_names(feat)

    train_df = pd.DataFrame(train_long, columns=colnames)
    train_df.to_csv(long_save_path, index=False)


# ------------------------------------ Create Tadpole Prediction Set ------------------------------------
def assemble_feature_test(
    pat_bl,
    pat,
    time_index,
    pat_index,
    age_col,
    days_till_time0,
    features,
):
    feature_vector = [
        pat_bl["patients"],
        pat.iloc[0, pat.columns.get_loc("RID")],
        time_index + 1,
        pat_bl["APOE"],
        pat_bl["isMale"],
        pat_bl["educ"],
        pat_bl["married"],
    ]

    feature_vector.append(
        pat[age_col].max() + (days_till_time0 + time_index * 30.25) / 365
    )
    pred_month_bl = (
        pat["Month_bl"].max() + days_till_time0 / 365 * 12 + time_index
    )
    feature_vector.append(pred_month_bl)

    # current_idx and history_idx are the same now
    # as all past visits are included in L2C transformation now
    current_idx = len(pat_index)
    feature_vector += create_dx_features(
        pat["DX"],
        pd.concat([pat["Month_bl"], pd.Series(pred_month_bl)]),
        range(current_idx + 1),
        current_idx,
        current_idx,
    )

    for feat in features:
        if feat in [
            "Ventricles",
            "Fusiform",
            "WholeBrain",
            "Hippocampus",
            "MidTemp",
        ]:
            feature_vector += create_long_features(
                pat[feat] / pat["ICV"],
                pd.concat([pat["Month_bl"], pd.Series(pred_month_bl)]),
                range(current_idx + 1),
                current_idx,
                current_idx,
            )
        else:
            feature_vector += create_long_features(
                pat[feat],
                pd.concat([pat["Month_bl"], pd.Series(pred_month_bl)]),
                range(current_idx + 1),
                current_idx,
                current_idx,
            )
    return feature_vector


def gen_L2C_test(
    data_df, base_test, test_save_path, features, site, forecast_horizon
):
    """L2C transformation for val/test set (only input half timepoints used)

    Args:
        data_df (dataframe): test_basic, contains all timepoints of training subjects
                and first half timepoints of val/test subjects
        base_test (dataframe): baseline features for val/test subjects only
        test_save_path (csv path): save path for generated dataframe
        features (list): features to include in L2C transformation
        site (str): training site
        forecast_horizon (int): how far into the future to predict. We need to generate L2C
                input features for each month into the future

    Returns:
        None
    """

    test_long = []
    age_col = "curr_age" if site == "ADNI" else "AGE"

    for ii, pat_bl in base_test.iterrows():

        pat = data_df[data_df.RID == pat_bl.patients]

        pat = pat.sort_values(
            by="Month_bl", ascending=True, na_position="last"
        )
        pat_index = pat.index

        # we start prediction from the last point in input half
        # unlike Frog starting from reference time. Thus, the gap
        # between input and prediction starting point is only 1 month (30.25 days)
        days_till_time0 = 30.25

        # give predictions for X months
        for time_index in range(forecast_horizon):
            feature_vector = assemble_feature_test(
                pat_bl,
                pat,
                time_index,
                pat_index,
                age_col,
                days_till_time0,
                features,
            )
            test_long.append(feature_vector)

    colnames = [
        "ptid",
        "rid",
        "forecast_month",
        "apoe",
        "male",
        "educ",
        "married",
        "curr_age",
        "month_bl",
    ] + gen_feature_names("dx")

    for feat in features:
        colnames += gen_feature_names(feat)
    test_inp_df = pd.DataFrame(test_long, columns=colnames)
    test_inp_df.to_csv(test_save_path, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default=0)
    parser.add_argument("--site", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--use_adas", action="store_true")
    parser.add_argument("--need_split", action="store_true")
    parser.add_argument("--independent_test", action="store_true")
    parser.add_argument("--forecast_horizon", type=int, default=120)

    return parser.parse_args()


def main(args):
    """
    use_full: use all 23 features, otherwise only use the common 10
    use_adas: whether predict ADAS13 (TADPOLE) or MMSE (DG)
    independent_test: whether we use the entire dataset as test or perform split
    need_split: whether generate the data for 20-folds split (train/val)
    If test on entire external dataset, no need to generate train/val, since we
    only train/val on ADNI.
    """
    np.random.seed(args.seed)

    # print(f"current site: {args.site}, current fold: {args.fold}")

    feature_dict = global_config.feature_dict
    feature_set = "FULL" if args.use_full else "PARTIAL"
    fold_gen_dir = (
        global_config.fold_gen_full
        if args.use_full
        else global_config.fold_gen_path
    )
    features = feature_dict[feature_set]

    # Generate the save directory
    save_path = path.join(global_config.frog_path, f"{args.site}")
    makedirs(save_path, exist_ok=True)

    if args.independent_test:  # Only input L2C features for the external test
        base_test_path = path.join(
            fold_gen_dir, f"{args.site}/baseline_test.csv"
        )
        test_basic_path = path.join(
            fold_gen_dir, f"{args.site}/basic_test.csv"
        )
        base_test = pd.read_csv(base_test_path)
        test_basic = pd.read_csv(test_basic_path, low_memory=False)
        test_save_path = path.join(save_path, "test.csv")
        gen_L2C_test(
            test_basic,
            base_test,
            test_save_path,
            features,
            args.site,
            args.forecast_horizon,
        )

    if args.need_split:  # train-val-test for ADNI
        data_basic_path = path.join(
            fold_gen_dir, f"{args.site}/fold{args.fold}_basic.csv"
        )
        base_train_path = path.join(
            fold_gen_dir, f"{args.site}/fold{args.fold}_baseline_train.csv"
        )
        base_val_path = path.join(
            fold_gen_dir, f"{args.site}/fold{args.fold}_baseline_val.csv"
        )

        data_df = pd.read_csv(data_basic_path, low_memory=False)
        base_train = pd.read_csv(base_train_path)
        base_val = pd.read_csv(base_val_path)

        long_save_path = path.join(save_path, f"fold{args.fold}_train.csv")
        val_save_path = path.join(save_path, f"fold{args.fold}_val.csv")
        gen_L2C_train(
            data_df,
            base_train,
            long_save_path,
            features,
            args.use_adas,
            args.site,
        )
        gen_L2C_test(
            data_df,
            base_val,
            val_save_path,
            features,
            args.site,
            args.forecast_horizon,
        )

        if not args.independent_test:
            base_test_path = path.join(
                fold_gen_dir, f"{args.site}/fold{args.fold}_baseline_test.csv"
            )
            base_test = pd.read_csv(base_test_path)
            test_save_path = path.join(save_path, f"fold{args.fold}_test.csv")
            gen_L2C_test(
                data_df,
                base_test,
                test_save_path,
                features,
                args.site,
                args.forecast_horizon,
            )


if __name__ == "__main__":
    main(get_args())
