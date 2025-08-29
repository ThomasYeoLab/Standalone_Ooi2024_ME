#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import os

import numpy as np
import pandas as pd


def load_adni_data(site, fold, data_path, split, use_columns, mri_features):
    """Function to load ADNI data and perform processing to fit AD-Map style

    Args:
        site: dataset site (we expect ADNI here, but can be other dataset with same style as ADNI)
        fold: integer indicating the fold to load
        data_path: data folder to locate the relevant files
        split: specify whether to load train/validation/test data
        use_columns: not all columns from the csv files are required, specify the columns to use
        mri_features: specify MRI columns to perform ICV normalization

    Return:
        df: processed dataframe
    """
    df = pd.read_csv(
        os.path.join(data_path, site, f"fold{fold}_basic.csv"),
        usecols=use_columns,
    )
    df.rename(columns={"RID": "ID", "curr_age": "TIME"}, inplace=True)
    base_df = pd.read_csv(
        os.path.join(data_path, site, f"fold{fold}_baseline_{split}.csv")
    )

    # List of brain ROI volume columns that need to be normalized
    roi_columns = list(mri_features.keys())
    # Normalize each ROI column by ICV inplace
    df[roi_columns] = df[roi_columns].div(df["ICV"], axis=0)

    # remove rows with missing TIME column as AD-Map requires TIME to function
    # (might lead to some subjects having no data)
    df = df[df.TIME.notnull()]

    # Define the target time and the tolerance threshold (for removing duplicate)
    # as AD-Map doesn't allow duplicate visits
    target_time = 80.878142
    tolerance = 1e-6

    # Filter rows where ID is 1195 and TIME is approximately 80.878142
    duplicated_rows = df[
        (df["ID"] == 1195) & (df["TIME"].sub(target_time).abs() < tolerance)
    ]

    # Drop the row where 'Hippocampus' is NaN
    df = df.drop(duplicated_rows[duplicated_rows["Hippocampus"].isna()].index)

    df = df[df.ID.isin(base_df.patients.unique())].copy()

    # Cast ID type to string to prevent leapsy bug
    df["ID"] = df["ID"].astype(str)

    return df


def load_external_data(site, data_path, use_columns, mri_features):
    """Function to load data from other sites and perform processing to fit AD-Map style

    Args:
        site: dataset site (e.g., AIBL, MACC, OASIS)
        fold: integer indicating the fold to load
        data_path: data folder to locate the relevant files
        use_columns: not all columns from the csv files are required, specify the columns to use
        mri_features: specify MRI columns to perform ICV normalization

    Return:
        df_test: processed dataframe
    """
    df = pd.read_csv(
        os.path.join(data_path, site, f"basic_test.csv"), usecols=use_columns
    )
    df.rename(columns={"RID": "ID", "AGE": "TIME"}, inplace=True)

    base_test = pd.read_csv(os.path.join(data_path, site, "baseline_test.csv"))

    # List of brain ROI volume columns that need to be normalized
    roi_columns = list(mri_features.keys())
    # Normalize each ROI column by ICV inplace
    df[roi_columns] = df[roi_columns].div(df["ICV"], axis=0)

    # remove rows with missing TIME column
    df = df[df.TIME.notnull()]

    df_test = df[df.ID.isin(base_test.patients.unique())].copy()

    # Cast ID type to string to prevent leapsy bug
    df_test["ID"] = df_test["ID"].astype(str)

    if site == "OASIS":
        # 23: OAS30004, 1403: OAS30447, 1512: OAS30484, 3145: OAS30970 , 3857: OAS31203
        rows_to_drop = [23, 1403, 1512, 3145, 3857]  # AD_dem rows to drop
        df_test = df_test.drop(rows_to_drop)
    return df_test


def normalize(df, feat, max_, min_, increase=True):
    """Helper function to normalize the feature columns into range (0, 1)

    Args:
        df: dataframe to normalize
        feat: feature column to normalize
        max_: the maximum value of the feature (saved from train statistics)
        min_: the minimum value of the feature (saved from train statistics)
        increase: whether a larger value indicate worse progression

    Return:
        df_study: normalized dataframe
    """
    df_study = df[feat].copy()
    df_study = (df_study - min_) / (max_ - min_)
    if not increase:
        df_study = 1 - df_study
    return df_study


def preprocess_data(df, mri_features, scores=None):
    """
    Pre-process numeric data into range (0-1) taking into account the value direction
    (i.e., whether a higher value indicate worse condition)
    For MRI features, need to first clip the data based on 1st and 99th percentiles
    Save the bounded scores to pre-process test data using training statistics
    """
    df_clipped = df.copy()
    save_flag = False
    if scores is None:  # train_df
        # Loop through each feature and clip it at the 1st and 99th percentiles
        for feature in mri_features.keys():
            # Compute the 1st and 99th percentiles
            lower_bound = df[feature].quantile(0.01)
            upper_bound = df[feature].quantile(0.99)
            # Clip the feature values to this range
            df_clipped[feature] = df[feature].clip(
                lower=lower_bound, upper=upper_bound
            )

        # Save bounded scores
        scores = {
            "MMSE": (30, 0, False),  # max, min, increase?
            "CDR": (3, 0, True),
        }
        for feature, increasing in mri_features.items():
            scores[feature] = (
                df_clipped[feature].max(),
                df_clipped[feature].min(),
                increasing,
            )

        std_dict = {
            "MMSEstd": df_clipped["MMSE"].std(),
            "VentICVstd": df_clipped["Ventricles"].std(),
        }  # already normalized by ICV
        save_flag = True
    else:
        # Loop through each feature, extract bounds from scores, then clip test/val df
        for feature in mri_features.keys():
            lower_bound = scores[feature][1]
            upper_bound = scores[feature][0]
            # Clip the feature values to this range
            df_clipped[feature] = df[feature].clip(
                lower=lower_bound, upper=upper_bound
            )

    for score_name, normalize_args in scores.items():
        df_clipped[score_name] = normalize(
            df_clipped, score_name, *normalize_args
        )

    if save_flag:
        return df_clipped, scores, std_dict
    else:
        return df_clipped


def transform_submission(submission_test, df_test):
    """
    Transform the submission dataframe to better accomodate the predictions
    """
    submission_test["RID"] = submission_test["RID"].astype(str)
    submission_test.rename(columns={"RID": "ID"}, inplace=True)

    # Step 1: Compute the maximum TIME for each ID in the input_df
    max_time_per_id = df_test.groupby("ID")["TIME"].max().reset_index()
    max_time_per_id.rename(columns={"TIME": "Max_TIME"}, inplace=True)

    # Step 2: Merge the maximum TIME back to the submission_df
    submission_df = submission_test.merge(max_time_per_id, on="ID")

    # Step 3: Convert Forecast Month to Years
    submission_df["Forecast Month in Years"] = (
        submission_df["Forecast Month"] / 12
    )

    # Step 4: Add the Max TIME to the Forecast Month (in years)
    submission_df["TIME"] = (
        submission_df["Max_TIME"] + submission_df["Forecast Month in Years"]
    )

    # Step 5: Drop the intermediate columns if not needed
    submission_df.drop(
        columns=["Max_TIME", "Forecast Month in Years"], inplace=True
    )

    return submission_df


def reverse_normalize(y, max_, min_, increase=True):
    """
    Reverse the normalization to get back original prediction scale
    """
    if not increase:
        y = 1 - y
    x = y * (max_ - min_) + min_
    return x


def get_probabilities(cdr, epsilon=1e-6):
    """
    Function to convert CDR global score to diagnosis probabilities
    """
    if cdr <= 0.5:
        # CN: 1 -> 0, MCI: 0 -> 1, AD: 0
        CN_prob = np.interp(cdr, [0, 0.5], [1, 0])
        MCI_prob = np.interp(cdr, [0, 0.5], [0, 1])
        AD_prob = epsilon
    elif cdr <= 1:
        # CN: 0, MCI: 1 -> 0, AD: 0 -> 1
        CN_prob = epsilon
        MCI_prob = np.interp(cdr, [0.5, 1], [1, 0])
        AD_prob = np.interp(cdr, [0.5, 1], [0, 1])
    else:
        # CN: 0, MCI: 0, AD: 1
        CN_prob = epsilon
        MCI_prob = epsilon
        AD_prob = 1
    return CN_prob, MCI_prob, AD_prob
