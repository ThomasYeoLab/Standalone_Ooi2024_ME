#!/usr/bin/env python
# Helper functions to to handle divergent logic
# between the three tasks cleanly so we can merge them
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import numpy as np
import pandas as pd


def _process_predictions_clin_step1(
    yhat_agg,
    X_train_features,
    X_test_features,
    tr_subset_with_target,
    l2c_train_df_filtered,
):
    """
    Handles clin-specific processing: historical distribution and filling missing predictions.
    Step1: within partition filling
    """
    class_prob_mat = np.zeros((3, 3), dtype=float)

    # Ensure 'mr_dx' is in X_train_features and 'curr_dx' in tr_subset_with_target
    assert "mr_dx" in X_train_features.columns, "mr_dx not in train features!"
    assert (
        "curr_dx" in tr_subset_with_target.columns
    ), "curr_dx not in train_df_with_target"
    for kk_dx in range(3):  # mr_dx can be 0, 1, 2 (e.g., CN, MCI, DEM)
        baseline_diagnosis_mask = X_train_features["mr_dx"] == kk_dx
        # if there is training data in the current window
        if baseline_diagnosis_mask.sum() > 0:
            # Get 'curr_dx' from the original filtered training data that includes the target
            relevant_curr_dx = tr_subset_with_target.loc[
                baseline_diagnosis_mask, "curr_dx"
            ]
            for ll_dx in range(3):
                class_prob_mat[kk_dx, ll_dx] = (
                    relevant_curr_dx == ll_dx
                ).sum() / baseline_diagnosis_mask.sum()
        # otherwise, we use data from the entire training set, ignore window
        else:
            baseline_diagnosis_mask = l2c_train_df_filtered["mr_dx"] == kk_dx
            # Get 'curr_dx' from the entire training data
            relevant_curr_dx = l2c_train_df_filtered.loc[
                baseline_diagnosis_mask, "curr_dx"
            ]
            for ll_dx in range(3):
                class_prob_mat[kk_dx, ll_dx] = (
                    relevant_curr_dx == ll_dx
                ).sum() / baseline_diagnosis_mask.sum()

    # Fill missing predictions using the class probability matrix
    assert "mr_dx" in X_test_features.columns, "mr_dx not in test features!"
    missing_indices = np.argwhere(np.isnan(np.sum(yhat_agg, axis=1)))
    if len(missing_indices) > 0:
        for idx in missing_indices:
            recent_dx_val = X_test_features["mr_dx"].iloc[idx]
            fill_probs = class_prob_mat[int(recent_dx_val)]
            yhat_agg[idx, :] = fill_probs
    return yhat_agg


def _process_predictions_clin_step2(
    yhat_agg, l2c_train_df_filtered, l2c_test_df_full
):
    """For cases when time_since_mr_dx is missing, need to run 2nd imputation"""
    class_prob_mat = np.zeros((3, 3), dtype=float)
    for kk_dx in range(3):
        baseline_diagnosis_mask = l2c_train_df_filtered["mr_dx"] == kk_dx
        relevant_curr_dx = l2c_train_df_filtered.loc[
            baseline_diagnosis_mask, "curr_dx"
        ]
        for ll_dx in range(3):
            class_prob_mat[kk_dx, ll_dx] = (
                relevant_curr_dx == ll_dx
            ).sum() / baseline_diagnosis_mask.sum()

    missing_indices = np.argwhere(np.isnan(np.sum(yhat_agg, axis=1)))

    if len(missing_indices) > 0:
        for idx in missing_indices:
            # if not pd.isnull(l2c_test_df_full["mr_dx"].iloc[idx]).item():
            if not pd.isnull(l2c_test_df_full["mr_dx"].iloc[idx]).item():
                recent_dx = int(l2c_test_df_full["mr_dx"].iloc[idx])
                yhat_agg[idx, :] = class_prob_mat[recent_dx]
            else:
                yhat_agg[idx, :] = np.nanmedian(
                    yhat_agg[l2c_test_df_full["forecast_month"] == 1], axis=0
                )
    return yhat_agg
