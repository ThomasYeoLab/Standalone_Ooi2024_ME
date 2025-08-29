#!/usr/bin/env python
# Helper functions to to handle divergent logic
# between the three tasks cleanly so we can merge them
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import numpy as np
import pandas as pd
import xgboost as xgb

from models.L2C_XGBnw.model_config import model_config


def _train_cn_models_mmse(
    xgb_params_train,
    X_train_features,
    y_train_target,
    X_test_features,
    num_ensemble_models,
    args_obj,
    partition_idx_str="",
):  # partition_idx_str for windowed models
    """
    Trains Cognitively Normal (CN) partition models for MMSE target.
    Returns a dictionary of trained CN models.
    """
    cn_model_dict = {}
    # Condition for training CN models (from original code)
    # Ensure 'mr_dx' is present in X_train_features
    if "mr_dx" not in X_train_features.columns:
        print(
            "Warning (_train_cn_models_mmse): 'mr_dx' not in X_train_features. Skipping CN model training."
        )
        return cn_model_dict

    # Check if there are enough samples with mr_dx < 2 (CN or MCI) for training
    # and if there are any mr_dx == 0 (CN) samples in the *validation* set

    if (
        len(X_train_features["mr_dx"] < 2)
        > 1200 & len(X_test_features["mr_dx"] == 0)
        > 0
    ):
        # print(f"{partition_idx_str}_passed_check")
        # Filter training data for CN/MCI (mr_dx < 2)
        X_train_cn_subset = X_train_features[X_train_features["mr_dx"] < 2]
        y_train_cn_subset = y_train_target[X_train_features["mr_dx"] < 2]

        if X_train_cn_subset.empty or y_train_cn_subset.empty:
            print(
                f"Warning (_train_cn_models_mmse{partition_idx_str}): "
                "Empty data for CN model training after filtering. Skipping."
            )
            return cn_model_dict

        dtrain_cn = xgb.DMatrix(X_train_cn_subset, label=y_train_cn_subset)
        watchlist_cn = [(dtrain_cn, "train")]

        # Determine num_boost_round for CN models, could be different
        num_b_cn = (
            10
            if args_obj.debug
            else model_config.num_boost_round_dict["mmse_CN"]
        )

        for i_model in range(
            num_ensemble_models
        ):  # num_ensemble_models for CN might be different
            param_cn_iter = xgb_params_train.copy()
            # Ensure seed is distinct for this sub-model
            param_cn_iter["seed"] = i_model

            gbm_cn = xgb.train(
                param_cn_iter,
                dtrain_cn,
                num_boost_round=num_b_cn,
                evals=watchlist_cn,
                verbose_eval=False,
            )
            cn_model_key = f"CN_partition{partition_idx_str}_seed_{i_model}"
            cn_model_dict[cn_model_key] = gbm_cn
    else:
        print(
            f"Info (_train_cn_models_mmse{partition_idx_str}): Conditions not met for CN model training."
        )

    return cn_model_dict


def _process_predictions_clin(
    yhat_agg, X_train_features, X_test_features, train_df_filtered_with_target
):
    """Handles clin-specific processing: historical distribution and filling missing predictions."""
    class_prob_mat = np.full((3, 3), 0)

    # Ensure 'mr_dx' is in X_train_features and 'curr_dx' in train_df_filtered_with_target
    assert "mr_dx" in X_train_features.columns, "mr_dx not in train features!"
    assert (
        "curr_dx" in train_df_filtered_with_target.columns
    ), "curr_dx not in train_df_with_target"
    for kk_dx in range(2):  # mr_dx can be 0, 1, 2 (e.g., CN, MCI, DEM)
        baseline_diagnosis_mask = X_train_features["mr_dx"] == kk_dx
        if baseline_diagnosis_mask.sum() > 0:
            # Get 'curr_dx' from the original filtered training data that includes the target
            relevant_curr_dx = train_df_filtered_with_target.loc[
                baseline_diagnosis_mask, "curr_dx"
            ]
            for ll_dx in range(2):
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


def _process_predictions_regression_step1(yhat_agg, X_test_features, target):
    """Fill missing yhat_agg (first pass using mr_Ventricles/mr_MMSE from X_test_features)."""
    assert target in (
        "mmse",
        "vent",
    ), "target must be either 'mmse' or 'vent'!"
    mr_col_name = "mr_MMSE" if target == "mmse" else "mr_Ventricles"
    assert (
        mr_col_name in X_test_features.columns
    ), f"{mr_col_name} not in test features!"
    missing_indices = np.argwhere(np.isnan(yhat_agg)).flatten()
    if len(missing_indices) > 0:
        for idx in missing_indices:
            mr_feat_val = X_test_features[mr_col_name].iloc[idx]
            if not pd.isnull(mr_feat_val):
                yhat_agg[idx] = mr_feat_val
            else:
                yhat_agg[idx] = X_test_features[mr_col_name].median()
    return yhat_agg


def _process_predictions_regression_step2(yhat_agg, l2c_test_df_full, target):
    """Fill remaining missing yhat_agg (second pass, using mr_Ventricles/mr_MMSE from full l2c_test_df)."""
    mr_col_name = "mr_MMSE" if target == "mmse" else "mr_Ventricles"
    assert (
        mr_col_name in l2c_test_df_full.columns
    ), f"{mr_col_name} not in test_full df!"
    assert (
        "forecast_month" in l2c_test_df_full.columns
    ), "forecast_month not in test_full df!"
    missing_indices_pass2 = np.argwhere(np.isnan(yhat_agg)).flatten()
    if len(missing_indices_pass2) > 0:
        for idx in missing_indices_pass2:
            mr_feat_val_full = l2c_test_df_full[mr_col_name].iloc[
                idx
            ]  # Using l2c_test_df_full here
            if not pd.isnull(mr_feat_val_full):
                yhat_agg[idx] = mr_feat_val_full
            else:
                month1_preds = yhat_agg[
                    l2c_test_df_full["forecast_month"] == 1
                ]
                yhat_agg[idx] = np.nanmedian(month1_preds)
    return yhat_agg


def _update_submission_clin(
    final_submission_df, predictions_arr, l2c_test_df_full, raw_df
):
    """Updates the submission DataFrame for 'clin' target."""
    # Assuming DX_cols are the 3 relevant columns for CN, MCI, DEM probabilities
    DX_cols = final_submission_df.columns[3:6]

    rids_in_submission = final_submission_df["RID"].unique()
    raw_df["EXAMDATE"] = pd.to_datetime(raw_df["EXAMDATE"])

    for rid_val in rids_in_submission:
        model_pred_mask = l2c_test_df_full["rid"] == rid_val
        submission_mask = final_submission_df["RID"] == rid_val
        num_to_fill_submission = submission_mask.sum()

        if model_pred_mask.sum() > 0:  # RID has predictions from the model
            assert model_pred_mask.sum() == num_to_fill_submission, (
                f"RID {rid_val} (clin): l2c_test_df sum {model_pred_mask.sum()} "
                f"!= submission sum {num_to_fill_submission}"
            )

            final_submission_df.loc[submission_mask, DX_cols] = (
                predictions_arr[model_pred_mask]
            )
        else:  # RID not in predictions, use raw_df for fallback
            pat_history = raw_df[
                raw_df["RID"] == rid_val
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning
            pat_history = pat_history.sort_values(
                by="EXAMDATE", ascending=True, na_position="first"
            )

            last_dx_records = pat_history[pat_history["DX"].notna()]
            if not last_dx_records.empty:
                last_known_dx = last_dx_records["DX"].iloc[-1]
                # this outer product is equivalent to repeat or tile
                final_submission_df.loc[submission_mask, DX_cols] = np.outer(
                    np.full(submission_mask.sum(), 1),
                    np.nanmean(
                        predictions_arr[
                            (l2c_test_df_full["mr_dx"] == last_known_dx)
                            & (l2c_test_df_full["forecast_month"] == 1)
                        ],
                        axis=0,
                    ),
                )
            else:  # No DX history, try global average for forecast_month=1
                print("Warning: Predicting using nothing.")
                global_avg_month1 = np.nanmean(
                    predictions_arr[l2c_test_df_full["forecast_month"] == 1],
                    axis=0,
                )
                final_submission_df.loc[submission_mask, DX_cols] = (
                    global_avg_month1
                )
    return final_submission_df


def _update_submission_regression(
    final_submission_df, predictions_arr, l2c_test_df_full, raw_df, target
):
    """Updates the submission DataFrame for regression target."""
    rids_in_submission = final_submission_df["RID"].unique()
    raw_df["EXAMDATE"] = pd.to_datetime(raw_df["EXAMDATE"])
    assert target in (
        "mmse",
        "vent",
    ), "target must be either 'mmse' or 'vent'!"
    sub_col = "MMSE" if target == "mmse" else "Ventricles_ICV"

    for rid_val in rids_in_submission:
        model_pred_mask = l2c_test_df_full["rid"] == rid_val
        submission_mask = final_submission_df["RID"] == rid_val
        num_to_fill_submission = submission_mask.sum()

        if model_pred_mask.sum() > 0:  # RID has predictions
            assert model_pred_mask.sum() == num_to_fill_submission, (
                f"RID {rid_val} ({target}): l2c_test_df sum {model_pred_mask.sum()} "
                f"!= submission sum {num_to_fill_submission}"
            )

            fallback_reg_val = predictions_arr[model_pred_mask]
            # current_preds_for_rid = predictions_arr[model_pred_mask]
            # final_submission_df.loc[submission_mask, sub_col] = current_preds_for_rid
            # final_submission_df.loc[submission_mask, f"{sub_col} 50% CI lower"] = current_preds_for_rid / 2
            # final_submission_df.loc[submission_mask, f"{sub_col} 50% CI upper"] = current_preds_for_rid * 2
        else:  # RID not in predictions, use raw_df for fallback
            pat_history = raw_df[raw_df["RID"] == rid_val].copy()
            pat_history = pat_history.sort_values(
                by="EXAMDATE", ascending=True, na_position="first"
            )

            if target == "mmse":
                last_val_records = pat_history[pat_history[sub_col].notna()]

                if not last_val_records.empty:
                    fallback_reg_val = last_val_records[sub_col].iloc[-1]
                else:  # No history, try global median from model predictions for forecast_month=1
                    fallback_reg_val = np.nanmedian(
                        predictions_arr[
                            l2c_test_df_full["forecast_month"] == 1
                        ]
                    )
            else:
                last_val_records = pat_history[
                    (pat_history["Ventricles"] / pat_history["ICV"]).notna()
                ]
                if not last_val_records.empty:
                    fallback_reg_val = (
                        last_val_records["Ventricles"].iloc[-1]
                        / last_val_records["ICV"].iloc[-1]
                    )
                else:  # No history, try global median from model predictions for forecast_month=1
                    fallback_reg_val = np.nanmedian(
                        predictions_arr[
                            l2c_test_df_full["forecast_month"] == 1
                        ]
                    )

        final_submission_df.loc[submission_mask, sub_col] = fallback_reg_val
        final_submission_df.loc[submission_mask, f"{sub_col} 50% CI lower"] = (
            fallback_reg_val / 2
        )
        final_submission_df.loc[submission_mask, f"{sub_col} 50% CI upper"] = (
            fallback_reg_val * 2
        )

    return final_submission_df
