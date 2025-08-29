#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import warnings


from models.L2C_XGBnw.train import (
    format_predictions_to_submission,
    _prepare_data_for_xgb,
    predict_with_xgb_models,
)
from models.L2C_XGBnw.eval import (
    eval_main,
    load_xgboost_models,
)
from models.L2C_XGBw.train import (
    _target_specific_window_setup,
)
from models.L2C_XGBnw.utils import (
    _process_predictions_regression_step1,
    _process_predictions_regression_step2,
)
from models.L2C_XGBw.utils import (
    _process_predictions_clin_step1,
    _process_predictions_clin_step2,
)


def process_windowed_xgb_predictions(
    raw_predictions,
    target,
    X_train_features,
    X_test_features,
    train_df_filtered,
    l2c_train_df_filtered,
):
    """Processes raw predictions, perform missing data filling."""

    final_processed_predictions = None
    if target == "clin":
        final_processed_predictions = _process_predictions_clin_step1(
            raw_predictions.copy(),
            X_train_features,
            X_test_features,
            train_df_filtered,
            l2c_train_df_filtered,
        )
    elif target in ("mmse", "vent"):
        final_processed_predictions = _process_predictions_regression_step1(
            raw_predictions.copy(), X_test_features, target
        )

    return final_processed_predictions


# --- Refactored Orchestrator for Non-Windowed Evaluation ---
def eval_xgboost_windowed_refactored(inp_data_dict, args, target):
    """
    Make predictions using a pre-trained XGBoost model.
    Merged function for 'clin', 'mmse', and 'vent', targets.

    Args:
        inp_data_dict: dictionary containing all required pandas dataframes
            'l2c_train_df': L2C features train dataframe
            'l2c_test_df': L2C features test dataframe (generated based on input half)
            'raw_df': raw features before L2C transformation
            'submission': dataframe to accomodate validation predictions
        args: Object with attributes like 'train_site', 'fold'.
        target: String, either "clin", "mmse", or "vent.

    Return:
        final_submission: Submission dataframe with predictions filled in.
    """
    l2c_train_df = inp_data_dict["l2c_train_df"]
    l2c_train_df_filtered = l2c_train_df[l2c_train_df["curr_dx"].notnull()]
    l2c_test_df = inp_data_dict["l2c_test_df"]

    time_since_col, overall_predictions, partition_boundaries = (
        _target_specific_window_setup(target, l2c_test_df)
    )

    # Load pretrained model
    trained_model_dict = load_xgboost_models(args.model_name, target, args)

    # --- Loop through window partitions ---
    for jj in range(len(partition_boundaries) - 1):
        lower_bound = partition_boundaries[jj]
        upper_bound = partition_boundaries[jj + 1]
        partition_id_str = f"_partition_{jj}"

        if (
            time_since_col not in l2c_train_df.columns
            or time_since_col not in l2c_test_df.columns
        ):
            print(
                f"Warning: Partition column '{time_since_col}' not found. Skipping partition {jj}."
            )
            continue

        tr_bin_mask = (l2c_train_df[time_since_col] > lower_bound) & (
            l2c_train_df[time_since_col] <= upper_bound
        )
        val_bin_mask = (l2c_test_df[time_since_col] > lower_bound) & (
            l2c_test_df[time_since_col] <= upper_bound
        )

        l2c_train_partition = l2c_train_df[tr_bin_mask]
        l2c_test_partition = l2c_test_df[val_bin_mask]

        # if l2c_test_partition.empty:
        #     print(f"Info: Partition {jj} ({time_since_col} {lower_bound}-{upper_bound}) "
        #           "has no validation samples. Skipping.")
        #     continue
        # if l2c_train_partition.empty:
        #     print(f"Warning: Partition {jj} ({time_since_col} {lower_bound}-{upper_bound}) "
        #           "has no training samples. Skipping.")
        #     continue

        if len(l2c_test_partition) == 0:
            continue

        # 1. Prepare Data for this partition
        X_train_p, _, X_test_p, train_df_filtered_p = _prepare_data_for_xgb(
            l2c_train_partition, l2c_test_partition, target
        )

        if X_train_p.empty or X_test_p.empty:
            print(
                f"Warning: Partition {jj} resulted in empty features after prep. Skipping."
            )
            continue

        # 2. Predict with Trained Models for this partition
        raw_predictions_p = predict_with_xgb_models(
            trained_model_dict,
            X_test_p,
            target,
            partition_idx_str=partition_id_str,
        )

        # 3-1. Process predictions for this partition (clin and regression_step1)
        processed_predictions_p = process_windowed_xgb_predictions(
            raw_predictions_p,
            target,
            X_train_p,
            X_test_p,
            train_df_filtered_p,
            l2c_train_df_filtered,
        )

        # Store predictions in the overall array (shape must match)
        if target == "clin":
            overall_predictions[val_bin_mask, :] = processed_predictions_p
        else:
            overall_predictions[val_bin_mask] = processed_predictions_p

    # --- After all partitions, process overall predictions and evaluate ---
    # Note this step is only required for MMSE and Ventricle. As per FROG original implementation,
    # clin was only filled once within the partition loop

    # 3-2. Process overall predictions (2nd pass, for MMSE and Ventricle)
    if target == "clin":
        overall_predictions = _process_predictions_clin_step2(
            overall_predictions.copy(), l2c_train_df_filtered, l2c_test_df
        )
    elif target in ("mmse", "vent"):
        overall_predictions = _process_predictions_regression_step2(
            overall_predictions.copy(), l2c_test_df, target
        )

    # 4. Format processed prediction into submission and evaluate
    final_submission = format_predictions_to_submission(
        inp_data_dict, overall_predictions, l2c_test_df, target
    )

    return final_submission


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--train_site", default="ADNI")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in_domain", action="store_true")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="L2C_XGBw")
    return parser


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_arg_parser().parse_args()
    eval_main(args, eval_xgboost_windowed_refactored)
