#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import json
import pickle
import warnings
from os import makedirs, path

import numpy as np
import pandas as pd

from config import global_config
from models.L2C_XGBnw.train import (
    format_predictions_to_submission,
    _prepare_data_for_xgb,
    predict_with_xgb_models,
    process_xgb_predictions,
)
from utils.evalOneSubmission import evalOneSub
from utils.evalOneSubmissionIndiMetric import evalOneSubNew


def load_xgboost_models(model_name, target, args):
    # --- Load Model ---
    # Construct paths (ensure global_config.checkpoint_dir is correct)
    base_model_path_dir = path.join(
        global_config.checkpoint_dir,
        model_name,
        args.train_site,
        target,
        f"model.f{args.fold}",
    )
    config_file_path = path.join(base_model_path_dir, "config.json")

    with open(config_file_path) as fhandler:
        config_loaded = json.load(fhandler)
    best_trial_num = config_loaded["best_trial"]
    model_file_path = path.join(
        base_model_path_dir, f"trial{best_trial_num}.pkl"
    )
    with open(model_file_path, "rb") as f_model:
        model_dict_loaded = pickle.load(f_model)
    return model_dict_loaded


# --- Refactored Orchestrator for Non-Windowed Evaluation ---
def eval_xgboost_refactored(inp_data_dict, args, target):
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
    l2c_test_df = inp_data_dict["l2c_test_df"]  # This is the validation set

    # 1. Prepare Data
    X_train, _, X_test, train_df_filtered = _prepare_data_for_xgb(
        l2c_train_df, l2c_test_df, target
    )

    # 2. Load Pretrained Models
    # Construct paths (ensure global_config.checkpoint_dir is correct)
    trained_model_dict = load_xgboost_models(args.model_name, target, args)

    # 3. Predict with Trained Models
    raw_predictions = predict_with_xgb_models(
        trained_model_dict, X_test, target
    )

    # 4. Process Predictions into submission
    # Pass original X_train, y_train, X_test, etc. for context if needed by processing functions
    final_processed_predictions = process_xgb_predictions(
        raw_predictions,
        target,
        X_train,
        X_test,
        train_df_filtered,
        l2c_test_df,
    )

    final_submission = format_predictions_to_submission(
        inp_data_dict, final_processed_predictions, l2c_test_df, target
    )

    return final_submission


def eval_main(args, eval_function):
    # set all the seed
    seed = args.seed
    np.random.seed(seed)

    # Generate the save directory
    old_metric_path = path.join(
        global_config.prediction_dir,
        args.model_name,
        args.site,
        "old_metric",
        f"fold_{args.fold}",
    )

    new_metric_path = path.join(
        global_config.prediction_dir,
        args.model_name,
        args.site,
        "new_metric",
        f"fold_{args.fold}",
    )

    makedirs(old_metric_path, exist_ok=True)
    makedirs(new_metric_path, exist_ok=True)

    if args.in_domain:
        inp_data_dict = {
            "l2c_train_df": pd.read_csv(
                path.join(
                    global_config.frog_path,
                    args.train_site,
                    f"fold{args.fold}_train.csv",
                )
            ),  # use  ADNI as training data
            "l2c_test_df": pd.read_csv(
                path.join(
                    global_config.frog_path,
                    args.site,
                    f"fold{args.fold}_test.csv",
                )
            ),
            "raw_df": pd.read_csv(
                path.join(
                    global_config.fold_gen_path,
                    args.site,
                    f"fold{args.fold}_basic.csv",
                )
            ),
        }
        if args.target == "mmse":
            inp_data_dict["submission"] = pd.read_csv(
                path.join(
                    global_config.fold_gen_path,
                    args.site,
                    f"fold{args.fold}_submission_test.csv",
                )
            )
        elif args.target in ("clin", "vent"):
            inp_data_dict["submission"] = pd.read_csv(
                path.join(old_metric_path, "prediction.csv")
            )
        else:
            raise ValueError(f"Unknown target: {args.target}")

        ref_frame = path.join(
            global_config.clean_gt_path,
            args.site,
            f"fold{args.fold}_test.csv",
        )
    else:
        # for external test data, same L2C features, raw input, and submissio files for all folds
        inp_data_dict = {
            "l2c_train_df": pd.read_csv(
                path.join(
                    global_config.frog_path,
                    args.train_site,
                    f"fold{args.fold}_train.csv",
                )
            ),  # use  ADNI as training data
            "l2c_test_df": pd.read_csv(
                path.join(global_config.frog_path, args.site, "test.csv")
            ),
            "raw_df": pd.read_csv(
                path.join(
                    global_config.fold_gen_path, args.site, "basic_test.csv"
                )
            ),
        }
        if args.target == "mmse":
            inp_data_dict["submission"] = pd.read_csv(
                path.join(
                    global_config.fold_gen_path,
                    args.site,
                    "submission_test.csv",
                )
            )
        elif args.target in ("clin", "vent"):
            inp_data_dict["submission"] = pd.read_csv(
                path.join(old_metric_path, "prediction.csv")
            )
        else:
            raise ValueError(f"Unknown target: {args.target}")

        ref_frame = path.join(
            global_config.clean_gt_path, args.site, "test_gt.csv"
        )

    prediction = eval_function(inp_data_dict, args, args.target)

    csv_path = path.join(old_metric_path, "prediction.csv")
    prediction.to_csv(csv_path, index=False)

    # The last target, after "vent" we would have the full prediction_df
    if args.target == "vent":
        old_result = evalOneSub(pd.read_csv(ref_frame), prediction, args.site)
        new_result, subj_metric, subj_dfs = evalOneSubNew(
            pd.read_csv(ref_frame), prediction, args.site
        )

        old_log_name = path.join(old_metric_path, "out.json")
        with open(old_log_name, "w") as f:
            json.dump(old_result, f)
        new_log_name = path.join(new_metric_path, "out.json")
        with open(new_log_name, "w") as f:
            json.dump(new_result, f)
        subj_metric_name = path.join(new_metric_path, "subj_metric.pkl")
        with open(subj_metric_name, "wb") as fhandler:
            pickle.dump(subj_metric, fhandler)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--train_site", default="ADNI")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in_domain", action="store_true")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="L2C_XGBnw")
    return parser


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_arg_parser().parse_args()
    eval_main(args, eval_xgboost_refactored)
