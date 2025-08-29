#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import json
import logging
import pickle
import random
import sys
import time
import warnings
from os import makedirs, path

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from config import global_config
from models.L2C_XGBnw.train import (
    define_model,
    evaluate_validation_score,
    format_predictions_to_submission,
    _prepare_data_for_xgb,
    predict_with_xgb_models,
    train_xgb_models,
)
from models.L2C_XGBnw.model_config import model_config
from models.L2C_XGBnw.utils import (
    _process_predictions_clin,
    _process_predictions_regression_step1,
    _process_predictions_regression_step2,
)


def process_windowed_xgb_predictions(
    raw_predictions,
    target,
    X_train_features,
    X_test_features,
    train_df_filtered,
):
    """Processes raw predictions, perform missing data filling."""

    final_processed_predictions = None
    if target == "clin":
        final_processed_predictions = _process_predictions_clin(
            raw_predictions.copy(),
            X_train_features,
            X_test_features,
            train_df_filtered,
        )
    elif target in ("mmse", "vent"):
        final_processed_predictions = _process_predictions_regression_step1(
            raw_predictions.copy(), X_test_features, target
        )

    return final_processed_predictions


def _target_specific_window_setup(target, l2c_test_df):
    """Target-specific setup for windowing"""
    overall_predictions = None  # Determined by target

    partition_boundaries = model_config.partition_dict[target]
    if target == "clin":
        time_since_col = "time_since_mr_dx"
        num_output_cols_pred = 3
        overall_predictions = np.full(
            (l2c_test_df.shape[0], num_output_cols_pred), np.nan
        )
    elif target == "mmse":
        time_since_col = "time_since_mr_MMSE"
        overall_predictions = np.full(l2c_test_df.shape[0], np.nan)
    elif target == "vent":
        time_since_col = (
            "time_since_mr_Ventricles"  # Example, ensure this column exists
        )
        overall_predictions = np.full(l2c_test_df.shape[0], np.nan)
    else:
        raise ValueError(
            f"Unsupported target for windowed processing: {target}"
        )
    return time_since_col, overall_predictions, partition_boundaries


def train_eval_xgboost_windowed_merged(
    xgb_hyperparams, inp_data_dict, args, target
):
    """Fit XGBoost model then evaluate on validation set, report score for Optuna search
        Note that we need to loop through window partitions and make predictions within each
    Args:
        param: hyper-parameters returned by "define_model" function
        (some default, some suggested by Optuna)
        inp_data_dict: dictionary containing all required pandas dataframes
            'l2c_train_df': L2C features train dataframe
            'l2c_test_df': L2C features validation dataframe (generated based on input half)
            'raw_df': raw features before L2C transformation
            'submission': dataframe to accomodate validation predictions

    Return:
        score: evaluation metric measuring Optuna trial performance
        model_dict: trained XGBoost models
    """
    l2c_train_df = inp_data_dict["l2c_train_df"]
    l2c_test_df = inp_data_dict["l2c_test_df"]  # This is the validation set

    model_dict_all_partitions = {}
    time_since_col, overall_predictions, partition_boundaries = (
        _target_specific_window_setup(target, l2c_test_df)
    )

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

        if len(l2c_test_partition) == 0:
            continue

        # if l2c_test_partition.empty:
        #     print(f"Info: Partition {jj} ({time_since_col} {lower_bound}-{upper_bound}) "
        #           "has no validation samples. Skipping.")
        #     continue
        # if l2c_train_partition.empty:
        #     print(f"Warning: Partition {jj} ({time_since_col} {lower_bound}-{upper_bound}) "
        #           "has no training samples. Skipping.")
        #     continue

        # 1. Prepare Data for this partition
        X_train_p, y_train_p, X_test_p, train_df_filtered_p = (
            _prepare_data_for_xgb(
                l2c_train_partition, l2c_test_partition, target
            )
        )

        if X_train_p.empty or X_test_p.empty:
            print(
                f"Warning: Partition {jj} resulted in empty features after prep. Skipping."
            )
            continue

        # 2. Train Models for this partition
        trained_model_dict_p = train_xgb_models(
            X_train_p,
            y_train_p,
            X_test_p,
            xgb_hyperparams,
            target,
            args,
            partition_idx_str=partition_id_str,
        )
        model_dict_all_partitions.update(trained_model_dict_p)

        # 3. Predict with Trained Models for this partition
        raw_predictions_p = predict_with_xgb_models(
            trained_model_dict_p,
            X_test_p,
            target,
            partition_idx_str=partition_id_str,
        )

        # 4-1. Process predictions for this partition (clin and regression_step1)
        processed_predictions_p = process_windowed_xgb_predictions(
            raw_predictions_p, target, X_train_p, X_test_p, train_df_filtered_p
        )

        # Store predictions in the overall array (shape must match)
        if target == "clin":
            overall_predictions[val_bin_mask, :] = processed_predictions_p
        else:
            overall_predictions[val_bin_mask] = processed_predictions_p

        # # Store predictions in the overall array (shape must match)
        # if target == "clin":
        #     overall_predictions[val_bin_mask, :] = raw_predictions_p
        # else: # mmse, vent
        #     overall_predictions[val_bin_mask] = raw_predictions_p

    # --- After all partitions, process overall predictions and evaluate ---
    # Note this step is only required for MMSE and Ventricle. As per FROG original implementation,
    # clin was only filled once within the partition loop

    # 4-2. Process overall predictions (2nd pass, for MMSE and Ventricle)
    if target in ("mmse", "vent"):
        overall_predictions = _process_predictions_regression_step2(
            overall_predictions.copy(), l2c_test_df, target
        )

    # 5. Format processed prediction into submission and evaluate
    final_submission = format_predictions_to_submission(
        inp_data_dict, overall_predictions, l2c_test_df, target
    )

    score = evaluate_validation_score(final_submission, args, target)

    return score, model_dict_all_partitions


class Objective(object):
    """
    Objective class for Optuna tuning, controlling:
    1) data loading and processing
    2) training and validation evaluation
    3) logging and returning validation score to Optuna
    """

    def __init__(self, args, inp_data_dict):
        self.args = args
        self.inp_data_dict = inp_data_dict

    def __call__(self, trial):
        # Generate the model.
        param = define_model(trial, self.args.target)
        score, model = train_eval_xgboost_windowed_merged(
            param, self.inp_data_dict, self.args, self.args.target
        )
        model_pt = path.join(self.args.checkpoint, f"trial{trial.number}.pkl")
        pickle.dump(model, open(model_pt, "wb"))
        return score


def get_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--site", type=str, default="ADNI")
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_arg_parser().parse_args()
    fold = args.fold + args.start_fold
    args.checkpoint = path.join(
        global_config.checkpoint_dir,
        "L2C_XGBw",
        args.site,
        args.target,
        f"model.f{fold}",
    )
    args.ref_frame = path.join(
        global_config.clean_gt_path, args.site, f"fold{fold}_val.csv"
    )
    makedirs(args.checkpoint, exist_ok=True)

    inp_data_dict = {
        "l2c_train_df": pd.read_csv(
            path.join(
                global_config.frog_path, args.site, f"fold{fold}_train.csv"
            )
        ),
        "l2c_test_df": pd.read_csv(
            path.join(
                global_config.frog_path, args.site, f"fold{fold}_val.csv"
            )
        ),
        "raw_df": pd.read_csv(
            path.join(
                global_config.fold_gen_path,
                args.site,
                f"fold{fold}_basic.csv",
            )
        ),
        "submission": pd.read_csv(
            path.join(
                global_config.fold_gen_path,
                args.site,
                f"fold{fold}_submission_val.csv",
            )
        ),
    }

    t_overall = time.time()  # seconds

    # set all the seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout)
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    objective = Objective(args, inp_data_dict)
    study.optimize(objective, n_trials=args.trials)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE]
    )

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    config_dict = trial.params
    config_dict["value"] = trial.value
    config_dict["best_trial"] = trial.number
    config_path = path.join(args.checkpoint, "config.json")
    with open(config_path, "w") as fhandler:
        print(json.dumps(config_dict), file=fhandler)
