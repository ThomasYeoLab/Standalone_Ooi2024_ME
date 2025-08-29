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
import xgboost as xgb
from optuna.trial import TrialState

from config import global_config
from models.L2C_XGBnw.model_config import model_config
from models.L2C_XGBnw.utils import (
    _process_predictions_clin,
    _process_predictions_regression_step1,
    _process_predictions_regression_step2,
    _train_cn_models_mmse,
    _update_submission_clin,
    _update_submission_regression,
)
from utils.evalOneSubmission import evalOneSub


# --- Modular Core Functions ---
def define_model(trial, target):
    """
    Define hyperparameters and the search ranges
    """
    max_depth = trial.suggest_int("max_depth", 3, 8)
    subsample = trial.suggest_int("subsample", 4, 10) / 10.0
    eta = trial.suggest_categorical("eta", [0.01, 0.05, 0.1, 0.2])

    param = {
        "eta": eta,
        "max_depth": max_depth,
        "colsample_bytree": 0.75,
        "subsample": subsample,
    }

    if target == "clin":
        param.update(
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "lambda": 0.01,
                "alpha": 0,
                "eval_metric": "mlogloss",
            }
        )
    elif target in ("mmse", "vent"):
        param.update(
            {
                "objective": "reg:squarederror",
                "lambda": 0.1,
                "alpha": 0.1,
                "eval_metric": "mae",
            }
        )
    else:
        raise ValueError(f"Unknown target: {target}")
    return param


def _prepare_data_for_xgb(l2c_train_df, l2c_test_df, target):
    """Prepares training and test data for XGBoost based on the target."""
    if target == "clin":
        target_col_name = "curr_dx"
        train_df_filtered = l2c_train_df[
            l2c_train_df[target_col_name].notnull()
        ].copy()
        y_train = train_df_filtered[target_col_name]
    elif target == "mmse":
        target_col_name = "curr_mmse"
        train_df_filtered = l2c_train_df[
            l2c_train_df[target_col_name].notnull()
        ].copy()
        y_train = train_df_filtered[target_col_name]
    elif target == "vent":
        target_col_name = "curr_ventricles"  # Numerator
        icv_col_name = "curr_icv"  # Denominator
        train_df_filtered = l2c_train_df[
            l2c_train_df[target_col_name].notnull()
        ].copy()
        y_train = (
            train_df_filtered[target_col_name]
            / train_df_filtered[icv_col_name]
        )
    else:
        raise ValueError(f"Unsupported target: {target}")

    X_train_features = train_df_filtered[
        train_df_filtered.columns.difference(model_config.not_predictors)
    ].copy()
    X_test_features = l2c_test_df[
        l2c_test_df.columns.difference(model_config.not_predictors_test)
    ].copy()

    # Assertion to prevebt potential differences (e.g. if one set had a rare feature)
    assert list(X_train_features.columns) == list(
        X_test_features.columns
    ), "Train and test feature columns do not match!"

    # Store original filtered train DF for context (e.g., for _process_predictions_clin)
    # and original test DF for context (e.g., for _process_predictions_mmse)
    return X_train_features, y_train, X_test_features, train_df_filtered


def train_xgb_models(
    X_train_features,
    y_train,
    x_test_features,
    xgb_params,
    target,
    args_obj,
    partition_idx_str="",
):
    """Trains main XGBoost models and any target-specific sub-models (e.g., MMSE CN)."""
    model_dict = {}
    num_ensemble_models = model_config.num_ensemble_models
    num_boost_round_main = (
        10 if args_obj.debug else model_config.num_boost_round_dict[target]
    )

    dtrain = xgb.DMatrix(X_train_features, label=y_train, missing=np.nan)
    watchlist = [(dtrain, "train")]

    for i_model in range(num_ensemble_models):
        current_xgb_params = xgb_params.copy()
        current_xgb_params["seed"] = (
            i_model  # Use loop index for seed if ensembling
        )

        gbm = xgb.train(
            current_xgb_params,
            dtrain,
            num_boost_round=num_boost_round_main,
            evals=watchlist,
            verbose_eval=False,
        )
        model_key = f"main_model{partition_idx_str}_seed_{i_model}"
        model_dict[model_key] = gbm

    if target == "mmse":
        # Pass relevant parts of model_config if _train_cn_models_mmse needs them
        cn_models = _train_cn_models_mmse(
            xgb_params,
            X_train_features,
            y_train,
            x_test_features,
            num_ensemble_models,
            args_obj,
            partition_idx_str,
        )
        model_dict.update(cn_models)

    return model_dict


def predict_with_xgb_models(
    model_dict, X_test_features, target, partition_idx_str=""
):
    """Generates predictions using trained XGBoost models (main and CN if applicable)."""
    num_ensemble_models = model_config.num_ensemble_models
    is_multiclass = (
        target == "clin"
    )  # Only clinical diagnosis is classification
    num_output_cols = 3 if is_multiclass else 1

    if is_multiclass:
        yhat_ensemble_output = np.full(
            (X_test_features.shape[0], num_output_cols, num_ensemble_models),
            np.nan,
        )
    else:
        yhat_ensemble_output = np.full(
            (X_test_features.shape[0], num_ensemble_models), np.nan
        )

    dtest_eval = xgb.DMatrix(X_test_features)
    for i_model in range(num_ensemble_models):
        main_model_key = f"main_model{partition_idx_str}_seed_{i_model}"
        if main_model_key not in model_dict:
            print(
                f"Warning (predict_with_xgb_models): Main model {main_model_key} not found. Skipping its prediction."
            )
            continue

        gbm = model_dict[main_model_key]
        model_predictions = gbm.predict(dtest_eval)

        if is_multiclass:
            yhat_ensemble_output[:, :, i_model] = model_predictions
        else:
            yhat_ensemble_output[:, i_model] = model_predictions

    # Aggregate predictions from ensemble of main models
    agg_axis = 2 if is_multiclass else 1
    aggregated_main_predictions = np.nanmean(
        yhat_ensemble_output, axis=agg_axis
    )

    if target == "mmse":
        # Apply CN models if they exist and conditions are met
        # (i.e., there are CN test samples and model is available)
        cn_test_mask = X_test_features["mr_dx"] == 0
        cn_models_available = [
            f"CN_partition{partition_idx_str}_seed_{i}"
            for i in range(num_ensemble_models)
            if f"CN_partition{partition_idx_str}_seed_{i}" in model_dict.keys()
        ]
        if len(cn_test_mask) > 0 and cn_models_available:
            dtest_cn_eval = xgb.DMatrix(X_test_features[cn_test_mask])
            cn_predictions_ensemble_eval = np.full(
                (cn_test_mask.sum(), len(cn_models_available)), np.nan
            )

            for i, model_key in enumerate(cn_models_available):
                gbm_cn = model_dict[model_key]
                cn_predictions_ensemble_eval[:, i] = gbm_cn.predict(
                    dtest_cn_eval
                )

            aggregated_cn_preds_eval = np.nanmean(
                cn_predictions_ensemble_eval, axis=1
            )
            aggregated_main_predictions[cn_test_mask] = (
                aggregated_cn_preds_eval
            )

    return aggregated_main_predictions


def process_xgb_predictions(
    raw_predictions,
    target,
    X_train_features,
    X_test_features,
    train_df_filtered,
    l2c_test_df,
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
        inter_processed_predictions = _process_predictions_regression_step1(
            raw_predictions.copy(), X_test_features, target
        )
        final_processed_predictions = _process_predictions_regression_step2(
            inter_processed_predictions.copy(), l2c_test_df, target
        )

    return final_processed_predictions


def format_predictions_to_submission(
    inp_data_dict, final_processed_predictions, l2c_test_df, target
):
    """Format processed predictions to fill submission_df"""
    submission_df_template = inp_data_dict["submission"].copy(deep=True)
    raw_df_full = inp_data_dict["raw_df"]
    final_submission_output = None

    if target == "clin":
        final_submission_output = _update_submission_clin(
            submission_df_template,
            final_processed_predictions,
            l2c_test_df,
            raw_df_full,
        )
    elif target in ("mmse", "vent"):
        final_submission_output = _update_submission_regression(
            submission_df_template,
            final_processed_predictions,
            l2c_test_df,
            raw_df_full,
            target,
        )

    return final_submission_output


def evaluate_validation_score(final_submission_output, args_obj, target):
    """Evaluate the validation submission, return score for Optuna"""
    score = float("inf")
    try:
        true_data_for_eval = pd.read_csv(args_obj.ref_frame)
        eval_results = evalOneSub(
            true_data_for_eval, final_submission_output, args_obj.site
        )  # Ensure evalOneSub can determine target or pass it

        if target == "clin":
            score = -eval_results.get("mAUC") - eval_results.get("bca")
        elif target == "mmse":
            score = eval_results.get("mmseMAE")
        elif target == "vent":
            score = eval_results.get("ventsMAE")
    except FileNotFoundError:
        print(f"Error: Eval ref file not found: {args_obj.ref_frame}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

    return score


# --- Refactored Orchestrator for Non-Windowed Training/Validation ---
def train_eval_xgboost_refactored(
    xgb_hyperparams, inp_data_dict, args_obj, target
):
    """
    Fit XGBoost model then evaluate on validation set, report score for Optuna search.
    Merged function for 'clin' (clinical diagnosis), 'mmse' (MMSE score) and
    'vent' (Ventricle volume normalized by ICV) targets.

    Args:
        param: hyper-parameters returned by "define_model" function
        (some default, some suggested by Optuna)
        inp_data_dict: dictionary containing all required pandas dataframes
            'l2c_train_df': L2C features train dataframe
            'l2c_test_df': L2C features validation dataframe (generated based on input half)
            'raw_df': raw features before L2C transformation
            'submission': dataframe to accomodate validation predictions
        args_obj: additional input arguments from bash
        target: target task variable to predict

    Return:
        score: evaluation metric measuring Optuna trial performance
        model_dict: trained XGBoost models
    """
    l2c_train_df = inp_data_dict["l2c_train_df"]
    l2c_test_df = inp_data_dict["l2c_test_df"]  # This is the validation set

    # 1. Prepare Data
    X_train, y_train, X_test, train_df_filtered = _prepare_data_for_xgb(
        l2c_train_df, l2c_test_df, target
    )

    if X_train.empty or X_test.empty:
        print(
            f"Warning (train_eval_xgboost_refactored for {target}): "
            "Empty training or test features after preparation. Aborting."
        )
        return float("inf"), {}

    # 2. Train Models
    trained_model_dict = train_xgb_models(
        X_train, y_train, X_test, xgb_hyperparams, target, args_obj
    )

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

    # 5. Evaluate Predictions and Return Score
    score = evaluate_validation_score(final_submission, args_obj, target)

    return score, trained_model_dict


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
        score, model = train_eval_xgboost_refactored(
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
        "L2C_XGBnw",
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
