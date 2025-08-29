#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import json
import pickle
import random
from os import makedirs, path

import numpy as np
import pandas as pd
from config import global_config
from leaspy import AlgorithmSettings, Data, Leaspy

from models.AD_Map.misc import (
    get_probabilities,
    load_adni_data,
    load_external_data,
    preprocess_data,
    reverse_normalize,
    transform_submission,
)
from models.AD_Map.model_config import model_config
from utils.evalOneSubmission import evalOneSub
from utils.evalOneSubmissionIndiMetric import evalOneSubNew


def predict(leaspy, data_test, submission_df, scores, args):
    """Make prediction for M months ahead for all N subjects (M*N total predictions)

    Args:
        leaspy: trained and loaded AD-Map model
        data_test: processed input test data
        submission_df: structured dataframe for storing predictions (M*N rows)
        scores: normalization statistics for pre-processing, saved from training data
        args: various input arguments, here mainly for seed

    Return:
        Filled submission_df
    """
    settings_personalization = AlgorithmSettings(
        "scipy_minimize", progress_bar=False, use_jacobian=True, seed=args.seed
    )
    # Adjust individual curves based on input half
    ip = leaspy.personalize(data_test, settings_personalization)

    # Estimate output half
    submission_multi = submission_df.set_index(["ID", "TIME"]).copy()
    reconstruction = leaspy.estimate(submission_multi.index, ip)

    # Cast back to original scale and save to dataframe
    predictions = reconstruction.reset_index().copy()

    # Apply reverse normalization to each feature in the DataFrame
    for feature, (max_, min_, increase) in scores.items():
        if feature in ["CDR", "MMSE", "Ventricles"]:
            predictions[feature] = predictions[feature].apply(
                reverse_normalize, max_=max_, min_=min_, increase=increase
            )

    # Merge frame1 and frame2 on 'ID' and 'TIME' columns
    combined_frame = pd.merge(
        submission_df,
        predictions[["ID", "TIME", "CDR", "MMSE", "Ventricles"]],
        on=["ID", "TIME"],
        how="inner",
    )
    combined_frame.rename(
        columns={"ID": "RID", "Ventricles": "Ventricles_ICV"}, inplace=True
    )

    # Applying the CDR->prob function to each row in the DataFrame
    combined_frame[
        [
            "CN relative probability",
            "MCI relative probability",
            "AD relative probability",
        ]
    ] = combined_frame["CDR"].apply(
        lambda cdr: pd.Series(get_probabilities(cdr))
    )

    # Add extra columns to submission
    for col in model_config.extra_sub_dx:
        combined_frame[col] = 0  # fill arbitrary values so no error pops up

    return combined_frame


def load_checkpoint(model_dir):
    """Load the AD-Map model with the best hyperparameter configurations
        (based on Optuna validation performance)

    Args:
        model_dir: checkpoint directory for saved models and configurations

    Return:
        Trained AD-Map model
    """
    config_file = path.join(model_dir, "config.json")
    with open(config_file) as fhandler:
        config = json.load(fhandler)
    trial = config["best_trial"]

    model_path = path.join(model_dir, f"trial{trial}.json")
    leaspy = Leaspy.load(model_path)
    return leaspy


def main(args):
    """
    Main execution function, loading models, data, and perform prediction & evaluation.
    Then save the evaluation results into log files
    """
    fold = args.fold

    # Generate save directories
    old_metric_path = path.join(
        global_config.prediction_dir,
        "AD_Map",
        args.site,
        "old_metric",
        f"fold_{fold}",
    )
    new_metric_path = path.join(
        global_config.prediction_dir,
        "AD_Map",
        args.site,
        "new_metric",
        f"fold_{fold}",
    )
    makedirs(old_metric_path, exist_ok=True)
    makedirs(new_metric_path, exist_ok=True)

    mri_features = model_config.mri_features

    # Load model
    model_dir = path.join(
        global_config.checkpoint_dir,
        "AD_Map",
        args.train_site,
        f"model.f{fold}",
    )
    leaspy = load_checkpoint(model_dir)

    score_path = path.join(model_dir, "norm_scores.json")
    with open(score_path, "r") as json_file:
        scores = json.load(json_file)

    if args.in_domain:
        use_columns = model_config.use_columns + ["curr_age"]
        # Load and generate Leapsy data format
        df = load_adni_data(
            args.site,
            fold,
            global_config.fold_gen_path,
            "test",
            use_columns,
            mri_features,
        )
        # Load reference frame and submission_df
        ref_frame = path.join(
            global_config.clean_gt_path, args.site, f"fold{fold}_test.csv"
        )
        submission_test = pd.read_csv(
            path.join(
                global_config.fold_gen_path,
                args.site,
                f"fold{fold}_submission_test.csv",
            ),
            usecols=["RID", "Forecast Month", "Forecast Date"],
        )
    else:
        use_columns = model_config.use_columns + ["AGE"]
        # Load and generate Leapsy data format
        df = load_external_data(
            args.site, global_config.fold_gen_path, use_columns, mri_features
        )
        # Load reference frame and submission_df
        ref_frame = path.join(
            global_config.clean_gt_path, args.site, "test_gt.csv"
        )
        submission_test = pd.read_csv(
            path.join(
                global_config.fold_gen_path, args.site, "submission_test.csv"
            ),
            usecols=["RID", "Forecast Month", "Forecast Date"],
        )

    df_clipped = preprocess_data(df, mri_features, scores)
    data_test = Data.from_dataframe(df_clipped[model_config.feature_columns])

    # Transform the submission_df to correct format
    submission_df = transform_submission(submission_test, df_clipped)

    prediction = predict(leaspy, data_test, submission_df, scores, args)

    csv_path = path.join(old_metric_path, "prediction.csv")
    prediction.to_csv(csv_path, index=False)

    old_result = evalOneSub(
        pd.read_csv(ref_frame), prediction, args.site, "AD_Map"
    )
    new_result, subj_metric, subj_dfs = evalOneSubNew(
        pd.read_csv(ref_frame), prediction, args.site, "AD_Map"
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
    parser.add_argument("--site", required=True)
    parser.add_argument("--train_site", default="ADNI")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--in_domain", action="store_true")
    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()

    # Set all seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    main(args)
