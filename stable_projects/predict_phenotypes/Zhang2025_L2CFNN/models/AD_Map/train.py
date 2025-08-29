#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import json
import logging
import random
import sys
import time
import warnings
from os import makedirs, path

import numpy as np
import optuna
import pandas as pd
from leaspy import AlgorithmSettings, Data, Leaspy
from optuna.trial import TrialState

from config import global_config
from models.AD_Map.evaluation import predict
from models.AD_Map.misc import (
    load_adni_data,
    preprocess_data,
    transform_submission,
)
from models.AD_Map.model_config import model_config
from models.model_utils import _calc_score
from utils.evalOneSubmission import evalOneSub

# Ignore all warnings
warnings.filterwarnings("ignore")


def eval1trial(leaspy, df_val, data_val, scores, std_dict, args):
    """
    Evaluate performance on validation set for a trial
    The score will be used to select best trial and prune bad trials
    """
    metric_dict = {}
    submission_df = pd.read_csv(
        args.sub_path, usecols=["RID", "Forecast Month", "Forecast Date"]
    )

    # Transform the submission_df to correct format
    submission_df = transform_submission(submission_df, df_val)

    prediction = predict(leaspy, data_val, submission_df, scores, args)

    result = evalOneSub(
        pd.read_csv(args.ref_frame), prediction, args.site, "AD_Map"
    )

    mmse = result["mmseMAE"] / std_dict["MMSEstd"]
    vent = result["ventsMAE"] / std_dict["VentICVstd"]
    score = _calc_score(result["mAUC"], result["bca"], mmse, vent)

    metric_dict["mAUC"] = result["mAUC"]
    metric_dict["bca"] = result["bca"]
    metric_dict["mmse"] = mmse
    metric_dict["vent"] = vent
    return score, metric_dict


def fit_leaspy(n_iter, nb_source, data_train, args, trial):
    """
    Function for fitting AD-Map model based on Optuna suggested hyperparameters

    Args:
        n_iter (int): multiplier controlling the number of fitting iterations
        nb_source (int): source dimension, number of ICA components used,
        refer to Leaspy tutorial for detailed definiation:
        https://disease-progression-modelling.github.io/pages/main.html
        data_train (dataframe): training data in pandas format
        args: input arguments
        trial: Optuna trial object, provide trial.number attribute

    Return:
        Fitted model
    """

    # Define Leapsy model
    n_iter_base = 10 if args.debug else args.n_iter_base
    leaspy_model = "logistic"  # 'Multivariate logistic'
    algo_settings = AlgorithmSettings(
        "mcmc_saem",
        n_iter=n_iter * n_iter_base,  # n_iter defines the number of iterations
        progress_bar=False,
        seed=args.seed,
    )

    log_path = path.join(args.checkpoint, f"trial{trial.number}_log")
    algo_settings.set_logs(
        path=log_path,
        save_periodicity=1000,  # change accordingly to control verbosity
        console_print_periodicity=None,  # Don't print to console
        overwrite_logs_folder=True,  # Default behaviour raise an error if the folder already exists.
    )

    # FIT
    leaspy = Leaspy(leaspy_model)
    leaspy.model.load_hyperparameters({"source_dimension": nb_source})
    leaspy.fit(data_train, settings=algo_settings)
    return leaspy


class Objective(object):
    """
    Objective class for Optuna tuning, controlling:
    1) data loading and processing
    2) training and validation evaluation
    3) logging and returning validation score to Optuna
    """

    def __init__(self, args):
        start = time.time()
        # Load and generate Leapsy data format (train set)
        mri_features = model_config.mri_features
        use_columns = model_config.use_columns + ["curr_age"]

        df_train = load_adni_data(
            args.site,
            args.fold,
            global_config.fold_gen_path,
            "train",
            use_columns,
            mri_features,
        )

        df_train, scores, std_dict = preprocess_data(
            df_train, mri_features, scores=None
        )

        score_path = path.join(args.checkpoint, "norm_scores.json")
        with open(score_path, "w") as fhandler:
            print(json.dumps(scores), file=fhandler)

        # Load and generate Leapsy data format (validation set)
        df_val = load_adni_data(
            args.site,
            args.fold,
            global_config.fold_gen_path,
            "val",
            use_columns,
            mri_features,
        )

        df_val = preprocess_data(df_val, mri_features, scores)

        self.args = args
        self.scores = scores
        self.df_val = df_val
        self.std_dict = std_dict
        self.data_train = Data.from_dataframe(
            df_train[model_config.feature_columns]
        )
        self.data_val = Data.from_dataframe(
            df_val[model_config.feature_columns]
        )

        end = time.time()
        print(f"time for get dataloader {end - start}")

    def __call__(self, trial):
        start = time.time()
        # Suggest Leapsy hyperparameters
        nb_source = trial.suggest_int("nb_source", 1, 5)
        n_iter = trial.suggest_int("n_iter", 1, 10)

        leaspy = fit_leaspy(n_iter, nb_source, self.data_train, args, trial)

        score, metric = eval1trial(
            leaspy,
            self.df_val,
            self.data_val,
            self.scores,
            self.std_dict,
            self.args,
        )

        model_path = path.join(args.checkpoint, f"trial{trial.number}.json")
        leaspy.save(model_path, indent=2)

        log_name = path.join(args.checkpoint, f"trial{trial.number}_log.json")
        log_dict = {"eval": metric}
        with open(log_name, "w") as fhandler:
            print(json.dumps(log_dict), file=fhandler)

        end = time.time()
        print(f"time for one trial {end - start}")
        return score


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--n_iter_base", type=int, default=500)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode (much faster with fewer iterations)",
    )
    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    fold = args.fold
    print(f"current fold {fold}")
    args.checkpoint = path.join(
        global_config.checkpoint_dir, "AD_Map", args.site, f"model.f{fold}"
    )
    makedirs(args.checkpoint, exist_ok=True)

    args.ref_frame = path.join(
        global_config.clean_gt_path, args.site, f"fold{fold}_val.csv"
    )
    args.sub_path = path.join(
        global_config.fold_gen_path,
        args.site,
        f"fold{fold}_submission_val.csv",
    )

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

    objective = Objective(args)
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
