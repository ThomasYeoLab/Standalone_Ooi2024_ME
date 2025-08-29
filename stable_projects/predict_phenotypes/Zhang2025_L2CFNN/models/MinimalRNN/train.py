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
from os import makedirs, path, remove

import numpy as np
import optuna
import pandas as pd
import torch
import torch.utils.data
from optuna.trial import TrialState
from torch.utils.data import DataLoader

from config import global_config
from models.L2C_FNN.train import unique_name
from models.MinimalRNN.dataset import (
    TestDataset,
    TrainDataset,
    pad_collate_test,
    pad_collate_train,
)
from models.MinimalRNN.evaluation import predict
from models.MinimalRNN.misc import build_pred_frame
from models.MinimalRNN.model import MODEL_DICT
from models.model_utils import _calc_score
from utils.evalOneSubmission import evalOneSub

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def define_model(trial, args):
    """
    Define hyperparameters and the search ranges
    Optimize the nb_layer, nb_head, head_dim, warmup, and dropout.
    """
    h_size = 2 ** trial.suggest_int("h_size", 7, 9)
    nb_layers = trial.suggest_int("nb_layer", 1, 3)
    h_drop = trial.suggest_int("h_drop", 0, 5) / 10.0
    i_drop = trial.suggest_int("i_drop", 0, 5) / 10.0
    c_drop = trial.suggest_int("c_drop", 0, 5) / 10.0
    model_class = MODEL_DICT["MinRNN"]
    model = model_class(
        nb_classes=3,
        nb_measures=args.nb_measures,
        nb_layers=nb_layers,
        h_size=h_size,
        h_drop=h_drop,
        i_drop=i_drop,
        c_drop=c_drop,
    )
    return model


def compute_loss_all_variables(
    cat_out, val_out, cat_true, val_true, cat_mask, val_mask
):
    """
    Loss calculation for all variables used in the model (not just the three task variables)
    """
    cat_mask = cat_mask.unsqueeze(-1)
    c_pred = cat_out.masked_select(cat_mask).view(-1, 3)
    c_true = cat_true.masked_select(cat_mask.squeeze())
    ce_loss = (
        torch.nn.functional.cross_entropy(c_pred, c_true, reduction="sum")
        / c_true.shape[0]
    )
    nb_tps = val_true.shape[0]
    v_true = val_true.masked_fill(~val_mask, 0)
    v_pred = val_out.masked_fill(~val_mask, 0)

    mae_loss = (
        torch.nn.functional.l1_loss(v_pred, v_true, reduction="sum") / nb_tps
    )
    return ce_loss, mae_loss


def train1epoch(model, opt, dataloader, device):
    """
    Train for one epoch, loop through all batches
    """
    model.train()
    ce_loss, mae_loss = 0.0, 0.0
    for _, batch in enumerate(dataloader):
        opt.zero_grad()
        i_cat, i_val, truth, mask = batch
        i_cat = i_cat.detach().cpu().numpy()
        i_val = i_val.detach().cpu().numpy()

        cat_mask = mask[:, :, 0].to(torch.bool).to(device)
        val_mask = mask[:, :, 1:].to(torch.bool).to(device)
        cat_true = truth[:, :, 0].to(torch.long).to(device)
        val_true = truth[:, :, 1:].to(device)

        cat_out, val_out = model(i_cat, i_val)
        ent, mae = compute_loss_all_variables(
            cat_out, val_out, cat_true, val_true, cat_mask, val_mask
        )

        loss = mae + ent

        loss.backward()
        opt.step()

        ce_loss += ent.detach().cpu().item()
        mae_loss += mae.detach().cpu().item()

    return ce_loss / len(dataloader), mae_loss / len(dataloader)


def eval1epoch(model, dataloader, data, args):
    """
    Evaluate model performance after training for one epoch
    Save a temporary prediction csv, compute evaluation metrics then delete it
    """
    metric_dict = {}
    prediction = predict(
        model,
        dataloader,
        data,
        data["pred_start"],
        data["duration"],
        args.site,
    )
    csv_path = unique_name(args)
    build_pred_frame(prediction, csv_path)

    result = evalOneSub(
        pd.read_csv(args.ref_frame), pd.read_csv(csv_path), args.site
    )

    remove(csv_path)
    mmse = result["mmseMAE"] / data["stds"]["MMSE"]
    vent = result["ventsMAE"] / data["VentICVstd"]
    score = _calc_score(result["mAUC"], result["bca"], mmse, vent)

    metric_dict["mAUC"] = result["mAUC"]
    metric_dict["bca"] = result["bca"]
    metric_dict["mmse"] = mmse
    metric_dict["vent"] = vent
    return score, metric_dict


class Objective(object):
    """
    Objective class for Optuna tuning, controlling:
    1) data loading and processing
    2) training and validation evaluation
    3) logging and returning validation score to Optuna
    """

    def __init__(self, args):
        start = time.time()
        # Get the dataset.
        with open(args.data, "rb") as f:
            data = pickle.load(f)

        train_data = TrainDataset(data["train"])
        train_dataloader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=pad_collate_train,
        )
        val_data = TestDataset(data["test"])
        val_dataloader = DataLoader(
            val_data,
            batch_size=len(val_data),
            shuffle=False,
            drop_last=False,
            collate_fn=pad_collate_test,
        )

        self.args = args
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.data = data
        end = time.time()
        print(f"time for get dataloader {end - start}")

    def __call__(self, trial):
        # Generate the model.
        model = define_model(trial, args)

        # Generate the optimizers.
        lr = trial.suggest_int("lr", -5, -2)
        weight_decay = trial.suggest_int("weight_decay", -7, -4)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=10**lr, weight_decay=10**weight_decay
        )

        mean = self.data["mean"]
        stds = self.data["stds"]
        setattr(model, "mean", mean)
        setattr(model, "stds", stds)
        model = model.to(DEVICE)

        prune_flag = False
        train_logs = {"ce_loss": [], "mae_loss": []}
        eval_logs = {"mAUC": [], "bca": [], "mmse": [], "vent": []}

        start = time.time()
        for epoch in range(self.args.epochs):
            # Training of model
            ce, mae = train1epoch(model, optimizer, self.train_loader, DEVICE)
            train_logs["ce_loss"].append(ce)
            train_logs["mae_loss"].append(mae)
            # Validation of the model.
            score, metric = eval1epoch(
                model, self.val_loader, self.data, self.args
            )
            for k, v in metric.items():
                eval_logs[k].append(v)

            trial.report(score, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                prune_flag = True
                raise optuna.exceptions.TrialPruned()
        end = time.time()
        print(f"time for one epoch {end - start}")

        if not prune_flag:
            model_pt = path.join(args.checkpoint, f"trial{trial.number}.pt")
            torch.save(model.state_dict(), model_pt)
            log_name = path.join(
                args.checkpoint, f"trial{trial.number}_log.json"
            )
            log_dict = {"train": train_logs, "eval": eval_logs}
            with open(log_name, "w") as fhandler:
                print(json.dumps(log_dict), file=fhandler)
        return score


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--site", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nb_measures", type=int, default=8)
    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    fold = args.fold + args.start_fold
    print(f"current fold {fold}")
    args.checkpoint = path.join(
        global_config.checkpoint_dir, "MinimalRNN", args.site, f"model.f{fold}"
    )
    makedirs(args.checkpoint, exist_ok=True)

    args.ref_frame = path.join(
        global_config.clean_gt_path, args.site, f"fold{fold}_val.csv"
    )
    args.data = path.join(
        global_config.minrnn_path, args.site, f"val.f{fold}.pkl"
    )

    # set all the seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout)
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    # print(f"Sampler is {study.sampler.__class__.__name__}")
    # print(f"Pruner is {study.pruner.__class__.__name__}")
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
    config_dict["nb_measures"] = args.nb_measures
    config_path = path.join(args.checkpoint, "config.json")
    with open(config_path, "w") as fhandler:
        print(json.dumps(config_dict), file=fhandler)
