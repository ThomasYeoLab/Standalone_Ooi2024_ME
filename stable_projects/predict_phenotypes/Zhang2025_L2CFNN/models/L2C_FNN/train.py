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
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
import torch
import torch.utils.data
from optuna.trial import TrialState
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from config import global_config
from models.L2C_FNN.dataset import TestDataset, TrainDataset
from models.L2C_FNN.evaluation import predict
from models.L2C_FNN.misc import build_pred_frame
from models.L2C_FNN.model_config import OneHot_Config
from models.L2C_FNN.models import NeuralNetwork
from models.model_utils import _calc_score
from utils.evalOneSubmission import evalOneSub


def define_model(trial):
    """
    Define hyperparameters and the search ranges
    """
    layer_num = trial.suggest_int("layer_num", 2, 5)
    hid_dim_list = []
    for l in range(layer_num):
        hid_dim_list.append(
            trial.suggest_categorical(f"hid_dim_{l}", [128, 256, 384, 512])
        )
    p_drop = trial.suggest_int("p_drop", 0, 5) / 10.0
    relu_slope = trial.suggest_categorical("relu_slope", [0.01, 0.05, 0.1])
    config = OneHot_Config(
        layer_num=layer_num,
        input_dim=args.input_dim,
        nb_classes=args.nb_classes,
        nb_measures=args.nb_measures,
        hid_dim_list=hid_dim_list,
        p_drop=p_drop,
        relu_slope=relu_slope,
    )
    model = NeuralNetwork(config)
    return model


def compute_loss(mask_i, label_i, cat_i, val_i):
    """
    Loss calculation for all target values (both categorical and continuous)
    """

    # compute cross_entropy loss for categorical value using PyTorch
    # CrossEntropy implementation, assuming we don't perform SoftMax in advance
    cat_mask = mask_i[:, 0]
    cat_true = label_i[:, 0].to(torch.long)
    cat_index = cat_true.masked_select(cat_mask)
    cat_pred = cat_i.masked_select(cat_mask.unsqueeze(-1)).view(-1, 3)
    ce_loss = (
        torch.nn.functional.cross_entropy(cat_pred, cat_index, reduction="sum")
        / cat_index.shape[0]
    )

    # compute mae loss for continuous targets
    val_pred = val_i
    val_true = label_i[:, 1:]
    val_mask = mask_i[:, 1:]
    v_true = val_true.masked_fill(~val_mask, 0)
    v_pred = val_pred.masked_fill(~val_mask, 0)
    mae_loss = torch.nn.functional.l1_loss(v_pred, v_true)
    return ce_loss, mae_loss


def unique_name(args):
    """
    Unique file name for prediction results, avoid naming conflict
    during parallel computing
    """
    timestamp = time.strftime("%m%d-%H%M%S", time.localtime())
    filename = f"{timestamp}-{uuid4()}.prediction.csv"
    checkpoint = path.join(args.checkpoint, filename)
    return checkpoint


def parse_batch_train(batch, device):
    """
    Parse one batch into proper input and send to GPU
    """
    inp = batch[0].to(device)
    label = batch[1].to(device)
    mask = batch[2].to(device)
    return inp, label, mask


def train1epoch(model, optimizer, train_loader, device):
    """
    Train for one epoch, loop through all batches
    """
    model.train()
    ce_loss, mae_loss = 0.0, 0.0
    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()
        parsed_data = parse_batch_train(batch, device)
        inp, label, mask = parsed_data
        cat_out, val_out = model(inp)
        ent, mae = compute_loss(mask, label, cat_out, val_out)
        loss = mae + ent

        # one graph is created here
        loss.backward()
        optimizer.step()
        ce_loss += ent.detach().cpu().item()
        mae_loss += mae.detach().cpu().item()

    return ce_loss / len(train_loader), mae_loss / len(train_loader)


def eval1epoch(model, dataloader, data, args, device):
    """
    Evaluate model performance after training for one epoch
    Save a temporary prediction csv, compute evaluation metrics then delete it
    """
    metric_dict = {}
    prediction = predict(model, dataloader, data, device)
    csv_path = unique_name(args)
    raw_input_matrix = data["val"][
        "input"
    ]  # input matrix (numpy array), last column contains RID
    build_pred_frame(
        args.sub_path, prediction, raw_input_matrix, csv_path
    )  # sub_path: submission csv path

    result = evalOneSub(
        pd.read_csv(args.ref_frame), pd.read_csv(csv_path), args.site
    )

    remove(csv_path)
    mmse = result["mmseMAE"] / data["MMSEstd"]
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
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_data = TestDataset(data["val"])
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data = data
        end = time.time()
        print(f"time for get dataloader {end - start}")

    def __call__(self, trial):
        # Generate the model.
        model = define_model(trial)
        model = model.to(DEVICE)

        # Generate the optimizers.
        lr = trial.suggest_int("lr", -5, -1)
        weight_decay = trial.suggest_int("weight_decay", -7, -4)
        momentum = trial.suggest_categorical(f"momentum", [0, 0.5, 0.9])
        gamma = trial.suggest_categorical(f"gamma", [0.1, 0.5, 0.9])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=10**lr,
            weight_decay=10**weight_decay,
            momentum=momentum,
        )

        # Initialize the scheduler
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        prune_flag = False  # default to False at the start of training
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
                model, self.val_loader, self.data, self.args, DEVICE
            )
            for k, v in metric.items():
                eval_logs[k].append(v)

            scheduler.step()
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

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=101)
    parser.add_argument("--nb_classes", type=int, default=3)
    parser.add_argument("--nb_measures", type=int, default=2)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: e.g., cpu, cuda, cuda:0, mps",
    )
    return parser


if __name__ == "__main__":
    """Main execution function."""
    args = get_arg_parser().parse_args()

    # Set GPU to use
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Global access, set a global variable
    global DEVICE
    DEVICE = device

    fold = args.fold + args.start_fold
    print(f"current fold {fold}")

    # Generate the save directory
    args.checkpoint = path.join(
        global_config.checkpoint_dir, f"L2C_FNN/{args.site}/model.f{fold}"
    )
    makedirs(args.checkpoint, exist_ok=True)

    args.ref_frame = path.join(
        global_config.clean_gt_path, f"{args.site}/fold{fold}_val.csv"
    )
    args.sub_path = path.join(
        global_config.fold_gen_path,
        f"{args.site}/fold{fold}_submission_val.csv",
    )
    args.data = path.join(
        global_config.processed_frog, f"{args.site}/fold_{fold}.pkl"
    )

    t_overall = time.time()  # seconds

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
    config_path = path.join(args.checkpoint, "config.json")
    with open(config_path, "w") as fhandler:
        print(json.dumps(config_dict), file=fhandler)
