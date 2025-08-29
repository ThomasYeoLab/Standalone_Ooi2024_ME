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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import global_config
from models.L2C_FNN.dataset import TestDataset
from models.L2C_FNN.misc import build_pred_frame, inverse_transform
from models.L2C_FNN.model_config import OneHot_Config
from models.L2C_FNN.models import NeuralNetwork
from utils.evalOneSubmission import evalOneSub
from utils.evalOneSubmissionIndiMetric import evalOneSubNew


def predict(model, dataloader, data, device):
    """Make prediction for M months ahead for all N subjects (M*N total predictions)
        Each batch contains # batch_size subject&timepoint samples

    Args:
        model: trained and loaded model
        dataloader: pytorch dataloader containing the input data
        data: raw data dictionary to load "df_dict" used for reversing predictions
        device: device to run the model on (cuda vs cpu)

    Return:
        ret: dictionary saving the predictions for each target variable
    """

    model.eval()
    ret = {}
    val_list, cat_list = [], []

    for idx, inp in enumerate(dataloader):
        inp = inp.to(device)
        cat_out, val_out = model(inp)
        cat_out = nn.functional.softmax(cat_out, dim=-1)
        cat_pred = cat_out.detach().cpu().numpy()
        val_pred = val_out.detach().cpu().numpy()
        val_list.append(val_pred)
        cat_list.append(cat_pred)

    val = np.vstack(val_list)
    cat = np.vstack(cat_list)
    df_dict = data["df_dict"]
    ret["mmse"] = inverse_transform(df_dict["mmse"], val[:, 0])
    ret["vent"] = inverse_transform(df_dict["vent"], val[:, 1])
    ret["clin"] = cat

    return ret


def load_checkpoint(model_dir, args):
    """
    Load best model configurations and initialize model with them
    """
    config_file = path.join(model_dir, "config.json")
    with open(config_file) as fhandler:
        config = json.load(fhandler)
    trial = config["best_trial"]
    layer_num = config["layer_num"]
    hid_dim_list = []
    for l in range(layer_num):
        hid_dim_list.append(config[f"hid_dim_{l}"])
    model_config = OneHot_Config(
        layer_num=layer_num,
        input_dim=args.input_dim,
        hid_dim_list=hid_dim_list,
        p_drop=config["p_drop"] / 10.0,
        relu_slope=config["relu_slope"],
    )
    model = NeuralNetwork(model_config)
    model_checkpoint = path.join(model_dir, f"trial{trial}.pt")
    model.load_state_dict(torch.load(model_checkpoint))
    return model


def main(args):
    """
    Main execution function, loading models, data, and perform prediction & evaluation.
    Then save the evaluation results into log files
    """

    fold = args.fold + args.start_fold

    # Generate save directories
    old_metric_path = path.join(
        global_config.prediction_dir,
        "L2C_FNN",
        args.site,
        "old_metric",
        f"fold_{fold}",
    )
    new_metric_path = path.join(
        global_config.prediction_dir,
        "L2C_FNN",
        args.site,
        "new_metric",
        f"fold_{fold}",
    )
    makedirs(old_metric_path, exist_ok=True)
    makedirs(new_metric_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = path.join(
        global_config.processed_frog, args.site, f"fold_{fold}.pkl"
    )
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    test_data = TestDataset(data["test"])
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    model_dir = path.join(
        global_config.checkpoint_dir,
        "L2C_FNN",
        args.train_site,
        f"model.f{fold}",
    )
    model = load_checkpoint(model_dir, args).to(device)

    if args.in_domain:
        ref_frame = path.join(
            global_config.clean_gt_path, args.site, f"fold{fold}_test.csv"
        )
        sub_frame = path.join(
            global_config.fold_gen_path,
            args.site,
            f"fold{fold}_submission_test.csv",
        )
    else:  # external test, no split
        ref_frame = path.join(
            global_config.clean_gt_path, args.site, "test_gt.csv"
        )
        sub_frame = path.join(
            global_config.fold_gen_path, args.site, "submission_test.csv"
        )

    prediction = predict(model, test_dataloader, data, device)
    csv_path = path.join(old_metric_path, "prediction.csv")
    tadpole = data["test"]["input"]
    build_pred_frame(sub_frame, prediction, tadpole, csv_path)

    old_result = evalOneSub(
        pd.read_csv(ref_frame), pd.read_csv(csv_path), args.site
    )
    new_result, subj_metric, subj_dfs = evalOneSubNew(
        pd.read_csv(ref_frame), pd.read_csv(csv_path), args.site
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
    parser.add_argument("--in_domain", action="store_true")
    parser.add_argument("--input_dim", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--start_fold", type=int, required=True)

    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()

    # set all the seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    main(args)
