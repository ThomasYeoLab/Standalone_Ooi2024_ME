#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import json
import pickle
import random
from os import makedirs, path

import models.MinimalRNN.misc as misc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import global_config
from models.MinimalRNN.dataset import TestDataset, pad_collate_test
from models.MinimalRNN.model import MODEL_DICT
from utils.evalOneSubmission import evalOneSub
from utils.evalOneSubmissionIndiMetric import evalOneSubNew


def predict_subject(
    model, cat_seq, val_seq, duration, len_list, device=None, args=None
):
    """
    First pad the actual input data with NaN (for future visits) then predict the future visits
    recursively using the trained MinimalRNN model.
    Main difference from Minh's implementation is we allow batch computation,
    whereas Minh's original implementation can only predict 1 subject a time
    """
    in_val = np.full(
        (duration + len(val_seq),) + val_seq.shape[1:],
        np.nan,
        dtype=np.float32,
    )
    in_cat = np.full(
        (duration + len(cat_seq),) + cat_seq.shape[1:],
        np.nan,
        dtype=np.float32,
    )
    for idx, l in enumerate(len_list):
        in_val[:l, idx] = val_seq[:l, idx]
        in_cat[:l, idx] = cat_seq[:l, idx]

    # if use constant features, input const here
    with torch.no_grad():
        cat_out, val_out = model(in_cat, in_val)
    cat_out = cat_out.detach().cpu()
    val_out = val_out.detach().cpu()

    return cat_out, val_out


def predict(model, dataloader, data, pred_start, duration, site):
    """
    Predict for 120 months ahead for all subject
    Each batch contains #batch_size subjects (one sequence per subject)
    """
    model.eval()
    ret = {"subjects": dataloader.dataset.subject_list}
    ret["DX"] = []  # 1. likelihood of CN, MCI, and Dementia
    ret["MMSE"] = []  # 2. (best guess, upper and lower bounds on 50% CI)
    ret["Ventricles"] = []  # 3. (best guess, upper and lower bounds on 50% CI)
    ret["dates"] = misc.make_date_col(
        [pred_start[s] for s in dataloader.dataset.subject_list],
        duration,
        site,
    )
    col = ["MMSE", "Ventricles", "ICV"]
    indices = misc.get_index(list(data["fields"]), col)
    mean = model.mean[col].values.reshape(1, -1)
    stds = model.stds[col].values.reshape(1, -1)
    for idx, batch in enumerate(dataloader):
        icat, ival, len_list = batch

        icat = icat.detach().cpu().numpy()
        ival = ival.detach().cpu().numpy()

        cat_out, val_out = predict_subject(
            model, icat, ival, duration, len_list
        )

    for i, l in enumerate(len_list):
        oval = val_out[l - 1: l + duration - 1, i, indices] * stds + mean
        ret["DX"].append(cat_out[l - 1: l + duration - 1, i, :])
        ret["MMSE"].append(misc.add_ci_col(oval[:, 0], 1, 0, 85))
        ret["Ventricles"].append(
            misc.add_ci_col(oval[:, 1] / oval[:, 2], 5e-4, 0, 1)
        )

    return ret


def load_checkpoint(model_dir, data):
    """
    Load best model configurations and initialize model with them
    """
    mean = data["mean"]
    stds = data["stds"]
    config_file = path.join(model_dir, f"config.json")
    with open(config_file) as fhandler:
        config = json.load(fhandler)
    best_trial = config["best_trial"]
    model_class = MODEL_DICT["MinRNN"]

    model = model_class(
        nb_classes=3,
        nb_measures=config["nb_measures"],
        nb_layers=config["nb_layer"],
        h_size=2 ** config["h_size"],
        h_drop=config["h_drop"] / 10.0,
        i_drop=config["i_drop"] / 10.0,
    )

    model_checkpoint = path.join(model_dir, f"trial{best_trial}.pt")
    model.load_state_dict(torch.load(model_checkpoint))

    setattr(model, "mean", mean)
    setattr(model, "stds", stds)

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
        "MinimalRNN",
        args.site,
        "old_metric",
        f"fold_{fold}",
    )
    new_metric_path = path.join(
        global_config.prediction_dir,
        "MinimalRNN",
        args.site,
        "new_metric",
        f"fold_{fold}",
    )
    makedirs(old_metric_path, exist_ok=True)
    makedirs(new_metric_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = path.join(
        global_config.minrnn_path, args.site, f"test.f{fold}.pkl"
    )
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    model_dir = path.join(
        global_config.checkpoint_dir,
        "MinimalRNN",
        args.train_site,
        f"model.f{fold}",
    )
    model = load_checkpoint(model_dir, data).to(device)

    test_data = TestDataset(data["test"])
    test_dataloader = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate_test,
    )
    prediction = predict(
        model,
        test_dataloader,
        data,
        data["pred_start"],
        data["duration"],
        args.site,
    )
    csv_path = path.join(old_metric_path, "prediction.csv")
    misc.build_pred_frame(prediction, csv_path)

    if args.in_domain:
        ref_frame = path.join(
            global_config.clean_gt_path, args.site, f"fold{fold}_test.csv"
        )
    else:  # external test, no split
        ref_frame = path.join(
            global_config.clean_gt_path, args.site, f"test_gt.csv"
        )

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
