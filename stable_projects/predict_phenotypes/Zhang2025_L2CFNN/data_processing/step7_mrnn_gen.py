#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
import argparse
import pickle
from os import makedirs, path

import data_processing.mrnn_dataloader as dataloader
import data_processing.mrnn_misc as misc
from config import global_config


def get_data(args, feature_cols, site, train_data=None):
    """
    Generate training/test data batches and save as pickle file
    *args* specify
        args: user input arguments
        feature_cols: feature columns
        site: site the data is from
        train_data: for external test data gen, training statistics required
    """

    ret = {"fields": feature_cols}

    if args.in_domain:
        split_mask_path = path.join(
            global_config.split_path, f"{site}/fold{args.fold}_mask.csv"
        )
    else:
        split_mask_path = path.join(
            global_config.split_path, f"{site}/test_mask.csv"
        )

    train_mask, pred_mask, pred_mask_frame = misc.get_mask(
        split_mask_path, args.validation, site, args.in_domain
    )

    ret["baseline"], ret["pred_start"] = misc.get_baseline_prediction_start(
        pred_mask_frame, site
    )

    ret["duration"] = args.forecast_horizon

    columns = ["RID", "Month_bl", "DX"] + feature_cols
    frame = misc.load_table(
        path.join(global_config.raw_data_path, f"{site}.csv"), columns, site
    )

    if args.in_domain:
        tf = frame.loc[train_mask, feature_cols]
        ret["mean"] = tf.mean()
        ret["stds"] = tf.std()
        ret["VentICVstd"] = (tf["Ventricles"] / tf["ICV"]).std()
    else:
        ret["mean"] = train_data["mean"]
        ret["stds"] = train_data["stds"]
        ret["VentICVstd"] = train_data["VentICVstd"]

    frame[feature_cols] = (frame[feature_cols] - ret["mean"]) / ret["stds"]

    default_val = {f: 0.0 for f in feature_cols}
    default_val["DX"] = 0.0

    if args.in_domain:
        data = dataloader.extract(
            frame[train_mask], args.strategy, feature_cols, default_val
        )
        ret["train"] = data[0]
        print("train", len(ret["train"]), "subjects")

    data = dataloader.extract(
        frame[pred_mask], args.strategy, feature_cols, default_val
    )

    ret["test"] = data[0]

    print("test", len(ret["test"]), "subjects")
    print(len(feature_cols), "features")

    return ret


def main(args):
    """Main execution function"""

    # Generate the save directory
    save_path = path.join(global_config.minrnn_path, f"{args.site}")
    makedirs(save_path, exist_ok=True)

    feature_dict = global_config.feature_dict
    feature_set = "MinFull" if args.use_full else "PARTIAL"
    feature_cols = feature_dict[feature_set]

    if args.in_domain:
        train_data = None
    else:
        train_data_path = path.join(
            global_config.minrnn_path,
            f"{args.train_site}/test.f{args.fold}.pkl",
        )
        with open(train_data_path, "rb") as f:
            train_data = pickle.load(f)

    ret = get_data(args, feature_cols, args.site, train_data)
    split_ver = "val" if args.validation else "test"

    out_path = path.join(save_path, f"{split_ver}.f{args.fold}.pkl")
    with open(out_path, "wb") as fhandler:
        pickle.dump(ret, fhandler)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--train_site", default="ADNI")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--in_domain", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--forecast_horizon", type=int, default=120)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
