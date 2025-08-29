#!/usr/bin/env python
# Process Frog features using training set statistics
# Pre-processing includes: imputation, normalization, pickling
# Normalization is done through gauss_rank (implemented with code from
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
# https://www.kaggle.com/code/tottenham/10-fold-simple-dnn-with-rank-gauss)
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import pickle
from os import makedirs, path

import joblib
import numpy as np

from config import global_config
from data_processing.step5_gauss_data_process import gen_data


def main(args):
    """Main execution function"""

    # Set random seed
    np.random.seed(args.seed)

    fold = args.fold

    # Generate the save directory
    save_path = path.join(global_config.processed_frog, f"{args.site}")
    makedirs(save_path, exist_ok=True)

    # Normalizer file path (from pretrained model directory)
    norm_param_file = path.join(
        global_config.pretrained_dir, "L2C_FNN_ADNI_param.joblib"
    )

    # Extrat the normalizer for corresponding fold
    normalization_list = joblib.load(norm_param_file)
    normalization_dict = normalization_list[fold]

    # Perform data processing
    ret = gen_data(fold, args, normalization_dict)

    # Save imputed and transformed data
    save_file = path.join(save_path, f"fold_{fold}.pkl")
    with open(save_file, "wb") as fhandler:
        pickle.dump(ret, fhandler)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--use_adas", action="store_true")
    parser.add_argument("--independent_test", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
