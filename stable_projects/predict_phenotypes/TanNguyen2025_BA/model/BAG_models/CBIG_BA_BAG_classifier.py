'''
Written by Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script computes the test ROC AUC performance for BAG and BAG-finetune models over
50 random test splits. ROC AUC is computed using trapezoidal rule.

Expected output(s):
    1. `bag_stats.txt`        : text file containing mean ± std of test ROC AUC
                                for AD classification.
    2. `test_auc_50seeds.mat` : 1x50 double MATLAB array containing ROC AUC values
                                for each of 50 random test splits.

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m model.BAG_models.CBIG_BA_BAG_classifier --in_dir /not/real/path \
        --out_dir /not/real/path --model_name BAG;
'''

import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import scipy.io
import argparse
TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError("ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from utils.CBIG_BA_complete_step import generate_complete_flag


def compute_tpr_fpr_for_split(model_name, in_dir, split):
    """
    Computes the True Positive Rate (TPR) and False Positive Rate (FPR) for a given test split
    by thresholding the predicted brain age gap (BAG) scores.

    For each participant in the test set, it uses that participant's BAG as a threshold to classify
    all other participants, generating a set of TPR and FPR values for the split.

    Parameters:
        model_name (str): The name of the model to evaluate. Must be one of:
                          - 'BAG': base model using full test array
                          - 'BAG_finetune': fine-tuned model with separate BAG and label files
        in_dir (str): Input directory containing input test data.
        split (int): Random seed index (e.g., from 0 to 49) indicating which test split to load.

    Returns:
        tpr_split (list of float): True Positive Rates for all thresholded predictions.
        fpr_split (list of float): False Positive Rates for all thresholded predictions.

    """

    # Extract input test ground truth diagnoses & brain age gap values
    if model_name == 'BAG':
        test_arr = np.load(os.path.join(in_dir, 'all_datasets', '997', 'seed_{}'.format(split), \
            'test_all_datasets_id_dx_bag.npy'), allow_pickle=True)
        test_bag = test_arr[:, 5]
        test_bag = test_bag.astype(float)
        test_true_dx = test_arr[:, 1]
        test_true_dx = np.where(test_true_dx == 'NCI', 0, test_true_dx)
        test_true_dx = np.where(test_true_dx == 'AD', 1, test_true_dx)
        test_true_dx = test_true_dx.astype(int)
    elif model_name == 'BAG_finetune':
        test_bag = np.load(os.path.join(in_dir, 'seed_{}'.format(split),
                                        'test_bag.npy'),
                           allow_pickle=True)
        test_true_dx = np.load(os.path.join(in_dir, 'seed_{}'.format(split),
                                            'test_dx.npy'),
                               allow_pickle=True)

    # Double-check identical number of participants
    assert test_bag.shape[0] == test_true_dx.shape[
        0], "for test, bag shape {} != dx shape {}".format(
            test_bag.shape[0], test_true_dx.shape[0])

    # Perform prediction using brain age gap
    tpr_split, fpr_split = [], []
    num_test_pts = test_bag.shape[0]
    # For each test participant,
    for idx_pt in range(num_test_pts):
        # Initialize true/false positives/negatives
        tn = 0
        fp = 0
        fn = 0
        tp = 0

        # Use current test participant's brain age gap
        # as a threshold to classify
        test_pred_dx = test_true_dx.copy()
        pred_thresh = test_bag[idx_pt]

        # If test participants' brain age gaps are equal
        # or lower than current test participant's,
        # then classify as cognitively normal. Else,
        # classify as having Alzheimer's disease.
        pred_mask = test_bag[:] <= pred_thresh
        test_pred_dx[pred_mask] = 0
        test_pred_dx[~pred_mask] = 1

        # Compute true/false positive/negatives to compute
        # true positive rate & false positive rate.
        tn, fp, fn, tp = confusion_matrix(test_true_dx, test_pred_dx).ravel()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tpr_split.append(tpr)
        fpr_split.append(fpr)

    return tpr_split, fpr_split

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--model_name', choices=['BAG', 'BAG_finetune'])

    return parser.parse_args()

if __name__ == "__main__":
    # Define input arguments
    args = get_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    model_name = args.model_name

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # For each of 50 random test splits,
    # compute 1x ROC AUC value
    auc_splits = []
    for split in range(50):
        tpr_split, fpr_split = compute_tpr_fpr_for_split(model_name, in_dir, split)
        tpr_split, fpr_split = np.array(tpr_split), np.array(fpr_split)
        tpr_split, fpr_split = tpr_split[np.argsort(fpr_split)], fpr_split[np.argsort(fpr_split)]
        auc_split = auc(fpr_split, tpr_split)
        auc_splits.append(auc_split)

    auc_splits = np.array(auc_splits)
    scipy.io.savemat(os.path.join(out_dir, 'test_auc_50seeds.mat'), {'test_auc': auc_splits})

    # Save mean ROC AUC value (out of 50 random test splits)
    mean_auc = np.mean(auc_splits)
    std_auc = np.std(auc_splits)

    output_file = os.path.join(out_dir, "bag_stats.txt")
    with open(output_file, 'w') as f:
        sys.stdout = f
        print("test AUC = {:.4f} ± {:.4f}".format(mean_auc, std_auc))
    sys.stdout = sys.__stdout__

    # Generate complete flags for next step to commence
    generate_complete_flag(out_dir, 'BAG_classifier')
