'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script performs logistic regression classification using intermediate 64D features
extracted from a trained neural network (e.g., SFCN_FC). It supports evaluation over
multiple random data splits and reports performance metrics such as AUC.

Expected outputs include:
- `test_predict_score.csv`: prediction results on the test set.
- `params_test.json`: JSON file storing the best hyperparameter and test performance.
- Summary logs and `.mat` files with test AUC values if multiple seeds are evaluated.

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m model.log_reg.CBIG_BA_logistic_regression --input_dir /not/real/path \
        --output_dir /not/real/path --model_name direct_ad;
'''

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.io import savemat
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import sys

TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from model.utils.CBIG_BA_io import txt2list
from CBIG_BA_config import global_config
from utils.CBIG_BA_complete_step import generate_complete_flag


def load_input(feature_npy, dx_txt, id_txt):
    """
    Load input features, diagnosis labels, and subject IDs.

    Args:
        feature_npy (str): Path to .npy file with extracted features.
        dx_txt (str): Path to .txt file with binary or multiclass diagnosis labels.
        id_txt (str): Path to .txt file with subject or image IDs.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: Features, labels, and IDs.
    """

    features = np.load(feature_npy, allow_pickle=True)
    labels = np.array([int(i) for i in txt2list(dx_txt)])
    ids = txt2list(id_txt)

    assert len(features) == len(labels), len(features) == len(ids)
    return features, labels, ids


def train_and_predict(input_data,
                      retrain_set,
                      regularizations,
                      seed,
                      model_name,
                      verbose=False):
    """
    Train logistic regression models with various regularization strengths,
    select best model based on validation AUC, and evaluate on test set.

    Args:
        input_data (dict): Dictionary with 'train', 'val', 'test' sets (features, labels, ids).
        retrain_set (str): Whether to retrain final model on 'train' or 'trainval' set.
        regularizations (List[float]): List of regularization parameters to try (C values).
        seed (int): Seed for model reproducibility.
        model_name (str): Name of model architecture ('direct_ad' or other variants).
        verbose (bool): If True, prints additional info.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray]: Best C, val AUC, test AUC, test scores, test predictions.
    """
    assert retrain_set in [
        'train', 'trainval'
    ], "retrain_set must be either 'train' or 'trainval'."

    # Read input to train, validation, and test sets
    train, val, test = input_data['train'], input_data['val'], input_data[
        'test']

    # Run through all hyperparameters
    best_val_auc, best_reg = -1, -1
    tune_log = {}
    for reg in regularizations:
        if model_name == 'direct_ad':
            model = LogisticRegression(solver="liblinear",
                                       random_state=seed,
                                       C=reg)
        else:
            model = LogisticRegression(solver="saga",
                                       random_state=1234,
                                       C=reg,
                                       multi_class='multinomial',
                                       max_iter=100000)
        model_fit = model.fit(train['X'], train['y'])
        auc = roc_auc_score(val['y'], model_fit.predict_proba(val['X'])[:, 1])

        if best_val_auc < auc:
            best_val_auc = auc
            best_reg = reg
        tune_log[reg] = {'auc': auc, 'model': model, 'model_fit': model_fit}

    # Retrain on train + validation
    if retrain_set == 'trainval':
        trainval = {
            'X': np.vstack((train['X'], val['X'])),
            'y': np.hstack((train['y'], val['y']))
        }
        model_trained = tune_log[best_reg]['model'].fit(
            trainval['X'], trainval['y'])
    else:
        model_trained = tune_log[best_reg]['model_fit']

    # Predict on test set
    test_proba = model_trained.predict_proba(test['X'])
    test_scores = test_proba[:, 1]
    test_predict_labels = test_proba.argmax(axis=1)
    test_auc = roc_auc_score(test['y'], test_scores)

    # Print stats
    if verbose:
        print('Best AUC: {:.4f} with regularization: {:.4f}'.format(
            best_val_auc, best_reg))
        print('Test AUC: {:.4f}'.format(test_auc))

    return best_reg, best_val_auc, test_auc, test_scores, test_predict_labels


def main_1_split(args):
    """
    Execute logistic regression training and evaluation on one seed (split).
    Saves model predictions and logs metrics for each data split.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Prepare output folders and files
    os.makedirs(args.output_dir, exist_ok=True)
    test_csv = os.path.join(args.output_dir, 'test_predict_score.csv')
    params_json = os.path.join(args.output_dir, 'params_test.json')
    #assert not os.path.exists(test_csv), f"{test_csv} already exists"
    #assert not os.path.exists(params_json), f"{params_json} already exists"

    # Splitting input matrices into features and targets
    input_data = {}
    for subset in ["train", "val", "test"]:
        features, labels, ids = load_input(
            feature_npy=os.path.join(args.input_dir,
                                     f'{subset}_{args.feature_dim}d.npy'),
            dx_txt=os.path.join(args.input_dir, f'{subset}_dx.txt'),
            id_txt=os.path.join(args.input_dir, f'{subset}_ids.txt'))

        input_data[subset] = {'X': features, 'y': labels, 'ids': ids}

    # Training and predicting
    best_reg, best_val_auc, test_auc, test_scores, test_predict = train_and_predict(
        input_data,
        retrain_set=args.retrain_set,
        regularizations=args.reg_values,
        seed=args.seed,
        model_name=args.model_name)

    # Save test scores
    test_scores_df = pd.DataFrame({
        'PROCESS_ID': ids,
        'TARGET_GT': input_data['test']['y'],
        'TARGET_PREDICT': test_predict,
        'PRED_SCORE': test_scores
    })
    test_scores_df.to_csv(test_csv, index=False)

    # Save hyperparameters and prediction AUC
    with open(os.path.join(args.output_dir, f'params_test.json'), 'w') as f:
        json.dump(
            {
                'best_reg': best_reg,
                'best_auc': best_val_auc,
                'test_auc': test_auc
            },
            f,
            indent=4)


def summary(args):
    """
    Summarizes AUC performance metrics across all seeds/data splits,
    and writes results to .csv and .mat files for further analysis.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    assert os.path.exists(
        args.output_dir), f"{args.output_dir} does not exist!"

    # Get timestamp for summary file
    start_time_fmt = datetime.now().strftime("%Y%m%d")
    summary_dir = os.path.join(args.output_dir, '_summary')
    os.makedirs(summary_dir, exist_ok=True)

    # Summary results of all the data_split
    best_model_all_splits = []
    for data_split in args.data_split:
        with open(
                os.path.join(args.output_dir, f'seed_{data_split}',
                             'params_test.json'), 'r') as f:
            data = json.load(f)
        best_model_all_splits.append(
            [data_split, data['best_reg'], data['best_auc'], data['test_auc']])

    # Save best model
    df = pd.DataFrame(
        best_model_all_splits,
        columns=['SEED_ID', 'BEST_REG', 'BEST_VAL_AUC', 'TEST_AUC'])

    # Save result to log_file
    log_file = os.path.join(
        summary_dir, 'results_{}seeds_{}.log'.format(len(args.data_split),
                                                     start_time_fmt))
    with open(log_file, 'w') as f:
        f.write('VALIDATION AUC = {:.4f} +/- {:.4f}\n'.format(
            np.mean(df['BEST_VAL_AUC']), np.std(df['BEST_VAL_AUC'], ddof=1)))
        f.write('TEST AUC = {:.4f} +/- {:.4f}\n'.format(
            np.mean(df['TEST_AUC']), np.std(df['TEST_AUC'], ddof=1)))

    # Save summary + AUC results for plotting
    output_csv = os.path.join(
        summary_dir, 'summary_{}seeds_{}.csv'.format(len(args.data_split),
                                                     start_time_fmt))
    output_mat = os.path.join(
        summary_dir, 'test_auc_{}seeds_{}.mat'.format(len(args.data_split),
                                                      start_time_fmt))
    #if os.path.exists(output_csv) or os.path.exists(output_mat):
    #raise FileExistsError(f"{output_csv} or {output_mat} already exists!")
    #else:
    df.to_csv(output_csv, index=False)
    savemat(output_mat, {'test_auc': df['TEST_AUC'].to_list()})


def main(args):
    """
    Read command line arguments, change input data split folder and pretrained file based on the current seed_id.
    """
    # Define input arguments
    input_dir_template = args.input_dir
    output_dir_template = args.output_dir
    if args.examples == 'n':
        print(f"Logistic regression on data splits: {args.data_split}...")

    # Determine if perform `summary` only or not?
    if args.summary_only == 'y':
        summary(args)
    else:
        for seed_id in args.data_split:
            args.input_dir = input_dir_template.replace('SEEDID', str(seed_id))
            args.output_dir = output_dir_template.replace(
                'SEEDID', str(seed_id))

            if not os.path.exists(args.input_dir):
                print(f"Split {seed_id} has no features, skipping.")
                continue
            main_1_split(args)

            # Generate complete flags for next step to commence
            generate_complete_flag(args.output_dir,
                                   '{}_LR'.format(args.model_name))

            # If examples, no need summary
            if args.examples == 'y':
                sys.exit(0)

        # Summary results
        args.output_dir = os.path.dirname(output_dir_template)
        summary(args)


def get_args():
    """
    Parses command-line arguments for logistic regression training and evaluation.

    Returns:
        argparse.Namespace: Argument object containing input paths, hyperparameters, etc.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        help=(
            "Input directory containing train, validation, and test data "
            "(features in .npy file, diagnosis in .txt, and subject's ID in .txt)"
        )
    )
    parser.add_argument('--output_dir', help="Output directory")
    parser.add_argument('--data_split',
                        nargs="+",
                        type=int,
                        help='List of datasplit ID',
                        default=list(range(50)))
    parser.add_argument('--reg_values',
                        nargs="+",
                        type=int,
                        help='List of C in sklearn logistic regression',
                        default=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
    parser.add_argument('--seed',
                        type=int,
                        help='Seed value',
                        default=global_config.SEED)
    parser.add_argument(
        '--retrain_set',
        choices=['trainval', 'train'],
        help='Set to retrain after choosing best hyperparameter',
        default='trainval')
    parser.add_argument(
        '--summary_only',
        choices=['y', 'n'],
        help=
        'If yes, output_dir must contain results from multiple data splits.',
        default='n')
    parser.add_argument('--model_name',
                        choices=[
                            'brainage64D', 'direct_ad',
                            'brainage64D_finetune_ad', 'log_reg'
                        ])
    parser.add_argument('--examples', choices=['y', 'n'], default='n')

    # Optional arugments
    parser.add_argument(
        '--feature_dim',
        help=
        'Dimension of the features for each sample (e.g. 64 to denote features at the second-to-last layer)',
        default=64)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
