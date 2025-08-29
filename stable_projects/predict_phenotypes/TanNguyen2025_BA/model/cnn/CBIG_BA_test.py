'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script uses a trained SFCN or SFCN_FC model to generate predictions on a test dataset
for either brain age regression or Alzheimer's disease classification tasks (AD classification
or MCI progression prediction).

Depending on the model type, it outputs either:
- Brain age predictions and MAE scores (for regression with SFCN),
- Classification predictions and AUC scores (for AD classification/
  MCI progression prediction with SFCN_FC).

For SFCN: MAE is computed and saved.
For SFCN_FC: ROC AUC is computed and saved.

Expected outputs include:
- `test_predict_score.csv`: prediction results on the test set.
- `params_test.json`      : file storing evaluation metrics.

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m model.cnn.CBIG_BA_test --data_split /not/real/path --model_dir /not/real/path \
        --test_csv /not/real/path --task ad_classification
'''

import json
import torch
import pandas as pd
import os
import argparse
from sklearn.metrics import mean_absolute_error, roc_auc_score
import torch.nn.functional as F
from model.cnn.modules.CBIG_BA_sfcn import SFCN, SFCN_FC
from model.cnn.modules.CBIG_BA_dataset import ADDataset

from CBIG_BA_config import global_config
from model.utils.CBIG_BA_io import read_json
from utils.CBIG_BA_complete_step import generate_complete_flag
import numpy as np


def predict_dataset(args,
                    model,
                    t1_dirs,
                    test_csv,
                    task,
                    device,
                    verbose=False):
    """
    Generates predictions using a trained model on a test dataset.

    Args:
        args (Namespace): Parsed command-line arguments.
        model (nn.Module): Trained PyTorch model (SFCN or SFCN_FC).
        t1_dirs (dict): Dictionary mapping dataset names to T1 image directories.
        test_csv (str): Path to CSV file containing test scan IDs and labels.
        task (str): Task type, e.g., 'ad_classification' or 'mci_progression'.
        device (torch.device): Device to perform inference (CPU/GPU).
        verbose (bool): If True, print detailed results to stdout.

    Returns:
        output (pd.DataFrame): DataFrame of predictions with actual and predicted labels/scores.
        score (float): Evaluation score (MAE or AUC) depending on `args.score`.
    """
    # Read data
    test_data = ADDataset(t1_dirs, label_csv=test_csv, task=task)
    print('Number of test samples', len(test_data))

    preds = []
    for img_id, t1_input, age, label in test_data:
        t1_input = t1_input.to(device)
        # print("t1_input at GPU?", t1_input.is_cuda)

        t1_input = t1_input.unsqueeze(0)
        # print('t1_input', t1_input.shape)

        # Inference
        with torch.no_grad():
            output = model(t1_input)
        if args.score == 'mae':
            pred_target = output.item()
        elif args.score == 'auc':
            _, pred = torch.max(output, 1)
            # Compute softmax for AUC
            outputs_sm = F.softmax(output, dim=1)
            # print('outputs_sm sum', outputs_sm.sum())
            pred_score = outputs_sm[:, 1].item()
            pred_target = pred.item()

        if args.model_type == 'SFCN':
            preds.append([img_id, age, pred_target])
            print(img_id, 'Actual', age, 'Predict', pred_target,
                  'pred_{}'.format(args.score))
            output = pd.DataFrame(
                preds, columns=['PROCESS_ID', 'TARGET_GT', 'TARGET_PREDICT'])
        elif args.model_type == 'SFCN_FC':
            preds.append([img_id, label, pred_target, pred_score])
            print(img_id, 'Actual', label, 'Predict', pred_target,
                  f'pred_{args.score}', pred_score)
            output = pd.DataFrame(preds,
                                  columns=[
                                      'PROCESS_ID', 'TARGET_GT',
                                      'TARGET_PREDICT', 'PRED_SCORE'
                                  ])

    # Print score to log
    if verbose:
        gt_list = output['TARGET_GT'].to_list()
        if args.score == 'mae':
            pred_target_list = output['TARGET_PREDICT'].to_list()
            score = mean_absolute_error(gt_list, pred_target_list)
        elif args.score == 'auc':
            pred_score_list = output['PRED_SCORE'].to_list()
            score = roc_auc_score(gt_list, pred_score_list)

    return output, score


def main_1_split(args):
    """
    Handles inference for a single data split using either SFCN or SFCN_FC.

    For SFCN:
        - Loads pretrained scalar-output model for brain age regression.
        - Predicts and evaluates MAE.

    For SFCN_FC:
        - Iterates over subdirectories containing trained classification models.
        - Predicts and evaluates ROC AUC.

    Args:
        args (Namespace): Parsed command-line arguments.
        
    Returns:
        float: Mean Absolute Error (MAE) score if model_type is 'SFCN'.
    """
    # Read input args
    t1_dirs = global_config.T1_DIRS
    test_csv = args.test_csv
    model_dir = args.model_dir

    # If model that outputs 1D/scalar predicted brain age,
    if args.model_type == 'SFCN':
        trained_model_pt = os.path.join(args.initialized_sfcn_fc_dir,
                                        "SFCN_initialized.pth")
        output_csv = os.path.join(model_dir, 'test_predict_score.csv')
        params_test_json = os.path.join(model_dir, 'params_test.txt')
        #assert not os.path.exists(output_csv), f"{output_csv} already existed!"
        #assert not os.path.exists(
            #params_test_json), f"{params_test_json} already existed!"

        # Read trained model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SFCN()
        model.load_state_dict(torch.load(trained_model_pt,
                                         map_location=device))
        model = model.to(device)
        model.eval()

        # Write test MAE
        results, score = predict_dataset(args,
                                         model,
                                         t1_dirs,
                                         test_csv,
                                         args.task,
                                         verbose=True,
                                         device=device)
        line_to_write = 'test_{} = {}'.format(args.score, score)

        # Save predicted outputs to file
        results.to_csv(output_csv, index=False)
        with open(params_test_json, 'w') as f:
            f.write(line_to_write)

        # Clear cache
        del model
        torch.cuda.empty_cache()

        # Return MAE
        return score
    # Elif model that outputs 64D features,
    elif args.model_type == 'SFCN_FC':
        trained_dirs = [
            i for i in os.listdir(model_dir) if i not in args.exclude_dirs
        ]
        for model_id in trained_dirs:
            trained_model_pt = os.path.join(model_dir, model_id, 'best_auc.pt')
            output_csv = os.path.join(model_dir, model_id,
                                      'test_predict_score.csv')
            params_json = os.path.join(model_dir, model_id, 'params.json')
            params_test_json = os.path.join(model_dir, model_id,
                                            'params_test.json')
            #assert not os.path.exists(
                #output_csv), f"{output_csv} already existed!"
            #assert not os.path.exists(
                #params_test_json), f"{params_test_json} already existed!"

            # Read trained model
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            model = SFCN_FC()
            model.load_state_dict(
                torch.load(trained_model_pt, map_location=device))
            model = model.to(device)
            model.eval()

            # Write test AUC
            results, auc = predict_dataset(args,
                                           model,
                                           t1_dirs,
                                           test_csv,
                                           args.task,
                                           verbose=True,
                                           device=device)
            data = read_json(params_json)
            data['test_auc'] = auc

            # Save predicted outputs to file
            results.to_csv(output_csv, index=False)
            with open(params_test_json, 'w') as fp:
                json.dump(data, fp, indent=4)

            # Clear cache
            del model
            torch.cuda.empty_cache()


def get_args():
    """
    Get arguments from command line and return args.
    """
    parser = argparse.ArgumentParser()

    # general parameters - folders and files
    parser.add_argument('--data_split',
                        nargs="+",
                        type=int,
                        help='List of dataseed')
    parser.add_argument('--test_csv', help='')
    parser.add_argument('--model_dir', help='')
    parser.add_argument(
        '--exclude_dirs',
        nargs="+",
        help='List of folders to exclude (e.g., symbolic links or log)',
        default=['best_model', 'log', 'complete_train'])
    parser.add_argument('--task',
                        type=str,
                        choices=['ad_classification', 'mci_progression'],
                        help='Task name. Used to determine dataset labels')
    parser.add_argument('--score',
                        type=str,
                        choices=['mae', 'auc'],
                        default='auc')
    parser.add_argument('--model_type',
                        type=str,
                        choices=['SFCN', 'SFCN_FC'],
                        help='Model architecture',
                        default='SFCN_FC')
    parser.add_argument('--initialized_sfcn_fc_dir', type=str)

    return parser.parse_args()


def main(args):
    """
    Read command line arguments, change input data split folder and pretrained file based on the current seed_id.
    """
    print('Input arguments:', str(args))

    # Read input arguments
    model_dir_template = args.model_dir
    test_csv_template = args.test_csv

    # Remove data split list out of input argument list
    data_split_list = args.data_split
    del args.data_split

    # If model that outputs 1D/scalar predicted brain age,
    if args.model_type == 'SFCN':
        splits_score = []
        for seed_id in data_split_list:
            args.model_dir = model_dir_template.replace('SEEDID', str(seed_id))
            args.test_csv = test_csv_template.replace('SEEDID', str(seed_id))
            print("model_dir = ", args.model_dir)
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir, exist_ok=True)
            split_score = main_1_split(args)
            splits_score += [split_score]

            # Generate complete flags for next step to commence
            generate_complete_flag(args.model_dir, append='test')

        # Compute MAE stats
        splits_score = np.array(splits_score)
        mean_score = np.mean(splits_score)
        std_score = np.std(splits_score)

        # Output stats file
        output_string = "mean ± std MAE across splits = {:.3f} ± {:.3f}".format(
            mean_score, std_score)
        output_file = os.path.join(model_dir, "mean_mae.txt")
        with open(output_file, 'w') as file:
            file.write(output_string)
    # Elif model that outputs 64D features,
    elif args.model_type == 'SFCN_FC':
        for seed_id in data_split_list:
            args.model_dir = model_dir_template.replace('SEEDID', str(seed_id))
            args.test_csv = test_csv_template.replace('SEEDID', str(seed_id))
            print("model_dir = ", args.model_dir)
            main_1_split(args)

            # Generate complete flags for next step to commence
            generate_complete_flag(args.model_dir, append='test')


if __name__ == "__main__":
    main(get_args())
