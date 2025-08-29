'''
Written by Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script prepares and transforms brain imaging features for downstream classification tasks 
(AD classification or MCI progression prediction), using features derived from 
a pretrained Pyment brain age model (Leonardsen et al., 2022). It generates logistic regression 
training inputs, selects optimal hyperparameters, converts pretrained weights from TensorFlow 
to PyTorch, and initializes fine-tuning models.

Outputs:
- `*_id_dx_features.npy`: diagnosis + 64D feature arrays for each dataset split.
- `*_all_datasets_id_dx_features.npy`: combined diagnosis + features across datasets.
- `coef.npy`, `intercept.npy`: trained logistic regression weights and biases.
- `SFCN_FC_initialized.pth`: PyTorch model initialized with converted weights.
- `*_ids.txt`, `*_dx.txt`, `*_64d.npy`: separate IDs, labels, and feature arrays for loading.
- Completion flags for pipeline step control (e.g., `brainage64D_TW.complete`).

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m model.pyment.CBIG_BA_transfer_weights --task ad_classification --in_dir \
        /not/real/path --logreg_in_dir /not/real/path --logreg_out_dir \
            /not/real/path --initialized_sfcn_fc_dir /not/real/path \
                 --logreg_pred /not/real/path --num_split 50 --step 50 \
                    --current_trainval_size 50 --full_trainval_size 997 \
                        --model_type SFCN_FC;
'''

import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError("ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from CBIG_BA_config import global_config
from model.cnn.CBIG_BA_train import set_random_seeds
from model.cnn.modules.CBIG_BA_sfcn import SFCN_FC
from model.cnn.modules.CBIG_BA_helper import init_weights_bias
sys.path.append(global_config.PYMENT_DIR)
from pyment.models import RegressionSFCN
from utils.CBIG_BA_complete_step import generate_complete_flag


def generate_logreg_input(matched_dir, in_base_dir, in_dir, out_base_dir,
                          seeds, task, dataset, processid_col, dx_col, current_trainval_size, max_trainval_size):
    '''
    Generate diagnosis + 64D feature arrays as NumPy files for logistic regression.

    Combines diagnosis and Pyment-derived features into a [#scans x 66] array for each dataset split
    (train, val, test). Each row includes [ID, dx, 64D features]. One file is saved per train/val/test split.

    Parameters
    ----------
    matched_dir : str
        Directory containing balanced diagnosis files for each dataset.
    in_base_dir : str
        Base directory for existing data splits.
    in_dir : str
        Directory containing ID_features.csv files with extracted 64D features.
    out_base_dir : str
        Directory to save output logreg input .npy arrays.
    seeds : int
        Number of seeds to loop over for different data splits.
    task : str
        Task type, either 'ad_classification' or 'mci_progression'.
    dataset : str
        Name of the dataset (e.g., AIBL, ADNI, MACC).
    processid_col : str
        Column name for subject/process ID.
    dx_col : str
        Column name for diagnosis.
    current_trainval_size : int
        Current number of samples in train+val set.
    max_trainval_size : int
        Maximum size of train+val set.
    """
    '''

    splits = ['train', 'val', 'test']

    # df_balanced, df_id_features, df_balanced_id_features, df_id_dx, df_features,
    # and df_id_dx_features comprise all scans from a dataset.

    # df_balanced contains scan ID + dx
    df_balanced = pd.read_csv('{}/{}/{}_balanced.csv'.format(
        matched_dir, task, dataset))

    # df_id_features contains scan ID + features
    df_id_features = pd.read_csv('{}/{}/ID_features.csv'.format(
        in_dir, dataset))
    df_id_features = df_id_features.rename(columns={'id': processid_col})

    # df_balanced_id_features contains scan ID + dx + features
    df_balanced_id_features = df_balanced.merge(df_id_features,
                                                on=processid_col,
                                                how='right')

    # df_id_dx contains scan ID + dx (with renamed dx column)
    df_id_dx = df_balanced_id_features[[processid_col, dx_col]]

    # df_features contains features
    df_features = df_balanced_id_features[list(map(str, range(1, 64 + 1)))]

    # df_id_dx_features contains scan ID + dx (with renamed dx column) + features
    df_id_dx_features = pd.concat([df_id_dx, df_features], axis=1)

    for seed in range(seeds):
        out_dir = os.path.join(out_base_dir, dataset, str(current_trainval_size),
                                'seed_{}'.format(seed))
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        # df_reference_split, and df_splits comprise only scans from a dataset, for a specific split
        for split in splits:
            # if full train+val size or test split, then point to full data split folder
            if (current_trainval_size == max_trainval_size) or (split == 'test'):
                df_reference_split = pd.read_csv(
                    '{}/{}/full/seed_{}/{}.csv'.format(
                        in_base_dir, task, seed, split))
            # else, point to corresponding non-full train+val size data split folder
            else:
                df_reference_split = pd.read_csv(
                    '{}/{}/{}/seed_{}/{}.csv'.format(
                        in_base_dir, task, current_trainval_size, seed, split))

            # df_reference_split contains scan ID for dataset, for split
            df_reference_split = df_reference_split[
                df_reference_split['DATASET'] == dataset]

            # df_splits contains dx + features for dataset, for split, df_splits has additional scan ID column
            df_splits = df_id_dx_features.merge(
                df_reference_split[[processid_col]],
                on=processid_col,
                how='right')
            np.save('{}/{}_id_dx_features'.format(out_dir, split),
                    df_splits)


def concat_all_datasets(out_base_dir, seeds, current_trainval_size):
    """
    Concatenate logreg input features from all datasets into a unified array for each split.

    Combines AIBL, ADNI, and MACC diagnosis+features arrays into one per split (train, val, test) 
    and saves them.

    Parameters
    ----------
    out_base_dir : str
        Base directory containing the *_id_dx_features.npy files.
    seeds : int
        Number of data splits (seeds) to loop over.
    current_trainval_size : int
        Current number of samples in train+val set.
    """

    # Concatenate features within each split, across datasets
    splits = ['train', 'val', 'test']
    for seed in range(seeds):
        out_dir = os.path.join(out_base_dir, 'all_datasets',
                                str(current_trainval_size), 'seed_{}'.format(seed))
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        for split in splits:
            aibl_arr = np.load(
                '{}/AIBL/{}/seed_{}/{}_id_dx_features.npy'.format(
                    out_base_dir, current_trainval_size, seed, split),
                allow_pickle=True)
            adni_arr = np.load(
                '{}/ADNI/{}/seed_{}/{}_id_dx_features.npy'.format(
                    out_base_dir, current_trainval_size, seed, split),
                allow_pickle=True)
            macc_arr = np.load(
                '{}/MACC/{}/seed_{}/{}_id_dx_features.npy'.format(
                    out_base_dir, current_trainval_size, seed, split),
                allow_pickle=True)
            all_datasets_arr = np.vstack((aibl_arr, adni_arr, macc_arr))
            np.save(
                '{}/{}_all_datasets_id_dx_features'.format(out_dir, split),
                all_datasets_arr)


def convert_labels_char2num(y_in, task):
    """
    Convert string diagnosis labels into numeric binary labels.

    Parameters
    ----------
    y_in : np.ndarray
        Array of diagnosis strings ('NCI', 'AD', 'sMCI', 'pMCI').
    task : str
        Task type ('ad_classification' or 'mci_progression').

    Returns
    -------
    np.ndarray
        Integer array where diagnoses are encoded as 0 or 1.
    """
    if task == 'ad_classification':
        y = np.where(y_in == 'NCI', 0, y_in)
        y_out = np.where(y == 'AD', 1, y)
    else:
        y = np.where(y_in == 'sMCI', 0, y_in)
        y_out = np.where(y == 'pMCI', 1, y)
    y_out = y_out.astype('int')

    return y_out


def generate_coef_and_intercept(in_dir, out_base_dir, seeds,
                                current_trainval_size, random_state, task):
    """
    Background: last layer in pre-trained brain age model was optimized for
                predicting brain age. Hence cannot use last layer as
                starting point for fine-tuning of Brainage64D-finetune
                (used for AD classification).
    High-level goal: initialize final fully connected layer of
                     Brainage64D-finetune (before fine-tuning), whereby the
                     fully connected layer is used for AD classification.
    Function purpose: upstream functions already generated 64-dimensional
                      brain age features. These are used as input to
                      logistic regression, for AD classification.
                      Regularization hyperparameter, C, is optimized on the
                      validation set, and using the optimized C, logistic
                      regression re-trains on the training split. The
                      coefficients and intercepts of logistic regression are
                      saved, to-be-used for initialization of the weights and
                      biases of Brainage64D-finetune (before fine-tuning)
                      downstream.

    Parameters
    ----------
    in_dir : str
        Input directory containing train/val 64D feature arrays.
    out_base_dir : str
        Output directory to save coefficients and intercepts.
    seeds : int
        Number of seeds to process.
    current_trainval_size : int
        Current number of samples in train+val set.
    random_state : int
        Random seed for reproducibility.
    task : str
        Classification task type.
    """

    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for seed in range(seeds):
        out_dir = os.path.join(out_base_dir, str(current_trainval_size),
                               'seed_{}'.format(seed))
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        # load train and val arrays, which are of dimension [# scans x 65].
        # 1 column is for dx, other 64 columns are for 64D features.
        train = np.load(
            '{}/all_datasets/{}/seed_{}/train_all_datasets_id_dx_features.npy'.
            format(in_dir, current_trainval_size, seed),
            allow_pickle=True)
        val = np.load(
            '{}/all_datasets/{}/seed_{}/val_all_datasets_id_dx_features.npy'.
            format(in_dir, current_trainval_size, seed),
            allow_pickle=True)

        # split arrays into features and labels
        X_train = train[:, 2:]
        y_train = train[:, 1]
        y_train = convert_labels_char2num(y_train, task)

        X_val = val[:, 2:]
        y_val = val[:, 1]
        y_val = convert_labels_char2num(y_val, task)

        # optimize regularization hyperparameter C
        C_dict = {}
        for C in C_range:
            clf = LogisticRegression(solver='saga',
                                     random_state=random_state,
                                     C=C,
                                     multi_class='multinomial',
                                     max_iter=100000).fit(X_train, y_train)
            C_dict[C] = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        C_optim = max(C_dict, key=C_dict.get)

        # re-train classifier on training set with optimized hyperparameter
        clf_optim = LogisticRegression(solver='saga',
                                       random_state=random_state,
                                       C=C_optim,
                                       multi_class='multinomial',
                                       max_iter=100000).fit(X_train, y_train)

        # generate log reg coefficients (to-be-used as weights)
        coef_positive_class = clf_optim.coef_
        coef_negative_class = -coef_positive_class
        coef = np.concatenate((coef_negative_class, coef_positive_class),
                              axis=0)
        np.save('{}/coef'.format(out_dir), coef)

        # generate log reg intercepts (to-be-used as biases)
        intercept_positive_class = clf_optim.intercept_
        intercept_negative_class = -intercept_positive_class
        intercept = np.concatenate(
            (intercept_negative_class, intercept_positive_class), axis=0)
        np.save('{}/intercept'.format(out_dir), intercept)


def obtain_tensorflow_weights():
    """
    Load pretrained weights from TensorFlow-based Pyment regression model.

    Returns
    -------
    list of np.ndarray
        List of model weights from the TensorFlow model.
    """
    tf_model = RegressionSFCN(weights='brain-age', include_top=True)
    tf_weights = tf_model.get_weights()
    return tf_weights


def transfer_weights_tensorflow2pytorch(out_base_dir,
                                        random_state,
                                        tf_weights,
                                        model_type,
                                        in_dir=None,
                                        current_trainval_size=None,
                                        seeds=None):
    """
    Convert TensorFlow (TF) Pyment weights to PyTorch format and initialize SFCN/SFCN_FC model.

    Transfers convolution and batchnorm layers from Pyment TF model to PyTorch equivalent,
    and initializes final FC layer weights using logistic regression if SFCN_FC.

    Parameters
    ----------
    out_base_dir : str
        Output directory to save initialized PyTorch model.
    random_state : int
        Random seed for initialization.
    tf_weights : list
        TensorFlow model weights.
    model_type : str
        Model architecture type ('SFCN' or 'SFCN_FC').
    in_dir : str, optional
        Directory containing logistic regression weights (only for SFCN_FC).
    current_trainval_size : int, optional
        Number of training samples used (required for SFCN_FC).
    seeds : int, optional
        Number of seeds (required for SFCN_FC).
    """

    # initialize PyTorch SFCN model
    torch_model = SFCN_FC()
    set_random_seeds(random_state)
    torch_model.apply(init_weights_bias)

    # establishing iterables
    conv_block = ['conv_0', 'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    conv_block_layer = ['conv3d', 'batchnorm']
    params = ['weight', 'bias']
    params_bn = params + ['running_mean', 'running_var']

    tf_weight_count = 0
    for conv_n in conv_block:
        for i, layer in enumerate(conv_block_layer):
            # if conv3d, transfer only weight and bias
            if layer == 'conv3d':
                for param in params:
                    # if weight, transpose.
                    if param == 'weight':
                        getattr(
                            getattr(torch_model.feature_extractor, conv_n)[i],
                            param).data = torch.from_numpy(
                                tf_weights[tf_weight_count]).permute(
                                    [4, 3, 0, 1, 2])
                    # other parameters, don't transpose.
                    else:
                        getattr(
                            getattr(torch_model.feature_extractor, conv_n)[i],
                            param).data = torch.from_numpy(
                                tf_weights[tf_weight_count])
                    tf_weight_count += 1
            else:
                # if batchnorm, transfer additional parameters: running_mean, and running_var
                for param in params_bn:
                    getattr(
                        getattr(torch_model.feature_extractor, conv_n)[i],
                        param).data = torch.from_numpy(
                            tf_weights[tf_weight_count])
                    tf_weight_count += 1

    # save initialized state
    if model_type == "SFCN_FC":
        for seed in range(seeds):
            out_dir = os.path.join(out_base_dir, str(current_trainval_size),
                                   'seed_{}'.format(seed))
            if not os.path.exists(out_dir):
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            torch_model.fc[0].weight.data = torch.from_numpy(
                np.load('{}/{}/seed_{}/coef.npy'.format(
                    in_dir, current_trainval_size, seed)))
            torch_model.fc[0].bias.data = torch.from_numpy(
                np.load('{}/{}/seed_{}/intercept.npy'.format(
                    in_dir, current_trainval_size, seed)))
            torch.save(torch_model.state_dict(),
                       '{}/SFCN_FC_initialized.pth'.format(out_dir))
    elif model_type == "SFCN":
        torch_model.fc[0].weight.data = torch.from_numpy(tf_weights[36].T)
        torch_model.fc[0].bias.data = torch.from_numpy(tf_weights[37])
        os.makedirs(out_base_dir, exist_ok=True)
        torch.save(torch_model.state_dict(),
                   os.path.join(out_base_dir, "SFCN_initialized.pth"))

        # Generate complete flags for next step to commence
        generate_complete_flag(out_base_dir, 'BAG_finetune_TW')
    else:
        print(
            f"model_type argument {model_type} not accepted. Only 'SFCN' or 'SFCN_FC' accepted."
        )


def convert_input_for_load_input(in_base_dir, out_base_dir, current_trainval_size,
                                 seeds, task, step):
    """
    Convert combined diagnosis+features arrays into ID, label, and feature files for model input.

    Splits each [ID, dx, 64D features] array into:
    - *_ids.txt
    - *_dx.txt
    - *_64d.npy

    Parameters
    ----------
    in_base_dir : str
        Input directory containing *_all_datasets_id_dx_features.npy files.
    out_base_dir : str
        Directory to save the split outputs.
    current_trainval_size : int
        Size of the train+val split.
    seeds : int
        Number of data split seeds.
    task : str
        Classification task ('ad_classification' or 'mci_progression').
    step : int
        Current training step (used for control flow flags).
    """
    splits = ['train', 'val', 'test']

    # convert scan IDs and diagnoses into 1 txt file each, save 64d features
    for seed in range(seeds):
        out_dir = os.path.join(out_base_dir, '{}'.format(current_trainval_size),
                                'seed_{}'.format(seed))
        if not os.path.exists(out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        for split in splits:
            in_file = os.path.join(
                in_base_dir, '{}'.format(current_trainval_size),
                'seed_{}'.format(seed),
                '{}_all_datasets_id_dx_features.npy'.format(split))
            arr = np.load(in_file, allow_pickle=True)
            id = arr[:, 0]
            dx = convert_labels_char2num(arr[:, 1], task)
            features = arr[:, 2:]
            np.savetxt(os.path.join(out_dir, '{}_ids.txt'.format(split)),
                        id,
                        fmt='%s')
            np.savetxt(os.path.join(out_dir, '{}_dx.txt'.format(split)),
                        dx,
                        fmt='%d')
            np.save(os.path.join(out_dir, '{}_64d'.format(split)),
                    features)

        # Generate complete flags for next step to commence
        generate_complete_flag(out_dir, 'brainage64D_TW')


def get_args():
    """
    Parse command-line arguments for configuring the pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        choices=['ad_classification', 'mci_progression'])
    parser.add_argument('--num_split', type=int, default=50)
    parser.add_argument('--step', type=int, default=50)
    parser.add_argument('--current_trainval_size', type=int)
    parser.add_argument('--full_trainval_size', type=int)
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--logreg_in_dir', type=str)
    parser.add_argument('--logreg_out_dir', type=str)
    parser.add_argument('--initialized_sfcn_fc_dir', type=str)
    parser.add_argument('--logreg_pred', type=str)
    parser.add_argument('--model_type',
                        choices=['SFCN', 'SFCN_FC'],
                        type=str,
                        default='SFCN_FC')

    return parser.parse_args()


if __name__ == "__main__":
    # Read input args
    args = get_args()
    model_type = args.model_type
    random_state = global_config.SEED

    # If model that outputs 64D features,
    if model_type == "SFCN_FC":
        # Read input args
        datasets = global_config.DATASETS
        seeds = args.num_split
        step = args.step
        current_trainval_size = args.current_trainval_size
        max_trainval_size = args.full_trainval_size
        processid_col = global_config.PROCESSID_COL
        dx_col = global_config.DX_COL
        matched_dir = global_config.MATCHED_DATA_DIR
        in_base_dir = global_config.DATA_SPLIT_DIR
        in_dir = args.in_dir
        task = args.task
        out_base_dir = args.logreg_in_dir

        # Loop through datasets to generate
        # input required for logistic regression
        for dataset in datasets:
            generate_logreg_input(matched_dir, in_base_dir, in_dir,
                                  out_base_dir, seeds, task, dataset,
                                  processid_col, dx_col, current_trainval_size, max_trainval_size)
        concat_all_datasets(out_base_dir, seeds, current_trainval_size)

        # Replace input args
        in_dir = args.logreg_in_dir
        out_base_dir = args.logreg_out_dir
        generate_coef_and_intercept(in_dir, out_base_dir, seeds,
                                    current_trainval_size, random_state, task)
        in_dir = args.logreg_out_dir
        out_base_dir = args.initialized_sfcn_fc_dir

        # Extract Tensorflow weights of pyment
        # regression variant model
        tf_weights = obtain_tensorflow_weights()

        # Convert weights from Tensorflow
        # to Pytorch
        transfer_weights_tensorflow2pytorch(out_base_dir, random_state,
                                            tf_weights, model_type, in_dir,
                                            current_trainval_size, seeds)

        # Replace input args
        in_base_dir = os.path.join(args.logreg_in_dir, 'all_datasets')
        out_base_dir = args.logreg_pred
        convert_input_for_load_input(in_base_dir, out_base_dir,
                                     current_trainval_size, seeds, task, step)
    # Elif model that outputs 1D/scalar predicted brain age,
    elif model_type == "SFCN":
        # Read input args
        out_base_dir = args.initialized_sfcn_fc_dir

        # Extract Tensorflow weights of pyment
        # regression variant model
        tf_weights = obtain_tensorflow_weights()

        # Convert weights from Tensorflow
        # to Pytorch
        transfer_weights_tensorflow2pytorch(out_base_dir, random_state,
                                            tf_weights, model_type)
    else:
        print(
            f"model_type argument {model_type} not accepted. Only 'SFCN' or 'SFCN_FC' accepted."
        )
