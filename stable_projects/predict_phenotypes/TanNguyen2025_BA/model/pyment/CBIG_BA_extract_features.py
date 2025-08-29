'''
Written by Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script prepares data and extracts features using a pretrained Pyment brain age model
(Leonardsen et al., 2022) for downstream tasks such as Alzheimer's disease (AD) classification
or MCI progression prediction.

The difference between ${TANNGUYEN2025_BA_DIR}/model/pyment/CBIG_BA_extract_features.py and
${TANNGUYEN2025_BA_DIR}/model/log_reg/CBIG_BA_extract_features.py is that
${TANNGUYEN2025_BA_DIR}/model/pyment/CBIG_BA_extract_features.py was used directly to
extract the 64D features from the TensorFlow version of the pretrained brain age
model (like in Leonardsen et al. 2022).
However, ${TANNGUYEN2025_BA_DIR}/model/log_reg/CBIG_BA_extract_features.py was used to
extract 64D features from the PyTorch version of our trained models.

Outputs:
- `labels.csv`: contains ID, sex, and age info.
- `processed_id_*.txt`: scan ID list for reference.
- `ID_features.csv` or `ID_bag.csv`: extracted features or brain age predictions.
- `.npy` files: for easy downstream analysis or plotting.

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m model.pyment.CBIG_BA_extract_features --task ad_classification \
        --in_dir /not/real/path --out_dir /not/real/path --feature_extract 1 \
             --model_name brainage64D;
'''

import sys
import os
TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from CBIG_BA_config import global_config
import argparse
from pathlib import Path
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(global_config.PYMENT_DIR)
from pyment.data import AsyncNiftiGenerator, NiftiDataset
from pyment.models import RegressionSFCN
from utils.CBIG_BA_complete_step import generate_complete_flag


def generate_labels(datasets, out_dir, task):
    '''
    Generate `labels.csv` for each dataset with columns: `id`, `sex`, and `age`.

    This file serves as input for the Pyment feature extractor. It also saves a
    `processed_id_*.txt` file listing subject IDs in the correct order for downstream use.

    Args:
        datasets (list of str): Names of datasets to process.
        out_dir (str): Output directory where labels and ID list will be stored.
        task (str): Task type (e.g., "ad_classification" or "mci_progression").
    '''
    for dataset in datasets:
        df_dir = os.path.join(global_config.MATCHED_DATA_DIR, task,
                              '{}_balanced.csv'.format(dataset))
        df = pd.read_csv(df_dir)

        # sort in ascending ID, and reset idx
        df = df.sort_values(by=['RID']).reset_index(drop=True)
        labels_dir = os.path.join(out_dir, dataset)
        if not os.path.exists(labels_dir):
            Path(labels_dir).mkdir(parents=True, exist_ok=True)

        # output list of processed IDs for input for downstream Pyment scripts
        df['PROCESS_ID'].to_csv('{}/processed_id_{}.txt'.format(
            labels_dir, dataset),
                                index=False,
                                header=False)
        df = df[['PROCESS_ID', 'GENDER', 'AGE']]
        df = df.rename(columns={
            'PROCESS_ID': 'id',
            'GENDER': 'sex',
            'AGE': 'age'
        })
        df['sex'] = df['sex'].replace({'Male': 'M', 'Female': 'F'})
        df['age'] = df['age'].astype(float)

        df.to_csv('{}/labels.csv'.format(labels_dir), index=False)


def link_scans(t1_dir, datasets, in_dir, out_dir):
    '''
    Create symbolic links to T1 scans required by Pyment's image loader.

    Each dataset folder will contain an `images/` subfolder with symlinks to preprocessed
    scans named as `<ID>.nii.gz`.

    Args:
        t1_dir (dict): Mapping from dataset name to path of T1 scan directories.
        datasets (list of str): Dataset names to process.
        in_dir (str): Input directory containing `processed_id_*.txt` files.
        out_dir (str): Output directory to store `images/` folders with symlinks.
    '''

    for dataset in datasets:
        processed_id_file = os.path.join(in_dir, dataset,
                                         'processed_id_{}.txt'.format(dataset))
        processed_id_list = []
        # generating scan ID list from scan ID txt file
        processed_id_file = open(processed_id_file, 'r')
        for line in processed_id_file:
            stripped_line = line.strip()
            processed_id_list.append(stripped_line)
        processed_id_file.close()

        img_dir = '{}/{}/images'.format(out_dir, dataset)
        if not os.path.exists(img_dir):
            Path(img_dir).mkdir(parents=True, exist_ok=True)
        # creating symlink for images
        for processed_id in processed_id_list:
            src = os.path.join(t1_dir[dataset], processed_id,
                               'T1_MNI152_1mm_linear_crop.nii.gz')
            dst = '{}/{}.nii.gz'.format(img_dir, processed_id)
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)


def extract_features(feature_extract_flag, dataset, out_dir):
    '''
    Extract 64D latent features or brain age predictions using the Pyment
    pretrained model.

    Outputs are saved as `.csv` and `.npy` files. If extracting brain age, additional
    statistics such as MAE and Pearson's r are computed.

    Args:
        feature_extract_flag (int): 1 for 64D feature extraction, 0 for brain age prediction.
        dataset (str): Name of the dataset to process.
        out_dir (str): Directory containing input data and location to store outputs.

    This function is adapted from `Predict brain age.ipynb` in the
    the Pyment codebase developed by Esten H. Leonardsen and colleagues.

    Original code is part of the study:
    Leonardsen, E. H., Peng, H., Kaufmann, T., Agartz, I., Andreassen, O. A.,
    Celius, E. G., Espeseth, T., Harbo, H. F., Hogestol, E. A., Lange, A. M.,
    Marquand, A. F., Vidal-Pineiro, D., Roe, J. M., Selbaek, G., Sorensen, O.,
    Smith, S. M., Westlye, L. T., Wolfers, T., & Wang, Y. (2022).
    Deep neural networks learn general and clinically relevant representations of
    the ageing brain. NeuroImage, 256, 119210.
    https://doi.org/10.1016/j.neuroimage.2022.119210

    Original source repository: https://github.com/estenhl/pyment-public

    © 2022 Esten H. Leonardsen

    Some functions have been adapted or modified for use in this project.

    This work is provided under the Creative Commons Attribution-NonCommercial 4.0
    International License (CC BY-NC 4.0).
    License: https://creativecommons.org/licenses/by-nc/4.0/

    The original function was from commit hash 00acac5e6d3f53d6a41f9ea7e62e87c00961775d.
    URL:
    https://github.com/estenhl/pyment-public/blob/\
        00acac5e6d3f53d6a41f9ea7e62e87c00961775d/\
            notebooks/Predict%20brain%20age.ipynb#L28
    '''

    dataset_dir = os.path.join(out_dir, dataset)
    if feature_extract_flag == False:
        include_top_flag = True
        log_file = os.path.join(dataset_dir, 'pred_age.log')
    else:
        include_top_flag = False
        log_file = os.path.join(dataset_dir, 'features.log')

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(f'Dataset = {dataset}')

    # create NiftiDataset class from input labels.csv and image folder
    Nifti_dataset = NiftiDataset.from_folder(dataset_dir, target='id')

    # set up preprocessor to normalize voxel values
    preprocessor = lambda x: x / 255

    # instantiate generator
    batch_size = 2
    threads = 4
    generator = AsyncNiftiGenerator(Nifti_dataset,
                                    preprocessor=preprocessor,
                                    batch_size=batch_size,
                                    threads=threads)

    # sanity check for generator
    batches = 0
    for X, y in generator:
        fig, ax = plt.subplots(1, batch_size)
        fig.suptitle(f'Batch {batches+1}')
        for i in range(batch_size):
            ax[i].imshow(X[i, 83, :, :])
            ax[i].axis('off')
            ax[i].set_title(y[i])
        plt.show()
        print(f'Image batch shape: {X.shape}')
        print(
            f'Image voxel value range: {round(np.amin(X), 2)}-{round(np.amax(X), 2)}'
        )
        batches += 1
        if batches > 5:
            break

    # implement regression variant of brain age pre-trained model
    model = RegressionSFCN(weights='brain-age', include_top=include_top_flag)

    Nifti_dataset.target = 'id'
    generator.reset()

    # pre-trained brain age model generates output (outputs) and scan ID (IDs)
    outputs, IDs = model.predict(generator, return_labels=True)
    IDs = IDs.reshape((len(IDs), 1))
    IDs_outputs = np.hstack((IDs, outputs))
    df = pd.DataFrame(IDs_outputs)
    df.rename(columns={df.columns[0]: 'id'}, inplace=True)

    # save output predicted brain age/feature extraction csv and numpy array
    if include_top_flag == True:  # if output brain age
        df.rename(columns={df.columns[1]: 'pred_age'}, inplace=True)
        df_labels = pd.read_csv(os.path.join(dataset_dir, 'labels.csv'))
        df_merged = pd.merge(df_labels, df, on='id')
        df_merged['pred_age'] = df_merged['pred_age'].astype(float)
        df_merged['bag'] = df_merged['pred_age'] - df_merged['age']

        df_merged.to_csv('{}/ID_bag.csv'.format(dataset_dir), index=False)
        np.save('{}/ID_bag'.format(dataset_dir), df_merged)
    else:  # else output 64D features
        df.to_csv('{}/ID_features.csv'.format(dataset_dir), index=False)
        np.save('{}/ID_features'.format(dataset_dir), df)

    # sanity check to ensure output generated for correct number of input scans
    labels_file = os.path.join(dataset_dir, 'labels.csv')
    num_sub = len(pd.read_csv(labels_file))
    assert np.shape(
        outputs
    )[0] == num_sub, 'number of rows in representation matrix not equal to input number of scans'
    logger.info(f'Output generated for {num_sub} scans')

    # if output brain age, generate report on brain age performance on input sample
    if include_top_flag == True:
        generate_brain_age_stats(labels_file, dataset_dir, outputs, logger)
    else:
        pass

    logger.removeHandler(fh)


def generate_brain_age_stats(labels_file, dataset_dir, outputs, logger):
    '''
    Compute and log performance metrics for brain age predictions.

    Also saves Numpy arrays for:
    - True ages (`true_age.npy`)
    - Predicted ages (`pred_ages.npy`)
    - Brain age gap (`brainage_gap.npy`)
    - MAE (`mae.npy`)

    Metrics logged:
    - MAE ± STD
    - R² score
    - Pearson's r and p-value

    Args:
        labels_file (str): Path to the `labels.csv` containing ground-truth ages.
        dataset_dir (str): Directory to store output Numpy arrays and logs.
        outputs (np.ndarray): Predicted brain ages.
        logger (logging.Logger): Logger to print and store performance metrics.
    '''

    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    logger.info(f'\nBrain age prediction statistics report')

    # save true age, predicted age, and brain age gap
    df_labels = pd.read_csv(labels_file)
    true_age = df_labels['age'].to_numpy()
    np.save('{}/true_age.npy'.format(dataset_dir), true_age)
    pred_age = np.reshape(outputs, np.shape(true_age))
    np.save('{}/pred_ages.npy'.format(dataset_dir), pred_age)
    brainage_gap = pred_age - true_age
    np.save('{}/brainage_gap.npy'.format(dataset_dir), brainage_gap)

    # compute MAE
    abs_err = np.absolute(brainage_gap)
    mae = np.mean(abs_err)
    std = np.std(abs_err)
    np.save('{}/mae.npy'.format(dataset_dir), mae)
    logger.info(f'MAE = {mae}±{std}')

    # compute r^2
    r2 = r2_score(true_age, pred_age)
    logger.info(f'r2_score = {r2}')

    # compute Pearson's r statistic and p-value
    r = pearsonr(true_age, pred_age)
    logger.info(f'r_stat = {r[0]}')
    logger.info(f'r_pvalue = {r[1]}')


def get_args():
    '''
    Parse command-line arguments for Pyment feature extraction.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        choices=['ad_classification', 'mci_progression'])
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--feature_extract',
                        type=int,
                        choices=[0, 1],
                        default=1)
    parser.add_argument('--model_name', choices=['brainage64D', 'BAG'])

    return parser.parse_args()


if __name__ == "__main__":
    # Define input arguments
    args = get_args()
    datasets = global_config.DATASETS
    out_dir = args.out_dir
    in_dir = args.in_dir
    t1_dir = global_config.T1_DIRS
    generate_labels(datasets, out_dir, task=args.task)
    link_scans(t1_dir, datasets, in_dir, out_dir)
    feature_extract_flag = args.feature_extract
    print("feature_extract_flag", feature_extract_flag)

    # Extract features for each dataset
    for dataset in datasets:
        extract_features(feature_extract_flag, dataset, out_dir)

    # Generate complete flags for next step to commence
    if args.model_name == 'brainage64D':
        generate_complete_flag(out_dir, 'brainage64D_EF')
    elif args.model_name == 'BAG':
        generate_complete_flag(out_dir, 'BAG_EF')
