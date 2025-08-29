'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script uses a trained SFCN or SFCN_FC model to extract encoded features from 
3D brain MRI scans. These features are used for downstream tasks such as 
logistic regression for Alzheimer's disease (AD) classification or MCI progression prediction.

Depending on the model type:
- For SFCN (e.g., in brain age regression), it computes brain age predictions and BAG (brain age gap).
- For SFCN_FC, it extracts intermediate layer features for classification.

The difference between ${TANNGUYEN2025_BA_DIR}/model/pyment/CBIG_BA_extract_features.py and
${TANNGUYEN2025_BA_DIR}/model/log_reg/CBIG_BA_extract_features.py is that
${TANNGUYEN2025_BA_DIR}/model/pyment/CBIG_BA_extract_features.py was used directly to
extract the 64D features from the TensorFlow version of the pretrained brain age
model (like in Leonardsen et al. 2022).
However, ${TANNGUYEN2025_BA_DIR}/model/log_reg/CBIG_BA_extract_features.py was used to
extract 64D features from the PyTorch version of our trained models.

Expected outputs for each split (train, val, test) include:
- Encoded features in `.npy` format (e.g., `*_64d.npy`)
- Subject IDs (`*_ids.txt`)
- Labels (`*_dx.txt`)
- BAG values and actual ages if using BAG_finetune (e.g., `*_bag.npy`, `*_age.txt`)

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m model.log_reg.CBIG_BA_logistic_regression --input_dir /not/real/path \
        --output_dir /not/real/path --model_name direct_ad;
'''

import os
import torch
import argparse
import numpy as np
from model.cnn.modules.CBIG_BA_sfcn import SFCN, SFCN_FC_Intermediate
from model.cnn.modules.CBIG_BA_dataset import ADDataset
from model.utils.CBIG_BA_io import list2txt
from CBIG_BA_config import global_config
import sys

TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from utils.CBIG_BA_complete_step import generate_complete_flag


def extract_feature(model, t1_dirs, label_csv, task, output_dim, device):
    """
    Extract 64D features or predicted brain age
    using a trained model from a dataset.

    Args:
        model (torch.nn.Module): Trained model to use for feature extraction.
        t1_dirs (str): Root directory containing all T1 MRI image paths.
        label_csv (str): CSV file mapping image IDs to labels.
        task (str): Task type, e.g., 'ad_classification' or 'mci_progression'.
        output_dim (int): Expected dimension of the feature vector.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        img_ids (List[str]): List of subject/image IDs.
        features (np.ndarray): Extracted feature vectors.
        ages (List[float]): Subject ages.
        labels (List[int]): Ground truth labels.
    """
    # Define dataset
    dataset = ADDataset(t1_dirs, label_csv, task)
    # print('Number of samples', len(dataset))

    features = np.empty((len(dataset), output_dim))
    img_ids = []
    ages = []
    labels = []
    for i, (img_id, t1_input, age, label) in enumerate(dataset):
        # Read image
        t1_input = t1_input.to(device)
        # print("t1_input at GPU?", t1_input.is_cuda)
        t1_input = t1_input.unsqueeze(0)

        # Extract feature(s)
        with torch.no_grad():
            output = model(t1_input)

        features[i, :] = output.cpu().detach().numpy()
        img_ids.append(img_id)
        ages.append(age)
        labels.append(label)

    return img_ids, features, ages, labels


def main(args):
    """
    Main routine to process multiple data splits, extract features from trained models,
    and save outputs (features, labels, ages, and IDs) for downstream ML models.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    print('Input arguments:', str(args))

    # Predict on all scans in the csv file
    t1_dirs = global_config.T1_DIRS

    trained_model_pt_template = args.trained_model
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Remove data split list out of input argument list
    data_split_list = args.data_split
    del args.data_split

    # Extract other variables
    model_name = args.model_name
    feature_dim = int(args.feature_dim)

    for seed_id in data_split_list:
        print('\nSEED {}'.format(seed_id))
        # Replace placeholder values with actual seed ID
        seed_input_dir = input_dir.replace('SEEDID', str(seed_id))
        seed_output_dir = output_dir.replace('SEEDID', str(seed_id))
        trained_model_pt = trained_model_pt_template.replace(
            'SEEDID', str(seed_id))
        if not os.path.exists(trained_model_pt):
            print(f"{trained_model_pt} not exist, skipping!")
            continue
        print('Trained model:', trained_model_pt)

        os.makedirs(seed_output_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            # Read trained model
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            if model_name == 'BAG_finetune':
                model = SFCN()
            else:
                model = SFCN_FC_Intermediate()
            model.load_state_dict(
                torch.load(trained_model_pt, map_location=device))
            model.to(device)
            model.eval()

            # Read data
            split_csv = os.path.join(seed_input_dir, f'{split}.csv')
            id_txt = os.path.join(seed_output_dir, f'{split}_ids.txt')
            label_txt = os.path.join(seed_output_dir, f'{split}_dx.txt')
            print('SPLIT', {split})
            print('--- Input', split_csv)

            if model_name == 'BAG_finetune':
                # Define output files
                out_npy = os.path.join(seed_output_dir,
                                       f'{split}_{feature_dim}d.npy')
                out_bag_npy = os.path.join(seed_output_dir, f'{split}_bag.npy')
                out_label_npy = os.path.join(seed_output_dir,
                                             f'{split}_dx.npy')
                age_txt = os.path.join(seed_output_dir, f'{split}_age.txt')
                print('--- Output', id_txt, age_txt, label_txt)

                # Extract features
                ids, pred_ages, ages, labels = extract_feature(model, t1_dirs, \
                    split_csv, args.task, feature_dim, device)
                pred_ages = np.squeeze(pred_ages)
                print(split, pred_ages.shape)

                print("shape of pred_ages = {}".format(np.shape(pred_ages)))
                print("shape of ages = {}".format(np.shape(ages)))

                # Compute brain age gap
                bags = pred_ages - ages
                print("shape of bags = {}".format(np.shape(bags)))

                # Save features
                np.save(out_bag_npy, bags, allow_pickle=False)
                np.save(out_label_npy, labels, allow_pickle=False)
                list2txt(ages, age_txt)
            else:
                # Define output file
                out_npy = os.path.join(seed_output_dir,
                                       f'{split}_{feature_dim}d.npy')
                print('--- Output', out_npy, id_txt, label_txt)

                # Extract features
                ids, encoded, _, labels = extract_feature(
                    model, t1_dirs, split_csv, args.task, feature_dim, device)
                print(split, encoded.shape)

                # Save features
                np.save(out_npy, encoded, allow_pickle=False)
            list2txt(ids, id_txt)
            list2txt(labels, label_txt)

            # Clear cache
            del model
            torch.cuda.empty_cache()

        if model_name == 'BAG_finetune':
            # Concatenate train + val into dev split
            dev_bag_npy = os.path.join(seed_output_dir, 'dev_bag.npy')
            dev_label_npy = os.path.join(seed_output_dir, 'dev_dx.npy')

            train_bag_npy = np.load(
                os.path.join(seed_output_dir, 'train_bag.npy'))
            val_bag_npy = np.load(os.path.join(seed_output_dir, 'val_bag.npy'))

            train_label_npy = np.load(
                os.path.join(seed_output_dir, 'train_dx.npy'))
            val_label_npy = np.load(os.path.join(seed_output_dir,
                                                 'val_dx.npy'))

            dev_bag = np.concatenate((train_bag_npy, val_bag_npy), axis=0)
            dev_label = np.concatenate((train_label_npy, val_label_npy),
                                       axis=0)

            np.save(dev_bag_npy, dev_bag, allow_pickle=False)
            np.save(dev_label_npy, dev_label, allow_pickle=False)

            # Generate complete flags for next step to commence
            generate_complete_flag(seed_output_dir,
                                   '{}_bag'.format(model_name))
        else:
            # Generate complete flags for next step to commence
            generate_complete_flag(seed_output_dir,
                                   '{}_pre_logreg'.format(model_name))


def get_args():
    """
    Parses command-line arguments for the feature extraction script.

    Returns:
        argparse.Namespace: Argument object containing input/output paths and configs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split',
                        nargs="+",
                        type=int,
                        help='List of data splits')

    parser.add_argument('--input_dir',
                        help='Directory contains datasplits of multiple seeds')
    parser.add_argument('--output_dir',
                        help='Directory to save extracted features')
    parser.add_argument('--trained_model', help='Path to trained model (.pt)')
    parser.add_argument('--task', type=str, choices=['ad_classification', 'mci_progression'], \
        help='Task name. Used to determine dataset labels and output folder')

    parser.add_argument('--model_type',
                        choices=['SFCN_FC'],
                        type=str,
                        help='Model architecture: SFCN_FC')
    parser.add_argument(
        '--feature_dim',
        help=
        'Dimension of the extracted features, default is 64 for features from the second-to-last layer',
        default=64)
    parser.add_argument(
        '--model_name',
        choices=['direct_ad', 'brainage64D_finetune_ad', 'BAG_finetune'])

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
