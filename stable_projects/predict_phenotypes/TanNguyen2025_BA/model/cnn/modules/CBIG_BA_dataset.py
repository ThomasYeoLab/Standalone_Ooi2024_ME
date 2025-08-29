'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script defines a PyTorch Dataset class for loading and preprocessing 3D T1-weighted MRI scans
for Alzheimer's Disease classification tasks.

Supported tasks:
1. AD classification: Binary classification between Normal Cognitive Impairment (NCI) and Alzheimer's Disease (AD).
2. MCI progression prediction: Binary classification between stable MCI (sMCI) and progressive MCI (pMCI).

The dataset assumes the T1 images have been Pyment-preprocessed and are stored in `.nii.gz` format
with shape (167, 212, 160).
'''

import os
import torch
import pandas as pd
import nibabel as nib
from torch.utils.data.dataset import Dataset


class ADDataset(Dataset):
    """
    PyTorch Dataset class for loading and preprocessing T1-weighted MRI scans
    for Alzheimer's disease-related classification tasks.

    This class reads image paths and diagnostic labels from a CSV file and
    loads the corresponding 3D MRI volume from `.nii.gz` files. It supports
    two binary classification tasks: AD classification and MCI progression.

    Args:
        t1_dirs (dict): Dictionary mapping dataset names (e.g., 'ADNI') to directories containing T1 images.
        label_csv (str): Path to CSV file containing 'DATASET', 'PROCESS_ID', 'AGE', and 'DX' columns.
        task (str): Either 'ad_classification' or 'mci_progression'. Determines label encoding.
    """

    def __init__(self, t1_dirs, label_csv, task):
        """
        Initialize the dataset with directories, label CSV, and task type.

        Args:
            t1_dirs (dict): Mapping of dataset names to T1 image directory paths.
            label_csv (str): CSV file containing metadata including diagnosis and image IDs.
            task (str): Task type; must be either 'ad_classification' or 'mci_progression'.
        """
        self.t1_dirs = t1_dirs
        self.label_df = pd.read_csv(
            label_csv, usecols=['DATASET', 'PROCESS_ID', 'AGE', 'DX'])
        self.task = task
        assert self.task in [
            'ad_classification', 'mci_progression'
        ], 'task must be ad_classification or mci_progression'

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of rows in the label CSV.
        """
        return len(self.label_df)

    def __getitem__(self, idx):
        """
        Retrieve and process the T1 MRI scan and metadata for a given index.

        Loads the corresponding `.nii.gz` image, normalizes the voxel values,
        converts to PyTorch tensor, and maps the diagnosis label to binary.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (img_id, 3D volume tensor, age, binary diagnosis label)
        """
        dataset, img_id, age, dx = self.label_df.iloc[idx]
        assert dataset in [
            'AIBL', 'ADNI', 'MACC'
        ], f"Only accept dataset of AIBL, ADNI, MACC. Error at {img_id}"

        # Determine T1 folder based on dataset
        nii_path = os.path.join(self.t1_dirs[dataset], img_id,
                                'T1_MNI152_1mm_linear_crop.nii.gz')

        # Read T1
        data = nib.load(nii_path)
        data = data.get_fdata()
        assert data.shape == (167, 212, 160)

        # Normalization
        data = data / 255

        # Move T1 to tensor and add 1 dimension
        data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(0)

        # Read diagnosis and convert to 0 or 1 based on the task
        if self.task == 'ad_classification':
            assert dx in [
                'NCI', 'AD'
            ], f"Only accept dx labels of NCI/AD. Error at {img_id}"
            dx = 0 if dx == 'NCI' else 1

        else:  # self.task == 'smci_pmci'
            assert dx in [
                'sMCI', 'pMCI'
            ], f"Only accept dx labels of sMCI/pMCI. Error at {img_id}"
            dx = 0 if dx == 'sMCI' else 1

        return img_id, data, age, dx
