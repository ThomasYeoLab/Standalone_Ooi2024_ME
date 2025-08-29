"""
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

Prepare input datasets (AIBL, ADNI, MACC) for downstream analysis by harmonizing column names, 
formatting diagnostic labels and gender codes, and saving unified CSV files for each dataset.

Expected output(s):
1. Cleaned and standardized dataset CSV files (e.g., ADNI.csv, AIBL.csv, MACC.csv)
   saved to RAW_DATA_DIR
2. Scanner model lists for each dataset (e.g., ADNI_scanners.txt)

Example:
    conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR};
    python -m data_split.CBIG_BA_uniform_data
"""
import os
import pandas as pd
import sys

TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from CBIG_BA_config import global_config

if __name__ == "__main__":
    # Output
    OUTPUT_CSV = {
        'AIBL': os.path.join(global_config.RAW_DATA_DIR, 'AIBL.csv'),
        'ADNI': os.path.join(global_config.RAW_DATA_DIR, 'ADNI.csv'),
        'MACC': os.path.join(global_config.RAW_DATA_DIR, 'MACC.csv')
    }
    os.makedirs(global_config.RAW_DATA_DIR, exist_ok=True)
    #assert all([not os.path.exists(csv) for csv in OUTPUT_CSV.values()])

    # General variables: these are hard-coded for each dataset
    id_col = {'AIBL': 'RID', 'ADNI': 'RID', 'MACC': 'subid'}
    process_id = {'AIBL': 'PROCESS_ID', 'ADNI': 'PROCESS_ID', 'MACC': 'scanid'}
    scanner_col = {'AIBL': 'MODEL', 'ADNI': 'MODEL', 'MACC': 'scannermodel'}
    age_col = {'AIBL': 'Age', 'ADNI': 'AGE', 'MACC': 'age'}
    gender_col = {'AIBL': 'Sex', 'ADNI': 'SEX', 'MACC': 'gender'}
    gender_codename = {
        'AIBL': {
            'M': 'Male',
            'F': 'Female'
        },
        'ADNI': {
            'M': 'Male',
            'F': 'Female'
        },
        'MACC': {
            1.0: 'Male',
            2.0: 'Female'
        }
    }

    # Main work, read data, rename columns, save
    for dataset in global_config.DATASETS:
        dataset_df = []
        scanner_models = []

        for task_name in global_config.TASKS:
            if task_name == 'ad_classification':
                dx_col = {
                    'AIBL': 'DXCURREN',
                    'ADNI': 'DX_CLOSEST',
                    'MACC': 'dx'
                }
                class_labels = {
                    'AIBL': [{
                        'name': 'NCI',
                        'code': 1
                    }, {
                        'name': 'AD',
                        'code': 3
                    }],
                    'ADNI': [{
                        'name': 'NCI',
                        'code': 1
                    }, {
                        'name': 'AD',
                        'code': 3
                    }],
                    'MACC': [{
                        'name': 'NCI',
                        'code': 0.0
                    }, {
                        'name': 'AD',
                        'code': 3.0
                    }]
                }
                input_csv = global_config.REPLICATION_ADCLASSICATION

            else:
                dx_col = {k: 'MCI_label' for k in global_config.DATASETS}
                class_labels = {
                    k: [{
                        'name': 'sMCI',
                        'code': 'sMCI'
                    }, {
                        'name': 'pMCI',
                        'code': 'pMCI'
                    }]
                    for k in global_config.DATASETS
                }
                input_csv = global_config.REPLICATION_MCIPROGRESSION

            classes = global_config.TASK_DX[task_name]
            use_columns = [
                id_col[dataset], process_id[dataset], scanner_col[dataset],
                age_col[dataset], gender_col[dataset], dx_col[dataset]
            ]
            input_df = pd.read_csv(input_csv[dataset], usecols=use_columns)

            # Get scanner models
            scanner_models.extend([
                i for i in input_df[scanner_col[dataset]].unique().tolist()
                if i not in scanner_models and not pd.isna(i)
            ])

            # Rename columns
            input_df.rename(columns={
                id_col[dataset]: global_config.ID_COL,
                scanner_col[dataset]: global_config.SCANNER_COL,
                age_col[dataset]: global_config.AGE_COL,
                gender_col[dataset]: global_config.GENDER_COL,
                dx_col[dataset]: global_config.DX_COL,
                process_id[dataset]: global_config.PROCESSID_COL
            },
                            inplace=True)
            # input_df = input_df[['RID', 'PROCESS_ID', 'SCANNERMODEL', 'AGE', 'GENDER', 'DX']]
            input_df = input_df[[
                global_config.ID_COL, global_config.PROCESSID_COL,
                global_config.SCANNER_COL, global_config.AGE_COL,
                global_config.GENDER_COL, global_config.DX_COL
            ]]

            # Rename class labels
            for class_label in class_labels[dataset]:
                input_df[global_config.DX_COL] = input_df[
                    global_config.DX_COL].replace(class_label['code'],
                                                  class_label['name'])
            input_df = input_df[input_df[global_config.DX_COL].isin(
                classes)].copy()

            # Rename gender
            input_df[global_config.GENDER_COL].replace(
                gender_codename[dataset], inplace=True)

            # Corner case 1: For MACC, there is nan in scanner -> remove
            if dataset == 'MACC':
                input_df = input_df[~input_df[global_config.SCANNER_COL].isna(
                )].copy()

            # Corner case 2: For AIBL, there is 1 subject with gender = X -> remove
            if dataset == 'AIBL':
                input_df = input_df[input_df[global_config.GENDER_COL] !=
                                    'X'].copy()

            # Append to dataset_df
            dataset_df.append(input_df)

        dataset_df = pd.concat(dataset_df, ignore_index=True).sort_values(
            by=[global_config.ID_COL]).reset_index(drop=True)
        print('Dataset: {}'.format(dataset.upper()),
              'Shape: {}'.format(dataset_df.shape))
        print(dataset_df.head())
        print('Scanner models: {}'.format(
            dataset_df[global_config.SCANNER_COL].unique()))
        print('Gender: {}'.format(
            dataset_df[global_config.GENDER_COL].unique()))
        print('DX: {}'.format(dataset_df[global_config.DX_COL].unique()))
        print('')

        # Save
        dataset_df.to_csv(OUTPUT_CSV[dataset], index=False)
        with open(
                os.path.join(global_config.RAW_DATA_DIR,
                             '{}_scanners.txt'.format(dataset)), 'w') as f:
            for item in scanner_models:
                f.write(f"{item}\n")
