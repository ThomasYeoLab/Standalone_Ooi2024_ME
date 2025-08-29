'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script comprises global variables used across TANNGUYEN2025_BA
stable project.

Example (in Python script itself that requires these config variables):
    from CBIG_BA_config import global_config
    REPLICATION_DATA = global_config.REPLICATION_DATA
'''

import os

CBIG_REPDATA_DIR = os.getenv('CBIG_REPDATA_DIR')
if not CBIG_REPDATA_DIR:
    raise ValueError("ERROR: CBIG_REPDATA_DIR environment variable not set.")
CBIG_CODE_DIR = os.getenv('CBIG_CODE_DIR')
if not CBIG_CODE_DIR:
    raise ValueError("ERROR: CBIG_CODE_DIR environment variable not set.")


class global_config:
    # Input directories
    PYMENT_DIR = os.path.join(CBIG_CODE_DIR, 'stable_projects',
                              'predict_phenotypes', 'TanNguyen2025_BA',
                              'pyment-public')
    REPLICATION_DATA = os.path.join(CBIG_REPDATA_DIR, 'stable_projects',
                                    'predict_phenotypes', 'TanNguyen2025_BA')
    T1_DIRS = {
        'AIBL': '/mnt/isilon/CSC2/Yeolab/Data/AIBL/process/pyment',
        'ADNI': '/mnt/nas/CSC7/Yeolab/Data/ADNI/process/pyment',
        'MACC': '/mnt/isilon/CSC2/Yeolab/Data/MACC/process/T1_pyment/Baseline'
    }
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    MATCHED_DATA_DIR = os.path.join(DATA_DIR,
                                    'matched')  # Matched data directory
    DATA_SPLIT_DIR = os.path.join(
        DATA_DIR,
        'data_split')  # Data split combined across datasets directory

    # Input files
    REPLICATION_ADCLASSICATION = {
        'AIBL':
        os.path.join(REPLICATION_DATA, 'data', 'ad_classification',
                     'AIBL.csv'),
        'ADNI':
        os.path.join(REPLICATION_DATA, 'data', 'ad_classification',
                     'ADNI.csv'),
        'MACC':
        os.path.join(REPLICATION_DATA, 'data', 'ad_classification', 'MACC.csv')
    }
    REPLICATION_MCIPROGRESSION = {
        'AIBL':
        os.path.join(REPLICATION_DATA, 'data', 'mci_progression', 'AIBL.csv'),
        'ADNI':
        os.path.join(REPLICATION_DATA, 'data', 'mci_progression', 'ADNI.csv'),
        'MACC':
        os.path.join(REPLICATION_DATA, 'data', 'mci_progression', 'MACC.csv')
    }

    # Dataset order
    DATASETS = ['AIBL', 'ADNI', 'MACC']  # DO NOT CHANGE THE ORDER

    # Classification tasks
    TASKS = ['ad_classification', 'mci_progression']

    # Corresponding diagnoses
    TASK_DX = {
        'ad_classification': ['NCI', 'AD'],
        'mci_progression': ['sMCI', 'pMCI']
    }

    # Shared columns across datasets
    ID_COL = 'RID'
    SCANNER_COL = 'SCANNERMODEL'
    AGE_COL = 'AGE'
    GENDER_COL = 'GENDER'
    DX_COL = 'DX'
    PROCESSID_COL = 'PROCESS_ID'

    # Seed
    SEED = 1234
