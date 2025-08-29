"""
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

Script to balance diagnostic class distributions (NCI vs AD or sMCI vs pMCI) across scanner models 
for multiple datasets. For each dataset and scanner model, it subsamples the larger class to match 
the size of the smaller one, while minimizing age and sex differences between matched samples.

Expected output(s):
1. Balanced CSV files (e.g., ADNI_balanced.csv, MACC_balanced.csv, AIBL_balanced.csv)
2. Excess CSV files with removed subjects (e.g., ADNI_excess.csv, etc.)
3. Log file (matching.log) summarizing class counts and matching process
4. Histogram plot (matched_histplot.png) showing age distributions of matched classes

Example:
    conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR};
    python -m data_split.CBIG_BA_matching --task ad_classification
"""

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, chisquare
import sys
TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError("ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)

from CBIG_BA_config import global_config


def find_best_match(group1, group2, id_col, age_col, gender_col, gender_penalty):
    """
    Match each row in group1 to the most similar row in group2 based on age and gender.
    gender_penalty = 1 means cost = 0 for same gender, cost = 1 for different gender.

    Parameters:
        group1 (pd.DataFrame): The smaller group to match from.
        group2 (pd.DataFrame): The larger group to match into.
        id_col (str): Column name for subject ID.
        age_col (str): Column name for age.
        gender_col (str): Column name for gender (values: 'Male', 'Female').
        gender_penalty (float): Penalty score for gender mismatch (1 = full penalty).

    Returns:
        group2_matched (pd.DataFrame): Rows from group2 best matched to group1.
        group2_excess (pd.DataFrame): Remaining unmatched rows from group2.
    """

    # Check that group2 has more samples than group1
    assert group1.shape[0] < group2.shape[0]

    group2_copy = group2.copy()
    best_matched_ids = []
    for _, row in group1.iterrows():
        age = row[age_col]
        gender = row[gender_col]

        group2_copy['same_gender'] = (1 - (group2_copy[gender_col] == gender).astype(int)) * gender_penalty
        group2_copy['age_diff'] = abs(group2_copy[age_col] - age)
        group2_copy.sort_values(by=['same_gender', 'age_diff'], inplace=True)

        best_matched_ids.append(group2_copy[id_col].values[0])
        group2_copy.drop(index=group2_copy.index[0], axis=0, inplace=True)
        group2_copy.drop(columns=['same_gender', 'age_diff'], inplace=True)

    group2_matched = group2.loc[group2[id_col].isin(best_matched_ids), :].copy()
    group2_excess = group2.loc[~group2[id_col].isin(best_matched_ids), :].copy()

    return group2_matched, group2_excess


def remove_imbalanced_class(group1,
                            group2,
                            id_col,
                            age_col,
                            gender_col,
                            gender_penalty=1,
                            log_file=None):
    """
    Balance two diagnostic groups by subsampling the larger group to match the smaller one
    while minimizing demographic mismatch (age, gender).

    Parameters:
        group1 (pd.DataFrame): Data for class 1.
        group2 (pd.DataFrame): Data for class 2.
        id_col (str): Column name for subject ID.
        age_col (str): Column name for age.
        gender_col (str): Column name for gender (values: 'Male', 'Female').
        gender_penalty (float): Penalty applied for gender mismatch during matching.
        log_file (file-like): Optional log file to write status messages.

    Returns:
        group1_dict (dict): {'balanced': kept samples, 'excess': dropped samples} for group1.
        group2_dict (dict): {'balanced': kept samples, 'excess': dropped samples} for group2.
    """
    swap = False

    if group1.shape[0] == group2.shape[0] and group1.shape[0] != 0:
        group1_balanced, group1_excess = group1, None
        group2_balanced, group2_excess = group2, None
        print('--- This model already have equal sample for each DX class.',
              file=log_file)

    elif group1.shape[0] == group2.shape[0] and group1.shape[0] == 0:
        group1_balanced, group1_excess = None, None
        group2_balanced, group2_excess = None, None
        print('--- Both groups have 0 scan.', file=log_file)

    elif group1.shape[0] == 0 and group2.shape[0] > 0:
        group1_balanced, group1_excess = None, None
        group2_balanced, group2_excess = None, group2
        print('--- Group 1 has 0 scan.', file=log_file)

    elif group2.shape[0] == 0 and group1.shape[0] > 0:
        group1_balanced, group1_excess = None, group1
        group2_balanced, group2_excess = None, None
        print('--- Group 2 has 0 scan.', file=log_file)

    else:
        assert group1.shape[0] != group2.shape[0]

        # Set group1 = group with smaller sample size
        if group1.shape[0] > group2.shape[0]:
            group1, group2 = group2, group1
            swap = True

        # Save both balanced and excess set
        group1.sort_values(by=[id_col], ignore_index=True, inplace=True)
        group2.sort_values(by=[id_col], ignore_index=True, inplace=True)

        group1_balanced, group1_excess = group1.copy(), pd.DataFrame()
        group2_balanced, group2_excess = find_best_match(
            group1, group2, id_col, age_col, gender_col, gender_penalty)

        assert (group1_balanced.shape[0] +
                group1_excess.shape[0]) == group1.shape[0]
        assert (group2_balanced.shape[0] +
                group2_excess.shape[0]) == group2.shape[0]
        assert group1_balanced.shape[0] == group1_balanced.shape[0], \
            "#{} != #{}".format('group1 balanced', 'group2 balanced')

    # Save balanced and excess dataframes for the current scanner model
    group1_dict = {'balanced': group1_balanced, 'excess': group1_excess}
    group2_dict = {'balanced': group2_balanced, 'excess': group2_excess}

    if swap:
        group1_dict, group2_dict = group2_dict, group1_dict

    return group1_dict, group2_dict


def main(args):
    """
    Main function for task-specific class balancing.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.

    Outputs:
        - Balanced and excess CSVs for each dataset.
        - Log file with matching details and statistical results.
        - Histogram plot of age distribution after matching.
    """
    task = args.task
    output_dir = os.path.join(global_config.MATCHED_DATA_DIR, task)
    output_csv = {(k, v): os.path.join(output_dir, f'{k}_{v}.csv') \
        for k in global_config.DATASETS for v in ['balanced', 'excess']}
    os.makedirs(output_dir, exist_ok=True)
    #assert all([not os.path.exists(v)
                #for v in output_csv.values()]), 'Output csv already exists.'
    # Log file
    log_file = os.path.join(output_dir, 'matching.log')
    f = open(log_file, 'w')

    # General variables
    id_col = global_config.ID_COL
    scanner_col = global_config.SCANNER_COL
    age_col = global_config.AGE_COL
    gender_col = global_config.GENDER_COL
    label_col = global_config.DX_COL

    raw_csv = {
        k: os.path.join(global_config.RAW_DATA_DIR, f'{k}.csv')
        for k in global_config.DATASETS
    }
    class_labels = global_config.TASK_DX[task]
    print('Dropping for {}'.format(task).upper(), file=f)

    input_csv = {}
    for k, v in raw_csv.items():
        tmp = pd.read_csv(v)
        input_csv[k] = tmp.loc[
            tmp[label_col].isin(class_labels), :].copy().reset_index(drop=True)

    # Read scanner list
    scanner_models = {}
    for dataset in global_config.DATASETS:
        scanner_models[dataset] = pd.read_csv(os.path.join(
            global_config.RAW_DATA_DIR, f'{dataset}_scanners.txt'),
                                              header=None)[0].to_list()

    # Check if values follows the assumped format
    for dataset in global_config.DATASETS:
        df = input_csv[dataset]
        assert df[label_col].isin(class_labels).all(
        ), "Some subjects in {} do not have the class labels corresponding to the task.".format(
            dataset)
        assert df[scanner_col].isna().sum(
        ) == 0, "Some subjects in {} do not have scanner information.".format(
            dataset)
        assert df[age_col].isna().sum(
        ) == 0, "Some subjects in {} do not have age information.".format(
            dataset)
        assert df[gender_col].isin(['Male', 'Female']).all(
        ), "Some subjects in {} have wrong gender value (only accept Male/Female)".format(
            dataset)

    # Iterate through each dataset
    g1_all_age, g2_all_age = [], []
    g1_all_sex, g2_all_sex = [], []
    for dataset in global_config.DATASETS:
        dataset_balanced, dataset_excess = [], []
        print('{}{}{}'.format('=' * 20, dataset.upper(), '=' * 20), file=f)

        # Read input and only take subjects with the class labels corresponding to the task
        df = input_csv[dataset]
        df = df.loc[df[label_col].isin(class_labels), :].copy()
        df = df.sort_values(by=[id_col]).reset_index(drop=True)

        # Check for missing scanner
        nan_scanner = df[scanner_col].isna().sum()
        assert nan_scanner == 0, 'Missing scanner information for {} scans'.format(
            nan_scanner)

        # Check if RID unique
        assert df[id_col].unique().shape[0] == df[id_col].shape[0]

        # Iterate through each scanner model
        for model in scanner_models[dataset]:
            group1 = df.loc[(df[scanner_col] == model) &
                            (df[label_col] == class_labels[0]), :].copy()
            group2 = df.loc[(df[scanner_col] == model) &
                            (df[label_col] == class_labels[1]), :].copy()

            print('{}. {} {}, {} {}'.format(model, class_labels[0],
                                            group1.shape[0], class_labels[1],
                                            group2.shape[0]),
                  file=f)
            g1_result, g2_result = remove_imbalanced_class(group1,
                                                           group2,
                                                           id_col,
                                                           age_col,
                                                           gender_col,
                                                           log_file=f)

            dataset_balanced.extend(
                [g1_result['balanced'], g2_result['balanced']])
            dataset_excess.extend([g1_result['excess'], g2_result['excess']])

        dataset_balanced = pd.concat(dataset_balanced, ignore_index=True)
        dataset_excess = pd.concat(dataset_excess, ignore_index=True)

        # Save
        dataset_balanced.to_csv(output_csv[(dataset, 'balanced')], index=False)
        dataset_excess.to_csv(output_csv[(dataset, 'excess')], index=False)

        # Get age for t-test
        g1_all_age.extend(dataset_balanced.loc[
            dataset_balanced[label_col] == class_labels[0], age_col].tolist())
        g2_all_age.extend(dataset_balanced.loc[
            dataset_balanced[label_col] == class_labels[1], age_col].tolist())
        # Get sex for chisquare
        g1_all_sex.extend(dataset_balanced.loc[dataset_balanced[label_col] ==
                                               class_labels[0],
                                               gender_col].tolist())
        g2_all_sex.extend(dataset_balanced.loc[dataset_balanced[label_col] ==
                                               class_labels[1],
                                               gender_col].tolist())

    g1_all_sex = [1 if item == 'Male' else 0 for item in g1_all_sex]
    g2_all_sex = [1 if item == 'Male' else 0 for item in g2_all_sex]

    # Compute stat
    group1_male_ratio = sum(g1_all_sex) / len(g1_all_sex)
    group1_sex_count = [len(g1_all_sex) - sum(g1_all_sex), sum(g1_all_sex)]
    group1_sex_freq = [1 - group1_male_ratio, group1_male_ratio]

    group2_male_ratio = sum(g2_all_sex) / len(g2_all_sex)
    group2_sex_count = [len(g2_all_sex) - sum(g2_all_sex), sum(g2_all_sex)]
    group2_sex_freq = [1 - group2_male_ratio, group2_male_ratio]

    print('\n', file=f)
    print('SEX', file=f)
    print('---',
          class_labels[0],
          'female:male',
          group1_sex_freq,
          '#female:#male',
          group1_sex_count,
          file=f)
    print('---',
          class_labels[1],
          'female:male',
          group2_sex_freq,
          '#female:#male',
          group2_sex_count,
          file=f)
    print('---', chisquare(group1_sex_freq, group2_sex_freq), file=f)
    print('AGE', file=f)
    print('---',
          class_labels[0],
          np.mean(g1_all_age),
          np.std(g1_all_age),
          file=f)
    print('---',
          class_labels[1],
          np.mean(g2_all_age),
          np.std(g2_all_age),
          file=f)
    print('---', ttest_rel(g1_all_age, g2_all_age), file=f)

    assert len(g2_all_sex) == len(g1_all_age) and len(g2_all_sex) == len(
        g2_all_age)

    # Plotting
    ages = pd.DataFrame({
        'age':
        g1_all_age + g2_all_age,
        'dx': [class_labels[0]] * len(g1_all_age) +
        [class_labels[1]] * len(g2_all_age)
    })
    ages = ages[['dx', 'age']]

    sns.histplot(ages, x='age', hue='dx', discrete=True, kde=True)
    plt.savefig(os.path.join(output_dir, 'matched_histplot.png'))
    plt.clf()

    # Close log file if opened
    if f is not None:
        f.close()


def get_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Contains selected task (ad_classification or mci_progression).
    """
    args = argparse.ArgumentParser()
    args.add_argument('--task', type=str, choices=['ad_classification', 'mci_progression'])

    return args.parse_args()


if __name__ == "__main__":
    main(get_args())
