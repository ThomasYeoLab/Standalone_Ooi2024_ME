#!/usr/bin/env python
# Process Frog features for easier use during train/val/test
# Pre-processing includes: imputation, normalization, pickling
# Normalization is done through gauss_rank (implemented with code from
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
# https://www.kaggle.com/code/tottenham/10-fold-simple-dnn-with-rank-gauss)

# Configurations:
#     1. GaussRank train, use the mapping to interpolate val/test (achieved through sklearn).
#     2. Perform median imputation before
#     3. Treat CDR as categorical value.
#     4. Perform GaussRank on target.
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import pickle
from os import makedirs, path

import joblib
import numpy as np
import pandas as pd
from scipy.special import erfinv
from sklearn.preprocessing import QuantileTransformer

from config import global_config
from data_processing.misc import categorical_dict, cdr_mapping


def rank_gauss(x):
    """GaussRank transformation with custom code, used for target variables (i.e., MMSE and
       ventricle volumes) only. This transformation has well-defined reverse function for
       us to convert the predicted values back to original scale
       Implemented with code from:
       https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
       https://www.kaggle.com/code/tottenham/10-fold-simple-dnn-with-rank-gauss)
    Args:
        x (array): un-transformed array

    Returns:
        efi_x: transformed array
    """

    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = np.sqrt(2) * erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x


def fit_and_save_normalizer(train_df, numeric_col, seed):
    """
    Computes normalization parameters (medians, QuantileTransformer) from training data,
    transforms the training data, and returns the parameters.

    Args:
        train_df (pd.DataFrame): Training set DataFrame.
        numeric_col (list): Features to perform transformation on.
        seed (int): Random seed for QuantileTransformer.

    Returns:
        np.array: Transformed training data (scaled).
        normalization_parameters: Training statistics to save
    """
    if not isinstance(numeric_col, list):
        numeric_col = list(numeric_col)

    train_num = train_df[numeric_col].copy()

    if train_num.empty:
        raise ValueError(
            "Training data for specified numeric columns is empty. Cannot fit normalizer."
        )

    # 1. Calculate and store medians for imputation
    medians = train_num.median()  # Pandas Series of medians

    # 2. Perform median imputation on the training data copy
    for col in train_num.columns:
        if train_num[col].isna().any():  # Check if any NaNs to impute
            if col in medians and not pd.isna(medians[col]):
                train_num[col].fillna(medians[col], inplace=True)
            else:
                # Handle cases where median might be NaN (e.g., all-NaN column originally)
                # or column not in medians (shouldn't happen if numeric_col is consistent)
                print(
                    f"Warning: Median for training column '{col}' is NaN or missing. NaNs may persist or cause errors."
                )

    # 3. Create and fit QuantileTransformer
    # n_quantiles=train_num.shape[0] matches your original implementation.
    # QuantileTransformer handles cases where n_quantiles > n_samples.
    # Ensure n_quantiles is at least 1.
    n_q = max(1, train_num.shape[0])
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=n_q,
        random_state=seed,
        copy=True,
    )
    train_scaled_array = qt.fit_transform(train_num)

    # 4. Prepare data to save
    normalization_parameters = {
        "medians": medians,
        "quantile_transformer": qt,
        "numeric_columns_used": numeric_col,  # Save the list of columns it was trained on
    }

    return train_scaled_array, normalization_parameters


def load_and_transform_data(new_df, normalization_parameters):
    """
    Use loaded normalization parameters to transform new data.

    Args:
        new_df (pd.DataFrame): New DataFrame to transform (e.g., validation or test set).
        normalization_parameters (dict): Saved normalization parameters.

    Returns:
        new_scaled_array: Transformed test data (scaled).
        (None, None) if loading or transformation fails.
    """

    medians = normalization_parameters["medians"]
    qt = normalization_parameters["quantile_transformer"]
    numeric_col_original = normalization_parameters["numeric_columns_used"]

    # Ensure new_df has the necessary columns
    missing_cols = [
        col for col in numeric_col_original if col not in new_df.columns
    ]
    if missing_cols:
        print(
            "Error: New DataFrame is missing expected numeric columns: ",
            f"{missing_cols}, which were used for training the normalizer."
        )
        raise ValueError(
            f"New DataFrame is missing expected numeric columns: {missing_cols}"
        )

    # Select only the columns that the transformer was trained on, in the correct order.
    new_num = new_df[numeric_col_original].copy()

    # 2. Perform median imputation using loaded medians
    for col in new_num.columns:
        if new_num[col].isna().any():  # Check if any NaNs to impute
            if col in medians and not pd.isna(medians[col]):
                new_num[col].fillna(medians[col], inplace=True)
            else:
                # If a column was all NaN in training, its median might be NaN.
                # Or, if a new column appears that wasn't in training.
                print(
                    f"Warning: Median for column '{col}' is NaN or not found in loaded medians."
                    "NaNs in this column might remain or cause errors during transform."
                )
                # Optionally, impute with a global default like 0 if median is NaN:
                # if pd.isna(medians.get(col)): new_num[col].fillna(0, inplace=True)

    # 3. Transform data using loaded QuantileTransformer
    try:
        new_scaled_array = qt.transform(new_num)
    except ValueError as e:
        print(f"Error during transforming data: {e}")
        print(
            "This might be due to an unexpected number of features or NaN values that couldn't be imputed."
        )
        return None

    return new_scaled_array


def gen_input(num_df, full_df, cat_dict):
    """Combine GaussRank transformed numeric features with one-hot encoded
       categorical features into a single matrix as final input data

    Args:
        num_df (dataframe): GaussRank transformed numeric features dataframe
        full_df (dataframe): raw L2C dataframe containing all features
        cat_dict (dict): dictionary indicating categorical features and class counts
    Returns:
        input_mat (array): combined data matrix
    """

    rid = np.expand_dims(full_df["rid"].values, axis=1)

    # convert CDR to proper label order
    full_df.replace(
        {
            "mr_CDR": cdr_mapping,
            "high_CDR": cdr_mapping,
            "low_CDR": cdr_mapping,
        },
        inplace=True,
    )
    # first convert NaN in categorical variables into Unknown class
    for cat, n in cat_dict.items():
        full_df[cat] = full_df[cat].fillna(n).astype(int)
    cat_col = list(cat_dict.keys())
    full_df[cat_col] = full_df[cat_col].astype(int)

    # then convert categorical to one-hot
    one_hot_list = []
    for cat, n in cat_dict.items():
        one_hot_list.append(
            np.eye(n + 1)[full_df[cat].values]
        )  # we add unknown class for all
    one_hot_mat = np.hstack(one_hot_list)

    # merge one_hot with numeric features
    input_mat = np.hstack([one_hot_mat, num_df, rid])
    return input_mat


def gen_data(f, args, normalization_dict=None):
    """Wrapper function controlling the entire data generation process, including:
    input data, ground truth, mask, train data statistics, and target variable reverse
    """

    root_path = global_config.frog_path

    if args.use_adas:
        curr_cog = "curr_adas13"
        cog_std = "ADASstd"
        cog_mean = "ADASmean"
    else:
        curr_cog = "curr_mmse"
        cog_std = "MMSEstd"
        cog_mean = "MMSEmean"

    # columns in L2C feature table that shouldn't be included in normalization
    not_predictors = [
        "ptid",
        "rid",
        "curr_dx",
        curr_cog,
        "curr_ventricles",
        "curr_icv",
        "mr_change_CDR",
        "time_since_milder",
    ]

    # categorical features are not included in normalization as well
    # (only numerical ones are included)
    categorical_col = list(categorical_dict.keys())
    not_predictors += categorical_col

    ret = {}

    if args.independent_test:
        # entire dataset as test input data (always the same input)
        test_path = path.join(root_path, f"{args.site}/test.csv")
        test_df = pd.read_csv(test_path)

        # Test sites have normalization_dict loaded
        test_num = load_and_transform_data(
            test_df.copy(), normalization_dict  # Pass a copy
        )

        test_input = gen_input(test_num, test_df, categorical_dict)
        ret["test"] = {"input": test_input}
        ret["df_dict"] = normalization_dict["df_dict"]

        return ret
    else:
        train_path = path.join(
            root_path, f"{args.site}/fold{f}_train.csv"
        )  # args.site: train site (ADNI)
        val_path = path.join(
            root_path, f"{args.site}/fold{f}_val.csv"
        )  # args.site: train site (ADNI)
        # 20 test folds are different (ADNI train 20 splits)
        test_path = path.join(root_path, f"{args.site}/fold{f}_test.csv")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        numeric_col = [
            col for col in train_df.columns if col not in not_predictors
        ]

        train_num, normalization_parameters = fit_and_save_normalizer(
            train_df.copy(),  # Pass a copy if train_df will be used later unmodified
            numeric_col,
            args.seed,
        )

        val_num = load_and_transform_data(
            val_df.copy(), normalization_parameters  # Pass a copy
        )

        test_num = load_and_transform_data(
            test_df.copy(), normalization_parameters  # Pass a copy
        )

        # combine one-hot encoded categorical features with normalized numerical features
        train_input = gen_input(train_num, train_df, categorical_dict)
        val_input = gen_input(val_num, val_df, categorical_dict)
        test_input = gen_input(test_num, test_df, categorical_dict)

        # save mean and std of curr_mmse, curr_vent/curr_icv (to balance hyperparameter tuning)
        ret[cog_std] = train_df[curr_cog].std()
        ret[cog_mean] = train_df[curr_cog].mean()
        ret["VentICVstd"] = (
            train_df["curr_ventricles"] / train_df["curr_icv"]
        ).std()
        ret["VentICVmean"] = (
            train_df["curr_ventricles"] / train_df["curr_icv"]
        ).mean()

        # save normalized train set y
        all_mmse = train_df[curr_cog].values
        src_mmse = all_mmse[~np.isnan(all_mmse)]
        all_vent = (train_df["curr_ventricles"] / train_df["curr_icv"]).values
        src_vent = all_vent[~np.isnan(all_vent)]

        tgt_mmse = rank_gauss(src_mmse)
        tgt_vent = rank_gauss(src_vent)
        all_mmse[~np.isnan(all_mmse)] = tgt_mmse
        all_vent[~np.isnan(all_vent)] = tgt_vent

        train_dx = train_df["curr_dx"].values
        train_out = np.vstack([train_dx, all_mmse, all_vent]).T
        mask = ~np.isnan(train_out)

        # save mapping_df for inverse_transform later
        df_mmse = {"src": src_mmse, "tgt": tgt_mmse}
        df_vent = {"src": src_vent, "tgt": tgt_vent}

        # save everything into ret
        ret["train"] = {"truth": train_out, "input": train_input, "mask": mask}
        ret["val"] = {"input": val_input}
        ret["test"] = {"input": test_input}
        ret["df_dict"] = {"mmse": df_mmse, "vent": df_vent}

        return ret, normalization_parameters


def main(args):
    """Main execution function"""

    # Set random seed
    np.random.seed(args.seed)

    fold = args.fold

    # Generate the save directory
    save_path = path.join(global_config.processed_frog, f"{args.site}")
    makedirs(save_path, exist_ok=True)

    # Define normalizer file path
    norm_path = path.join(global_config.processed_frog, f"{args.train_site}")
    makedirs(norm_path, exist_ok=True)
    norm_param_file = path.join(norm_path, f"norm_param_{fold}.joblib")

    if args.independent_test:
        normalization_dict = joblib.load(norm_param_file)
        ret = gen_data(fold, args, normalization_dict)
    else:
        ret, normalization_dict = gen_data(fold, args)
        normalization_dict["df_dict"] = ret["df_dict"]
        joblib.dump(normalization_dict, norm_param_file)

    # Save imputed and transformed data
    save_file = path.join(save_path, f"fold_{fold}.pkl")
    with open(save_file, "wb") as fhandler:
        pickle.dump(ret, fhandler)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--train_site", required=True)
    parser.add_argument("--use_adas", action="store_true")
    parser.add_argument("--independent_test", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
