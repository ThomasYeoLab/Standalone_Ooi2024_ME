#!/usr/bin/env python
# Perform ensemble prediction for user-custom dataset
# User can use up to 20 models for ensemble
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import json
import pickle
import random
from os import makedirs, path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import global_config
from models.L2C_FNN.dataset import TestDataset
from models.L2C_FNN.evaluation import predict
from models.L2C_FNN.misc import build_pred_frame
from models.L2C_FNN.model_config import OneHot_Config
from models.L2C_FNN.models import NeuralNetwork
from utils.evalOneSubmission import evalOneSub
from utils.evalOneSubmissionIndiMetric import evalOneSubNew


# --- Function to load models from the bundle ---
def load_models_from_bundle(
    bundle_file_path, n_ensemble, input_dim_for_models
):
    """
    Loads model configurations and state_dicts from a bundle file,
    reconstructs the models, and loads their weights.

    Args:
        bundle_file_path (str): Path to the combined bundle file.
        n_ensemble (int): number of models to use for ensemble
        input_dim_for_models (int): The input dimension required by OneHot_Config.
                                   This mirrors the `args.input_dim` from your example.

    Returns:
        list: A list of instantiated and loaded NeuralNetwork models.
              Returns an empty list if loading fails or bundle is empty.
    """
    if not path.exists(bundle_file_path):
        print(f"Error: Bundle file not found at {bundle_file_path}")
        return []

    try:
        # Load the list of dictionaries from the bundle file
        # Use map_location='cpu' if you want to ensure it loads on CPU
        all_models_data = torch.load(
            bundle_file_path, map_location=torch.device("cpu")
        )
    except Exception as e:
        print(f"Error loading the bundle from {bundle_file_path}: {e}")
        return []

    reconstructed_models = []
    print(f"\nLoading models from bundle: {bundle_file_path}")
    print(f"Found {len(all_models_data)} model entries in the bundle.")

    for i in range(n_ensemble):
        model_entry = all_models_data[i]
        # print(f"Reconstructing model {i+1}/{n_ensemble}...")
        config = model_entry["original_config_data"]
        model_state_dict = model_entry["model_state_dict"]

        # Reconstruct hid_dim_list and other parameters from the loaded config,
        # similar to your original loading script.
        layer_num = config["layer_num"]
        hid_dim_list = []
        for l_idx in range(layer_num):
            hid_dim_list.append(config[f"hid_dim_{l_idx}"])

        # Crucially, apply the p_drop scaling
        p_drop_scaled = config["p_drop"] / 10.0

        try:
            # Create the model configuration object
            # Note: input_dim_for_models is passed here, as it was external in your script
            model_specific_config = OneHot_Config(
                layer_num=layer_num,
                input_dim=input_dim_for_models,
                hid_dim_list=hid_dim_list,
                p_drop=p_drop_scaled,
                relu_slope=config["relu_slope"],
            )

            # Instantiate the model
            model = NeuralNetwork(model_specific_config)

            # Load the state dictionary
            model.load_state_dict(model_state_dict)
            model.eval()  # Set to evaluation mode if you're doing inference

            reconstructed_models.append(model)
            # print(f"Successfully reconstructed model {i+1}.")
        except Exception as e:
            print(
                f"Error reconstructing or loading state_dict for model {i+1}: {e}"
            )
            # Optionally, decide if you want to skip this model or stop entirely

    print(
        f"\nSuccessfully loaded {len(reconstructed_models)} models from the bundle."
    )
    return reconstructed_models


def average_ensemble_predictions(list_of_dfs, cols_to_average):
    """
    Computes the average of predictions from a list of pandas DataFrames.

    Args:
        list_of_dfs (list): A list of pandas DataFrames. Each DataFrame
                            should have the same columns and index, representing
                            predictions from different models in an ensemble.
        cols_to_average (list): A list of column names to average on.
    Returns:
        pandas.DataFrame: A DataFrame with the averaged predictions.
                          Non-averaged columns are taken from the first DataFrame.
                          Returns None if the input list is empty.
    """
    if not list_of_dfs:
        print("Input list of DataFrames is empty.")
        return None

    if len(list_of_dfs) == 1:
        print("Only one DataFrame provided, returning it as is.")
        return list_of_dfs[0].copy()

    # Take the first DataFrame as a reference for structure and non-averaged columns
    df_reference = list_of_dfs[0].copy()

    # Check if all DataFrames have the same shape (optional but good practice)
    first_shape = df_reference.shape
    for i, df in enumerate(list_of_dfs[1:], 1):
        if df.shape != first_shape:
            raise ValueError(
                f"DataFrame at index {i} has shape {df.shape}, "
                f"expected {first_shape} (same as the first DataFrame)."
            )

    # --- Averaging Logic ---
    # Create a list of NumPy arrays for the columns to be averaged from each DataFrame
    # This is efficient for large datasets as it avoids repeated DataFrame indexing
    arrays_to_average = [df[cols_to_average].to_numpy() for df in list_of_dfs]

    # Compute the mean of these arrays along axis=0 (row-wise across DataFrames)
    averaged_values = np.mean(arrays_to_average, axis=0)

    # Create the final DataFrame for the averaged results
    # Start with all columns from the reference DataFrame (this preserves non-averaged columns)
    df_averaged = df_reference.copy()

    # Overwrite only the specified columns with their newly averaged values
    df_averaged[cols_to_average] = averaged_values

    return df_averaged


def main(args):
    """
    Main execution function, loading models, data, and perform prediction & evaluation.
    Then save the evaluation results into log files
    """

    # Generate save directories
    old_metric_path = path.join(
        global_config.prediction_dir,
        "L2C_FNN",
        args.site,
        "old_metric",
    )
    new_metric_path = path.join(
        global_config.prediction_dir,
        "L2C_FNN",
        args.site,
        "new_metric",
    )
    makedirs(old_metric_path, exist_ok=True)
    makedirs(new_metric_path, exist_ok=True)

    # Set GPU to use
    if args.device != "None":
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reconstruct ensemble models
    bundle_file_path = path.join(
        global_config.pretrained_dir, "L2C_FNN_weights.pt"
    )
    reconstructed_models = load_models_from_bundle(
        bundle_file_path, args.n_ensemble, args.input_dim
    )

    ref_frame = path.join(
        global_config.clean_gt_path, args.site, "test_gt.csv"
    )
    sub_frame = path.join(
        global_config.fold_gen_path, args.site, "submission_test.csv"
    )

    # Loop through ensemble members to generate predictions
    all_predictions = []
    for fold in range(args.n_ensemble):
        data_path = path.join(
            global_config.processed_frog, args.site, f"fold_{fold}.pkl"
        )
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        test_data = TestDataset(data["test"])
        test_dataloader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        model = reconstructed_models[fold].to(device)
        prediction = predict(model, test_dataloader, data, device)

        # Save predictions into csv file
        csv_path = path.join(old_metric_path, f"prediction_{fold}.csv")
        tadpole = data["test"]["input"]
        build_pred_frame(sub_frame, prediction, tadpole, csv_path)
        all_predictions.append(pd.read_csv(csv_path))

    columns_to_average = [
        "CN relative probability",
        "MCI relative probability",
        "AD relative probability",
        "MMSE",
        "MMSE 50% CI lower",
        "MMSE 50% CI upper",
        "Ventricles_ICV",
        "Ventricles_ICV 50% CI lower",
        "Ventricles_ICV 50% CI upper",
    ]

    avg_csv_path = path.join(old_metric_path, "ensembled_prediction.csv")
    averaged_pred = average_ensemble_predictions(
        all_predictions, columns_to_average
    )
    averaged_pred.to_csv(avg_csv_path, index=False)

    old_result = evalOneSub(pd.read_csv(ref_frame), averaged_pred, args.site)
    new_result, subj_metric, subj_dfs = evalOneSubNew(
        pd.read_csv(ref_frame), averaged_pred, args.site
    )

    old_log_name = path.join(old_metric_path, "out.json")
    with open(old_log_name, "w") as f:
        json.dump(old_result, f)
    new_log_name = path.join(new_metric_path, "out.json")
    with open(new_log_name, "w") as f:
        json.dump(new_result, f)
    subj_metric_name = path.join(new_metric_path, "subj_metric.pkl")
    with open(subj_metric_name, "wb") as fhandler:
        pickle.dump(subj_metric, fhandler)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_ensemble", type=int, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: e.g., cpu, cuda, cuda:0, mps",
    )
    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()

    # set all the seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    main(args)
