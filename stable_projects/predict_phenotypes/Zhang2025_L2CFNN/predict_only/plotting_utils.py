#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Chen Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import global_config

# --- Configuration ---
# Dictionaries mapping target name to column name
gt_col_dict = {"MMSE": "MMSE", "VENT": "Ventricles", "DX": "DX"}

inp_col_dict = {"MMSE": "MMSE", "VENT": "Ventricles_ICV", "DX": "DX"}

y_label_dict = {
    "MMSE": "MMSE Score",
    "VENT": "Ventricle volume",
    "DX": "Clinical Diagnosis",
}

# Mapping for DX labels (used for plotting y-axis)
# This should be consistent with dx_map in main()
DX_PLOT_LABELS = {0: "CN", 1: "MCI", 2: "DEM"}


# --- Helper Functions ---
def plot_individual_trajectory(
    input_half, gtDf, forecasrDf, target, save_path, user_total_count=None
):
    rids = input_half.RID.unique()
    inp_col_name = inp_col_dict[target]
    gt_col_name = gt_col_dict[target]

    num_available_subjects = len(rids)
    num_to_plot = num_available_subjects

    if user_total_count is not None:
        try:
            requested_count = int(user_total_count)
            if 0 < requested_count < num_available_subjects:
                num_to_plot = requested_count
            elif requested_count <= 0:
                print(
                    f"Info: --total_count ({user_total_count}) is not positive. Plotting all subjects."
                )
        except ValueError:
            print(
                f"Warning: Invalid --total_count value '{user_total_count}'. Plotting all subjects."
            )

    if (
        num_to_plot == 0 and num_available_subjects > 0
    ):  # Handles case where requested_count might be 0 but data exists
        print(
            "Info: Number of plots to generate is 0 based on counts. Defaulting to plot 1 subject if available."
        )
        num_to_plot = 1

    print(
        f"Attempting to plot trajectories for {num_to_plot} subjects for target '{target}'."
    )

    for i in range(num_to_plot):
        if i >= len(
            rids
        ):  # Safety break if num_to_plot was set unexpectedly high
            break
        subject = rids[i]
        input_half_sub = input_half[input_half["RID"] == subject].copy()
        gtDf_sub = gtDf[gtDf["RID"] == subject].copy()
        forecasrDf_sub = forecasrDf[forecasrDf["RID"] == subject].copy()

        # Convert dates to datetime
        input_half_sub["EXAMDATE"] = pd.to_datetime(input_half_sub["EXAMDATE"])
        gtDf_sub["CognitiveAssessmentDate"] = pd.to_datetime(
            gtDf_sub["CognitiveAssessmentDate"]
        )
        forecasrDf_sub["Forecast Date"] = pd.to_datetime(
            forecasrDf_sub["Forecast Date"]
        )

        # Sort all datasets by date
        input_half_sub.sort_values(by="EXAMDATE", inplace=True)
        gtDf_sub.sort_values(by="CognitiveAssessmentDate", inplace=True)
        forecasrDf_sub.sort_values(by="Forecast Date", inplace=True)

        # Compute baseline month (Month 0)
        baseline_date = input_half_sub["EXAMDATE"].min()

        # Convert dates to months (relative to baseline)
        input_half_sub["Months"] = (
            input_half_sub["EXAMDATE"] - baseline_date
        ).dt.total_seconds() / (60 * 60 * 24 * 30)
        input_half_sub["Months"] = input_half_sub["Months"].astype(int)

        gtDf_sub["Months"] = (
            gtDf_sub["CognitiveAssessmentDate"] - baseline_date
        ).dt.total_seconds() / (60 * 60 * 24 * 30)
        gtDf_sub["Months"] = gtDf_sub["Months"].astype(int)

        forecasrDf_sub["Months"] = (
            forecasrDf_sub["Forecast Date"] - baseline_date
        ).dt.total_seconds() / (60 * 60 * 24 * 30)
        forecasrDf_sub["Months"] = forecasrDf_sub["Months"].astype(
            int
        )  # Convert to integer months

        # Initialize seaborn style
        sns.set_theme(style="whitegrid")

        plt.figure(figsize=(6, 3.5))

        # Plot input_half (Ground Truth - First 50%)
        sns.scatterplot(
            x=input_half_sub["Months"],
            y=input_half_sub[inp_col_name],
            color="black",
            marker="o",
            s=100,
            edgecolor="white",
            label="Observed (Input)",
        )

        # Plot gtDf (Ground Truth - Second 50%)
        sns.scatterplot(
            x=gtDf_sub["Months"],
            y=gtDf_sub[gt_col_name],
            color="blue",
            marker="s",
            s=100,
            edgecolor="white",
            label="Observed (Future GT)",
        )

        # Plot DNN Predictions
        sns.lineplot(
            x=forecasrDf_sub["Months"],
            y=forecasrDf_sub[inp_col_name],
            color="red",
            linestyle="dashed",
            label="Prediction",
        )

        # Labels and Title
        plt.xlabel("Months Since Baseline")
        plt.ylabel(y_label_dict[target])
        plt.title(f"{target} Progression for Subject {subject}")

        # Customize Y-axis for DX plots
        if target == "DX":
            y_ticks_values = sorted(DX_PLOT_LABELS.keys())
            y_ticks_labels = [DX_PLOT_LABELS[val] for val in y_ticks_values]
            plt.yticks(y_ticks_values, y_ticks_labels)
            # Ensure y-axis limits accommodate these discrete labels nicely
            plt.ylim(-0.5, len(DX_PLOT_LABELS) - 0.5)

        plt.legend()
        plot_filename = save_path / f"{subject}.jpg"
        plt.savefig(plot_filename, dpi=600, bbox_inches="tight")
        plt.close()

    print(f"Finished plotting {i+1 if num_to_plot > 0 else 0} subjects.")


def get_arg_parser():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize individual trajectories."
    )
    parser.add_argument(
        "--site", required=True, help="Dataset name (e.g., ADNI, OASIS)"
    )
    parser.add_argument(
        "--total_count",
        default="None",
        type=str,  # Keep as str to handle 'None' or numbers
        help="Total number of individuals to plot (e.g., 10 or 'all')",
    )
    parser.add_argument(
        "--target",
        choices=["MMSE", "VENT", "DX"],
        required=True,  # Made target required and choice
        help="Target variable to plot (MMSE, VENT, DX)",
    )
    return parser


def main(args):
    """Main execution function."""
    assert args.target in (
        "MMSE",
        "VENT",
        "DX",
    ), f"{args.target} not in (MMSE, VENT, DX), please change your input argument"

    # Step1: Load basic_test.csv (input half raw data)
    inp_filename = (
        global_config.fold_gen_path / f"{args.site}" / "basic_test.csv"
    )
    input_half = pd.read_csv(inp_filename)
    input_half = input_half[
        ["RID", "EXAMDATE", "DX", "MMSE", "Ventricles", "ICV"]
    ]
    input_half["Ventricles_ICV"] = input_half["Ventricles"] / input_half["ICV"]

    # Step 2: Load ground truth and prediction csv
    gt_filename = global_config.clean_gt_path / f"{args.site}" / "test_gt.csv"
    gtDf = pd.read_csv(gt_filename)
    forecast_filename = (
        global_config.prediction_dir
        / "L2C_FNN"
        / f"{args.site}"
        / "old_metric"
        / "ensembled_prediction.csv"
    )
    forecasrDf = pd.read_csv(forecast_filename)

    # Step 3: Pre-process the loaded dataframes
    # Map diagnosis to integers (so we can plot them)
    dx_map = {"CN": 0, "MCI": 1, "AD": 2}
    gtDf["DX"] = gtDf["Diagnosis"].map(dx_map)

    # Compute forecast diagnosis by taking argmax over probabilities
    diagnosis_cols = [
        "CN relative probability",
        "MCI relative probability",
        "AD relative probability",
    ]
    forecasrDf["DX"] = forecasrDf[diagnosis_cols].values.argmax(axis=1)

    # Step 4: Define save_path and start plotting
    save_path = (
        global_config.root_path
        / "trajectory_plots"
        / f"{args.site}"
        / f"{args.target}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    # Handle "all" for total_count
    total_count_arg = None
    if args.total_count != "None" and args.total_count.lower() != "all":
        total_count_arg = args.total_count

    plot_individual_trajectory(
        input_half, gtDf, forecasrDf, args.target, save_path, total_count_arg
    )


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)
