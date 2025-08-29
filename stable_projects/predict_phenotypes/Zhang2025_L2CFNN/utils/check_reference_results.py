#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Chen Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from config import global_config

# --- Constants ---
NUM_FOLDS = 20
# Explicitly define the metrics and their source folders/paths
METRIC_SOURCES = {
    "mAUC": {"folder": "old_metric", "key": "mAUC"},
    "mmseMAE": {"folder": "new_metric", "key": "mmseMAE"},
    "ventsMAE": {"folder": "new_metric", "key": "ventsMAE"},
}
DEFAULT_TOLERANCE = 1e-6  # Tolerance for floating point comparisons

# --- Helper Functions ---


def load_single_json(filepath: Path):
    """Loads a single JSON file, handling potential errors."""
    if not filepath.is_file():
        print(f"  Warning: JSON file not found: {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(
            f"  Warning: Error decoding JSON file: {filepath}", file=sys.stderr
        )
        return None
    except Exception as e:
        print(
            f"  Warning: Unexpected error reading {filepath}: {e}",
            file=sys.stderr,
        )
        return None


def load_results_to_dataframe(
    pred_path_base: Path,
    model: str,
    site: str,
    num_folds: int,
    predict_only: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Loads 20-fold results for target metrics into a Pandas DataFrame.

    Args:
        pred_path_base: Base directory for predictions.
        model: The model name.
        site: The site name.

    Returns:
        A DataFrame with columns ['SITE', 'MODEL', 'Metric', 'Fold', 'Value']
        or None if loading fails significantly.
    """
    print(f"Loading results for Model: {model}, Site: {site}...")
    df_dict = defaultdict(list)
    data_found = False

    for fold in range(num_folds):
        fold_data_found_this_iter = False
        # --- Load from new_metric folder ---
        if predict_only:
            new_metric_folder_path = (
                pred_path_base / model / site / "new_metric"
            )
        else:
            new_metric_folder_path = (
                pred_path_base / model / site / "new_metric" / f"fold_{fold}"
            )
        new_metric_file_path = new_metric_folder_path / "out.json"

        tp_result_new = load_single_json(new_metric_file_path)

        if tp_result_new:
            for metric_name, info in METRIC_SOURCES.items():
                if (
                    info["folder"] == "new_metric"
                    and info["key"] in tp_result_new
                ):
                    df_dict["MODEL"].append(model)
                    df_dict["SITE"].append(site)
                    df_dict["Metric"].append(
                        metric_name
                    )  # Use consistent metric name
                    df_dict["Fold"].append(fold)
                    df_dict["Value"].append(tp_result_new[info["key"]])
                    data_found = True
                    fold_data_found_this_iter = True

        # --- Load from old_metric folder ---
        if predict_only:
            old_metric_folder_path = (
                pred_path_base / model / site / "old_metric"
            )
        else:
            old_metric_folder_path = (
                pred_path_base / model / site / "old_metric" / f"fold_{fold}"
            )
        old_metric_file_path = old_metric_folder_path / "out.json"
        tp_result_old = load_single_json(old_metric_file_path)

        if tp_result_old:
            for metric_name, info in METRIC_SOURCES.items():
                # Ensure we don't double-add if keys overlap and exist in both files
                is_already_added = (
                    (
                        info["folder"] == "old_metric"
                        and model == df_dict["MODEL"][-1]
                        and site == df_dict["SITE"][-1]
                        and metric_name == df_dict["Metric"][-1]
                        and fold == df_dict["Fold"][-1]
                    )
                    if fold_data_found_this_iter
                    else False
                )

                if (
                    info["folder"] == "old_metric"
                    and info["key"] in tp_result_old
                    and not is_already_added
                ):
                    df_dict["MODEL"].append(model)
                    df_dict["SITE"].append(site)
                    df_dict["Metric"].append(
                        metric_name
                    )  # Use consistent metric name
                    df_dict["Fold"].append(fold)
                    df_dict["Value"].append(tp_result_old[info["key"]])
                    data_found = True
                    # fold_data_found_this_iter = True # No need to set again

    if not data_found:
        print(
            f"  Warning: No data loaded for Model: {model}, Site: {site}",
            file=sys.stderr,
        )
        return None

    df = pd.DataFrame(df_dict)
    # Check if all folds were loaded for each metric
    fold_counts = df.groupby("Metric")["Fold"].nunique()
    for metric, count in fold_counts.items():
        if count < num_folds:
            print(
                f"  Warning: Loaded only {count}/{num_folds} folds for Metric: {metric}, Model: {model}, Site: {site}",
                file=sys.stderr,
            )

    return df


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare prediction results against reference values."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to check.",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        required=True,
        help="List of site names to check.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Absolute and relative tolerance for float comparison (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=NUM_FOLDS,
        help=f"Number of folds used for comparison, for example and unit test it should be 1 (default: {NUM_FOLDS})",
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help=f"Whether perform comparison for predict_only folder",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    failures = []
    comparison_count = 0
    models_sites_processed = 0

    # --- Load Reference Data ---
    try:
        ref_csv_path = global_config.reference_dir / "ref_results.csv"
        ref_df = pd.read_csv(ref_csv_path)
        # Prepare reference DataFrame for easy lookup
        ref_df.set_index(["MODEL", "SITE", "Metric"], inplace=True)
        print(f"Loaded reference data from: {ref_csv_path}")
    except FileNotFoundError:
        print(
            f"Error: Reference file not found at {ref_csv_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyError as e:
        print(
            f"Error: Reference file {ref_csv_path} missing required column(s): {e}. "
            "Expected: model, site, metric, ref_mean, ref_std",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"Error loading reference file {ref_csv_path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Iterate and Compare ---
    print("\nStarting comparison...")
    for model in args.models:
        for site in args.sites:
            models_sites_processed += 1
            # Load results for current model/site
            loaded_df = load_results_to_dataframe(
                global_config.prediction_dir,
                model,
                site,
                args.num_folds,
                args.predict_only,
            )

            if loaded_df is None or loaded_df.empty:
                failures.append(
                    f"FAIL [{model}/{site}]: No prediction data loaded."
                )
                print(
                    f"Skipping comparison for Model: {model}, Site: {site} due to loading issues."
                )
                continue

            # Calculate mean and std dev across folds
            try:
                # Use ddof=1 for sample standard deviation, matching pandas default
                stats = loaded_df.groupby("Metric")["Value"].agg(
                    ["mean", "std"]
                )
            except Exception as e:
                failures.append(
                    f"FAIL [{model}/{site}]: Error calculating stats: {e}"
                )
                print(
                    f"Skipping comparison for Model: {model}, Site: {site} due to stats calculation error."
                )
                continue

            # Compare each target metric
            for metric in METRIC_SOURCES.keys():
                comparison_count += 1
                calc_mean, calc_std = None, None
                ref_mean, ref_std = None, None
                metric_passed = True
                fail_reasons = []

                # Get calculated stats
                if metric in stats.index:
                    calc_mean = stats.loc[metric, "mean"]
                    calc_std = stats.loc[metric, "std"]
                    # Handle potential NaN std if only one fold or all values identical
                    if pd.isna(calc_std):
                        calc_std = 0.0
                else:
                    fail_reasons.append("Calculated metric missing")
                    metric_passed = False

                # Get reference stats
                try:
                    ref_mean = ref_df.loc[(model, site, metric), "ref_mean"]
                    if args.num_folds != 1:
                        ref_std = ref_df.loc[(model, site, metric), "ref_std"]
                except KeyError:
                    fail_reasons.append("Reference metric missing")
                    metric_passed = False
                except Exception as e:  # Catch other potential lookup errors
                    fail_reasons.append(f"Reference lookup error: {e}")
                    metric_passed = False

                # Perform comparison if possible
                if metric_passed:
                    # Compare Mean
                    mean_match = math.isclose(
                        calc_mean,
                        ref_mean,
                        rel_tol=args.tolerance,
                        abs_tol=args.tolerance,
                    )
                    if not mean_match:
                        fail_reasons.append(
                            f"Mean mismatch (Calc: {calc_mean:.6f}, Ref: {ref_mean:.6f})"
                        )
                        metric_passed = False

                    if args.num_folds != 1:
                        # Compare Std Dev
                        std_match = math.isclose(
                            calc_std,
                            ref_std,
                            rel_tol=args.tolerance,
                            abs_tol=args.tolerance,
                        )
                        if not std_match:
                            # Check for edge case: both are zero or near-zero, which should match
                            if not (
                                abs(calc_std) < args.tolerance * 10
                                and abs(ref_std) < args.tolerance * 10
                            ):
                                fail_reasons.append(
                                    f"Std mismatch (Calc: {calc_std:.6f}, Ref: {ref_std:.6f})"
                                )
                                metric_passed = False

                # Report result for this metric only for replication (i.e., num_folds == 20)
                if metric_passed:
                    if args.num_folds != 1:
                        print(
                            f"  PASS [{model}/{site}/{metric}] (Mean: {calc_mean:.6f}, Std: {calc_std:.6f})"
                        )
                    else:
                        print(
                            f"  [{model}/{site}/{metric}] (Mean: {calc_mean:.6f})"
                        )
                else:
                    reason_str = "; ".join(fail_reasons)
                    failure_msg = (
                        f"FAIL [{model}/{site}/{metric}]: {reason_str}"
                    )
                    failures.append(failure_msg)
                    if args.num_folds != 1:
                        print(f"  {failure_msg}")

    # --- Final Summary ---
    print("\n" + "=" * 30 + " Summary " + "=" * 30)
    print(f"Processed {models_sites_processed} Model/Site combinations.")
    print(f"Performed {comparison_count} comparisons.")

    if not failures:
        print("All comparisons passed successfully!")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"\n{len(failures)} COMPARISON(S) FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("=" * 70)
        sys.exit(1)

    assert not failures, f"\n{len(failures)} COMPARISON(S) FAILED:"


if __name__ == "__main__":
    main()
