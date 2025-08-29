#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Chen Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import argparse
import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import (  # For adding months accurately
    relativedelta,
)

from config import global_config


# --- Configuration ---
NUM_PARTICIPANTS = 200
MIN_VISITS = 2
MAX_VISITS = 7

# Visit interval stats based on ADNI data
MEAN_VISIT_INTERVAL_MONTHS = 8.44
STD_DEV_VISIT_INTERVAL_MONTHS = 5.69
MIN_VISIT_INTERVAL_MONTHS = 3.0

# Date range for random baseline visit dates
BASELINE_DATE_START = datetime.date(2005, 1, 1)
BASELINE_DATE_END = datetime.date(2008, 12, 31)

# --- Helper Functions ---


def get_probs(counts_dict):
    """
    Calculates options and probabilities from a dictionary of counts.

    Args:
        counts_dict: dictionary of counts for categorical features

    Returns:
        options: possible categories
        probabilities: probabilities for caategories
    """
    total = sum(counts_dict.values())
    # Handle both numeric and string keys
    options = [
        k if isinstance(k, str) else float(k) for k in counts_dict.keys()
    ]
    probabilities = [v / total for v in counts_dict.values()]
    return options, probabilities


# --- Data Generation Functions ---


def apply_constraints(value, constraints):
    """Applies min/max clipping, integer conversion, and rounding."""
    if constraints is None:
        return value
    # Apply min/max constraints first
    if "min" in constraints:
        # Ensure type compatibility for comparison if value is numeric
        if isinstance(value, (int, float)):
            value = max(constraints["min"], value)
    if "max" in constraints:
        if isinstance(value, (int, float)):
            value = min(constraints["max"], value)

    # Apply type/format constraints
    if constraints.get("integer", False):
        # Attempt conversion only if value is numeric-like
        try:
            value = int(round(float(value)))
        except (ValueError, TypeError):
            # Keep original value if conversion fails (e.g., it's already a string)
            pass
    elif "decimals" in constraints:
        # Attempt conversion only if value is numeric-like
        try:
            value = round(float(value), constraints["decimals"])
        except (ValueError, TypeError):
            # Keep original value if conversion fails
            pass
    return value


def generate_baseline_value(config):
    """Generates a baseline value based on distribution and constraints."""
    dist_type = config["baseline_dist"][0]
    params = config["baseline_dist"][1:]
    value = None

    if dist_type == "normal":
        mean, std_dev = params[0], params[1]
        value = np.random.normal(mean, std_dev)
    elif dist_type == "uniform":
        low, high = params[0], params[1]
        value = np.random.uniform(low, high)
    elif dist_type == "categorical":
        options, probs = params[0], params[1]
        # np.random.choice works correctly with string options
        value = np.random.choice(options, p=probs)

    # Apply constraints *after* generating value (handles strings correctly)
    return apply_constraints(value, config.get("constraints"))


def generate_progression_rate(config):
    """Generates a per-participant progression rate."""
    if "progression_rate" not in config:
        return 0  # Default to 0 if no rate specified
    dist_type = config["progression_rate"][0]
    params = config["progression_rate"][1:]
    if dist_type == "normal":
        mean, std_dev = params[0], params[1]
        return np.random.normal(mean, std_dev)
    return 0  # Default for other types


def generate_random_date(start_date, end_date):
    """Generates a random date between start_date and end_date."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    # Ensure days_between_dates is non-negative
    if days_between_dates < 0:
        return start_date  # Or handle error appropriately
    random_number_of_days = random.randrange(
        max(1, days_between_dates)
    )  # Use max(1,...) if start=end
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    return random_date


def get_viscode_from_month(month):
    """Derives VISCODE string ('bl', 'm06', 'm12' etc.) from months."""
    if month < 1.5:  # Allow some tolerance around 0 for baseline
        return "bl"
    else:
        # Simple approach: round to nearest month and format
        # More complex logic could round to nearest 6m interval if needed
        rounded_month = int(round(month))
        # Handle potential negative months if interval calculation leads to it
        if rounded_month < 0:
            rounded_month = 0
        return f"m{rounded_month:02d}"  # Format as mXX (e.g., m06, m12, m24)


def build_column_config(site):
    """Creates and returns the main COLUMN_CONFIG dictionary."""

    # Calculate probabilities *inside* this function
    ptgender_opts, ptgender_probs = get_probs({"2.0": 1151, "1.0": 1270})
    apoe_opts, apoe_probs = get_probs({"0.0": 1200, "1.0": 803, "2.0": 211})
    ptmarry_counts = {
        "Married": 1820,
        "Widowed": 265,
        "Divorced": 228,
        "Never married": 98,
        "Unknown": 9,
    }
    ptmarry_opts, ptmarry_probs = get_probs(ptmarry_counts)

    dx_opts, dx_probs = get_probs({"0.0": 889, "1.0": 1095, "2.0": 413})
    cdr_opts, cdr_probs = get_probs(
        {"0.0": 905, "0.5": 1294, "1.0": 219, "2.0": 3}
    )

    age_col = "curr_age" if site == "ADNI" else "AGE"
    # Define the dictionary using the locally calculated variables
    config = {
        "RID": {"type": "id"},
        "Month_bl": {"type": "time_numeric"},  # Numeric months since baseline
        "VISCODE": {
            "type": "time_code"
        },  # Derived visit code ('bl', 'm06', etc.)
        "EXAMDATE": {"type": "time_date"},  # Derived exam date
        age_col: {
            "type": "static_num",  # Age at baseline, will calculate Age at visit later
            "baseline_dist": ("normal", 72.92, 7.38),
            "constraints": {"min": 50.4, "max": 999, "decimals": 1},
        },
        "PTEDUCAT": {
            "type": "static_num",
            "baseline_dist": ("normal", 16.05, 2.73),
            "constraints": {"min": 4, "max": 20, "integer": True},
        },
        "PTGENDER": {
            "type": "static_cat",
            "baseline_dist": ("categorical", ptgender_opts, ptgender_probs),
            "constraints": {"integer": True},
        },  # Assuming 0/1 should be int
        "APOE": {
            "type": "static_cat",
            "baseline_dist": ("categorical", apoe_opts, apoe_probs),
            "constraints": {"integer": True},
        },  # Assuming 0/1/2 should be int
        "PTMARRY": {
            "type": "static_cat",  # Now uses string options
            "baseline_dist": ("categorical", ptmarry_opts, ptmarry_probs),
            "constraints": {},
        },  # No longer constrained to integer
        # Dynamic Features - Progression logic added
        "MMSE": {
            "type": "dynamic_num",
            "baseline_dist": ("normal", 27.38, 2.65),
            "constraints": {
                "min": 0,
                "max": 30,
                "integer": True,
            },  # Changed min from 16 to 0
            "progression_rate": (
                "normal",
                -0.08,
                0.04,
            ),  # Avg decline per month + variability
            "noise_std": 0.7,
        },  # Visit-to-visit noise
        "DX": {
            "type": "dynamic_ord",  # Ordinal progression
            "baseline_dist": ("categorical", dx_opts, dx_probs),
            "constraints": {"integer": True},
            "options": [0.0, 1.0, 2.0],  # Possible states
            "progression_prob": 0.15,
        },  # Chance of worsening by one step per visit
        "CDR": {
            "type": "dynamic_ord",  # Ordinal progression
            "baseline_dist": ("categorical", cdr_opts, cdr_probs),
            "constraints": {},  # Keep 0.5 as float? Or specify decimals: 1
            "options": [0.0, 0.5, 1.0, 2.0],  # Possible states
            "progression_prob": 0.15,
        },  # Chance of worsening by one step per visit
        # Brain Volumes - Progression logic added
        "Hippocampus": {
            "type": "dynamic_num",
            "baseline_dist": ("normal", 6938.54, 1083.45),
            "constraints": {"min": 3209.5, "max": 11049.0, "decimals": 1},
            "progression_rate": (
                "normal",
                -16.0,
                8.0,
            ),  # Avg decline per month + variability
            "noise_std": 100.0,
        },
        "Fusiform": {
            "type": "dynamic_num",
            "baseline_dist": ("normal", 17320.42, 2465.6),
            "constraints": {"min": 9814.0, "max": 26560.0, "decimals": 1},
            "progression_rate": (
                "normal",
                -30.0,
                16.0,
            ),  # Avg decline per month + variability
            "noise_std": 200.0,
        },
        "MidTemp": {
            "type": "dynamic_num",
            "baseline_dist": ("normal", 19747.07, 3043.92),
            "constraints": {"min": 10319.0, "max": 32128.0, "decimals": 1},
            "progression_rate": (
                "normal",
                -40.0,
                20.0,
            ),  # Avg decline per month + variability
            "noise_std": 250.0,
        },
        "Ventricles": {
            "type": "dynamic_num",
            "baseline_dist": ("normal", 41304.60, 22901.79),
            "constraints": {"min": 7403.3, "max": 154256.0, "decimals": 1},
            "progression_rate": (
                "normal",
                100.0,
                50.0,
            ),  # Avg increase per month + variability
            "noise_std": 500.0,
        },
        "ICV": {
            "type": "dynamic_num",  # Intracranial Volume - should be relatively stable
            "baseline_dist": ("normal", 1.53e06, 1.69e05),
            "constraints": {"min": 9.83e05, "max": 2.23e06, "decimals": 1},
            "progression_rate": (
                "normal",
                0.0,
                50.0,
            ),  # Very small change around 0
            "noise_std": 1000.0,
        },
        "WholeBrain": {
            "type": "dynamic_num",
            "baseline_dist": ("normal", 1.04e06, 1.08e05),
            "constraints": {"min": 7.05e05, "max": 1.46e06, "decimals": 1},
            "progression_rate": (
                "normal",
                -200.0,
                100.0,
            ),  # Avg decline per month + variability
            "noise_std": 1000.0,
        },
    }
    return config


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate dummy data for illustration purpose."
    )
    parser.add_argument(
        "--site", required=True, help="Site name for the fake dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # --- Define COLUMN_CONFIG at top level by calling the function ---
    COLUMN_CONFIG = build_column_config(args.site)

    # --- Main Generation Loop ---
    print(f"Generating fake data for {NUM_PARTICIPANTS} participants...")
    age_col = "curr_age" if args.site == "ADNI" else "AGE"
    all_visits_data = []
    participant_id_counter = 1

    for i in range(NUM_PARTICIPANTS):
        # Use RID as participant identifier
        rid = participant_id_counter
        participant_id_counter += 1

        num_visits = np.random.randint(MIN_VISITS, MAX_VISITS + 1)

        # Generate static baseline characteristics and progression rates
        baseline_data = {"RID": rid}
        progression_rates = {}  # For dynamic_num features
        age_at_baseline = 0  # Store separately
        baseline_exam_date = generate_random_date(
            BASELINE_DATE_START, BASELINE_DATE_END
        )

        for col, config in COLUMN_CONFIG.items():
            col_type = config["type"]
            if col_type in ["static_cat", "static_num"]:
                baseline_data[col] = generate_baseline_value(config)
                if col == age_col:  # Store baseline age
                    age_at_baseline = baseline_data[col]
            elif col_type in ["dynamic_num", "dynamic_ord"]:
                # Store baseline value
                baseline_data[col] = generate_baseline_value(config)
                # Store progression rate if applicable
                if col_type == "dynamic_num":
                    progression_rates[col] = generate_progression_rate(config)

        # Generate visits for this participant
        current_time_months = 0.0
        last_visit_data = {}  # Store previous visit data for progression

        for visit_num in range(num_visits):
            visit_data = {}
            visit_data["RID"] = rid

            # --- Time Columns ---
            # Store numeric months
            visit_data["Month_bl"] = round(
                current_time_months, 1
            )  # Store with 1 decimal place

            # Derive VISCODE string
            visit_data["VISCODE"] = get_viscode_from_month(current_time_months)

            # Calculate and store EXAMDATE
            # Use relativedelta for accurate month addition
            # Add integer months for simplicity in date calculation
            months_to_add = int(round(current_time_months))
            # Handle potential negative months if interval calculation leads to it
            if months_to_add < 0:
                months_to_add = 0
            current_exam_date = baseline_exam_date + relativedelta(
                months=months_to_add
            )
            visit_data["EXAMDATE"] = current_exam_date.strftime("%Y-%m-%d")

            # --- Populate other data for this visit ---
            for col, config in COLUMN_CONFIG.items():
                # Skip time/id columns already handled
                if config["type"] in [
                    "id",
                    "time_numeric",
                    "time_code",
                    "time_date",
                ]:
                    continue

                col_type = config["type"]

                if col_type in ["static_cat", "static_num"]:
                    # Handle specific cases like Age at Visit
                    if col == age_col:
                        visit_data[col] = apply_constraints(
                            age_at_baseline + (current_time_months / 12.0),
                            config.get("constraints"),
                        )
                    else:
                        # Copy static value from baseline
                        visit_data[col] = baseline_data[col]

                elif col_type == "dynamic_num":
                    if visit_num == 0:  # Baseline visit
                        visit_data[col] = baseline_data[col]
                    else:
                        # Calculate progression: baseline + rate*time + noise
                        baseline_val = baseline_data[col]
                        rate = progression_rates[col]
                        noise = np.random.normal(0, config.get("noise_std", 0))
                        # Use the precise current_time_months for calculation
                        current_val = (
                            baseline_val + (rate * current_time_months) + noise
                        )
                        visit_data[col] = apply_constraints(
                            current_val, config.get("constraints")
                        )

                elif col_type == "dynamic_ord":
                    if visit_num == 0:  # Baseline visit
                        visit_data[col] = baseline_data[col]
                    else:
                        # Ordinal progression logic
                        prev_val = last_visit_data[col]
                        options = config["options"]
                        prog_prob = config["progression_prob"]
                        current_val = prev_val  # Default: stay the same

                        # Find index of previous value
                        try:
                            # Handle potential float vs int comparison if options are numeric
                            if isinstance(prev_val, float) and all(
                                isinstance(opt, int) for opt in options
                            ):
                                prev_idx = options.index(int(prev_val))
                            elif isinstance(prev_val, int) and all(
                                isinstance(opt, float) for opt in options
                            ):
                                prev_idx = options.index(float(prev_val))
                            else:
                                prev_idx = options.index(prev_val)
                        except ValueError:
                            prev_idx = (
                                -1
                            )  # Should not happen if baseline is valid

                        # Check if worsening is possible and roll the dice
                        if prev_idx != -1 and prev_idx < len(options) - 1:
                            if random.random() < prog_prob:  # Worsen
                                current_val = options[prev_idx + 1]

                        visit_data[col] = apply_constraints(
                            current_val, config.get("constraints")
                        )

            all_visits_data.append(visit_data)
            last_visit_data = (
                visit_data.copy()
            )  # Store current visit data for next iteration

            # Calculate time for the next visit
            if visit_num < num_visits - 1:
                interval = np.random.normal(
                    MEAN_VISIT_INTERVAL_MONTHS, STD_DEV_VISIT_INTERVAL_MONTHS
                )
                # Ensure minimum interval and positive value
                interval = max(MIN_VISIT_INTERVAL_MONTHS, interval)
                interval = max(
                    0.1, interval
                )  # Ensure interval is slightly positive
                current_time_months += interval
                # No rounding here, keep precise for next calculation

    # --- Create DataFrame and Save ---
    df = pd.DataFrame(all_visits_data)

    # Define desired column order based on COLUMN_CONFIG and generated columns
    ordered_columns = [
        "RID",
        "VISCODE",
        "Month_bl",
        "EXAMDATE",
        "curr_age",
        "PTEDUCAT",
        "PTGENDER",
        "APOE",
        "PTMARRY",
        "MMSE",
        "DX",
        "CDR",
        "Hippocampus",
        "Fusiform",
        "MidTemp",
        "Ventricles",
        "ICV",
        "WholeBrain",
    ]
    # Add any extra columns that might have been generated but not in the explicit list
    final_columns = ordered_columns + [
        col for col in df.columns if col not in ordered_columns
    ]
    # Ensure all defined columns exist before ordering
    final_columns = [col for col in final_columns if col in df.columns]
    df = df[final_columns]
    # Add SCANDATE column (just copy the EXAMDATE column)
    df["SCANDATE"] = df["EXAMDATE"]

    # Save path for generated dummy data
    OUTPUT_FILENAME = global_config.raw_data_path / f"{args.site}.csv"

    # Ensure output directory exists (if specified)
    output_path = Path(OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Successfully generated fake data and saved to '{OUTPUT_FILENAME}'")
    print(f"DataFrame shape: {df.shape}")
    print("First 5 rows:")
    print(df.head().to_string())  # Use to_string to prevent truncation


if __name__ == "__main__":
    main()
