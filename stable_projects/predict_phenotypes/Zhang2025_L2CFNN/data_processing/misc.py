#!/usr/bin/env python
# encoding: utf-8
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import json
from datetime import datetime

import numpy as np
import pandas as pd


# Mainly for step 0
def load_feature(feature_file_path):
    """
    Load feature dict from json file
    """
    with open(feature_file_path, "rb") as file:
        data = json.load(file)
    return data


def str2date(string):
    """Convert string to datetime object"""
    if string != "":
        return datetime.strptime(string, "%Y-%m-%d")
    return np.NaN


CONVERTERS = {
    "CognitiveAssessmentDate": str2date,
    "ScanDate": str2date,
    "SCANDATE": str2date,
    "Forecast Date": str2date,
    "EXAMDATE": str2date,
}


# Mainly for step 1
def APOE_conv(value):
    """
    Convert APOE from string to float
    based on number of E4
    """
    if isinstance(value, str):
        if value == "":
            return float("NaN")
        else:
            return value.count("E4")
    return float("NaN")


def MACC_marriage_conv(value):
    """
    Convert marital status from string to float

    Mapping:
        1, Never Married | 2, Married | 3, Divorced | 4, Widowed | 5, Others
    """
    try:
        if value == "":
            return float("nan")
        val = float(value)
    except (ValueError, TypeError):
        return float("nan")

    if val == 2.0:
        return 1.0
    elif val == 5.0:
        return 2.0
    else:
        return 0.0


def OASIS_marriage_conv(value):
    """
    Convert marital status from string to float

    Mapping:
        1-Married | 2-Widowed | 3-Divorced | 4-Separated
        5-Never married (or marriage was annulled)
        6-Living as married/domestic partner
        9-Unknown
    """
    try:
        if value == "":
            return float("nan")
        val = float(value)
    except (ValueError, TypeError):
        return float("nan")

    if val in (1.0, 6.0):
        return 1.0
    elif val == 9.0:
        return 2.0
    else:
        return 0.0


def ADNI_marriage_conv(value):
    """
    Convert marital status from string to float

    Mapping:
        1=Married | 2=Widowed | 3=Divorced | 4=Never married | 5=Unknown
    """
    if isinstance(value, str):
        if value == "Married":
            return 1.0
        elif value == "Unknown":
            return 2.0
        elif value == "":
            return float("NaN")
        else:
            return 0.0
    return float("NaN")


def AIBL_marriage_conv(value):
    """
    Convert marital status from string to float

    Mapping:
        Married | Widowed | Divorced | Single | Cohabiting |
        Separated | NaN
    """
    if isinstance(value, str):
        if value == "Married":
            return 1.0
        elif value == "Cohabiting":
            return 1.0
        elif value == "":
            return float("NaN")
        else:
            return 0.0
    return float("NaN")


def Gender_conv(value):
    """
    Convert gender from string to float (whether it's male)

    Mapping:
        1, Male | 2, Female | 99,Unknown
    """
    try:
        if value == "":
            return float("nan")
        val = float(value)
    except (ValueError, TypeError):
        return float("nan")

    if val == 1.0:
        return 1.0
    elif val == 2.0:
        return 0.0
    else:
        return float("nan")


# Mainly for step 2
def date_conv(value):
    """
    Convert date format from 'YY-MM-DD' to 'MM/DD/YY'
    """
    if isinstance(value, str):
        splitted = value.split("-")
        return f"{int(splitted[1])}/{int(splitted[2])}/{splitted[0]}"
    return float("NaN")


# Specifically for step 3
def _calculate_extreme_value_features(past_values, past_times, current_time):
    """Calculates lowest/highest values and time since their last occurrence."""
    lowest_val = past_values.min(skipna=True)
    highest_val = past_values.max(skipna=True)

    # Time since last occurrence of the lowest value
    time_since_lowest = (
        current_time - past_times[past_values == lowest_val].iloc[-1]
    )
    # Time since last occurrence of the highest value
    time_since_highest = (
        current_time - past_times[past_values == highest_val].iloc[-1]
    )

    return lowest_val, time_since_lowest, highest_val, time_since_highest


def _calculate_most_recent_value_features(
    past_values, past_times, current_time, non_missing_mask
):
    """Calculates the most recent value and time since it occurred."""
    most_recent_val = past_values[non_missing_mask].iloc[-1]
    time_since_most_recent = (
        current_time - past_times[non_missing_mask].iloc[-1]
    )
    return most_recent_val, time_since_most_recent


def _calculate_most_recent_change_rate(
    past_values, past_times, non_missing_mask
):
    """Calculates the rate of change for the most recent value."""
    if non_missing_mask.sum() > 1:
        # Ensure the time difference between the last two points is not zero
        time_diff = (
            past_times[non_missing_mask].iloc[-1]
            - past_times[non_missing_mask].iloc[-2]
        )
        if time_diff != 0:
            # Calculate change rate: (value_change / time_change)
            change_rate = (
                past_values[non_missing_mask].iloc[-1]
                - past_values[non_missing_mask].iloc[-2]
            ) / time_diff
            return change_rate
        else:
            return np.nan  # Time difference is zero
    return np.nan  # Not enough data points to calculate change


def create_long_features(
    value_series,
    time_series,
    ordered_selection_indices,
    history_lookback_count,
    current_point_selector_idx,
):
    """
    Calculates longitudinal features based on past data points.

    The function assumes `ordered_selection_indices` contains indices (from the original
    DataFrame where `value_series` and `time_series` come from) that are already
    sorted chronologically for a given subject/entity.

    `history_lookback_count` determines how many of the earliest points from
    `ordered_selection_indices` are considered "past" data.
    `current_point_selector_idx` is an index into `ordered_selection_indices`
    that identifies the "current" time point for which features are being calculated.
    The past data should ideally precede this current time point.

    Args:
        value_series: Input data column (e.g., pandas Series like MMSE).
        time_series: Corresponding time points (e.g., pandas Series like Month_bl).
        ordered_selection_indices: Ordered list/array of integer indices to select
                                   the correct rows from `value_series` and `time_series`.
                                   These are assumed to be chronologically sorted.
        history_lookback_count: The number of data points from the beginning of
                                `ordered_selection_indices` to consider as history.
                                (corresponding to inner loop index)
        current_point_selector_idx: The index within `ordered_selection_indices` that
                                    points to the current observation being evaluated.
                                    (corresponding to outer loop index)

    Returns:
        A list containing calculated features:
        [most_recent_value, time_since_most_recent, most_recent_change_rate,
         lowest_value, time_since_lowest, highest_value, time_since_highest]
        Returns NaNs if insufficient data.
    """

    # Select past data points based on the provided indices up to history_lookback_count
    # These are the actual indices from the original dataframe
    past_indices_to_select = ordered_selection_indices[:history_lookback_count]
    past_values = value_series.iloc[past_indices_to_select]
    past_times = time_series.iloc[past_indices_to_select]

    # Identify non-missing past data points
    non_missing_mask = past_values.notna()

    # Check if there's at least one non-missing past data point
    if non_missing_mask.sum() > 0:
        # Get the current time value for calculating "time since" features
        current_time_iloc = ordered_selection_indices[
            current_point_selector_idx
        ]
        current_time = time_series.iloc[current_time_iloc]

        # Calculate lowest and highest value features
        lowest_val, time_since_lowest, highest_val, time_since_highest = (
            _calculate_extreme_value_features(
                past_values, past_times, current_time
            )
        )

        # Calculate most recent value features
        most_recent_val, time_since_mr = _calculate_most_recent_value_features(
            past_values, past_times, current_time, non_missing_mask
        )

        # Calculate the rate of change for the most recent value
        mr_change_rate = _calculate_most_recent_change_rate(
            past_values, past_times, non_missing_mask
        )

        return [
            most_recent_val,
            time_since_mr,
            mr_change_rate,
            lowest_val,
            time_since_lowest,
            highest_val,
            time_since_highest,
        ]

    # Return NaNs for all features if there are no non-missing past data points
    return [np.nan] * 7


def _calculate_milder_state_features(
    past_diagnosis_values,
    past_times,
    most_recent_diagnosis,  # Diagnosis codes can be float/int
    current_time,
):
    """
    Calculates features related to the occurrence of a 'milder' diagnosis state.
    A milder state is defined as a numerically lower diagnosis code.

    Args:
        past_diagnosis_values: Series of historical diagnosis codes.
        past_times: Series of corresponding historical time points.
        most_recent_diagnosis: The most recent diagnosis code for comparison.
        current_time: The current time point for calculating "time since".

    Returns:
        A tuple: (milder_occurred_flag, time_since_last_milder_state)
        time_since_last_milder_state is 999 (as per original logic) or np.nan
        if no milder state occurred or current diagnosis is NaN.
    """
    # If the current diagnosis is NaN, comparison for "milder" is not meaningful.
    if pd.isna(most_recent_diagnosis):
        # Consider returning (np.nan, np.nan) for consistency if 999 is not a required sentinel.
        return 0, 999

    # Identify past diagnosis values that were 'milder' (numerically less).
    # Only consider non-NaN past values for comparison.
    valid_past_mask = past_diagnosis_values.notna()
    milder_mask = (
        past_diagnosis_values[valid_past_mask] < most_recent_diagnosis
    )

    if milder_mask.sum() == 0:
        # No past state was milder
        milder_occurred_flag = 0
        # Using 999 as per original logic. np.nan might be more standard for "not applicable".
        time_since_milder = 999
    else:
        milder_occurred_flag = 1
        # Time since the *last* occurrence of a milder state.
        # We need to index past_times with the original indices of milder values.
        time_since_milder = (
            current_time - past_times[valid_past_mask][milder_mask].iloc[-1]
        )

    return milder_occurred_flag, time_since_milder


def create_dx_features(
    diagnosis_series,
    time_series,
    ordered_selection_indices,
    history_lookback_count,
    current_point_selector_idx,
):
    """
    It first calls `create_longitudinal_features` to get base metrics,
    adapts its output (e.g., 'lowest' becomes 'best_dx', 'rate of change' is excluded),
    and then adds features related to the time since a 'milder' (numerically lower)
    diagnosis state occurred in the past.

    Args:
        diagnosis_series: Input data column (e.g., pandas Series like MMSE).
        time_series: Corresponding time points (e.g., pandas Series like Month_bl).
        ordered_selection_indices: Ordered list/array of integer indices to select
                                   the correct rows from `diagnosis_series` and `time_series`.
                                   These are assumed to be chronologically sorted.
        history_lookback_count: The number of data points from the beginning of
                                `ordered_selection_indices` to consider as history.
                                (corresponding to inner loop index)
        current_point_selector_idx: The index within `ordered_selection_indices` that
                                    points to the current observation being evaluated.
                                    (corresponding to outer loop index)

    Returns:
        A list containing calculated features:
        [mr_dx, time_since_mr_dx, best_dx_overall, time_since_best_dx_overall,
         worst_dx_overall, time_since_worst_dx_overall, milder_occurred_flag,
         time_since_last_milder_state]
        Base features can be NaNs if insufficient data.
        'milder' features default to (0, 999) if no milder state found or current dx is NaN.
    """
    # Get base longitudinal features
    # Expected return: [mr_val, ts_mr, mr_change, low_val, ts_low, high_val, ts_high]
    base_long_features = create_long_features(
        value_series=diagnosis_series,  # Pass diagnosis_series as the value_series
        time_series=time_series,
        ordered_selection_indices=ordered_selection_indices,
        history_lookback_count=history_lookback_count,
        current_point_selector_idx=current_point_selector_idx,
    )

    # Unpack and adapt features from create_longitudinal_features.
    # The 'mr_change' (index 2) is not used for diagnosis features.
    # "best_dx" implies numerically lowest diagnosis code.
    # "worst_dx" implies numerically highest diagnosis code.
    mr_dx = base_long_features[0]  # most_recent_value
    time_since_mr_dx = base_long_features[1]  # time_since_most_recent
    # base_long_features[2] is most_recent_change_rate - skipped for dx
    best_dx_overall = base_long_features[3]  # lowest_value
    time_since_best_dx_overall = base_long_features[4]  # time_since_lowest
    worst_dx_overall = base_long_features[5]  # highest_value
    time_since_worst_dx_overall = base_long_features[6]  # time_since_highest

    dx_features_base = [
        mr_dx,
        time_since_mr_dx,
        best_dx_overall,
        time_since_best_dx_overall,
        worst_dx_overall,
        time_since_worst_dx_overall,
    ]

    # For milder state calculation, we need the historical diagnosis data and current time.
    # Ensure .iloc is used for label-based indexing with ordered_selection_indices.
    past_indices_to_select = ordered_selection_indices[:history_lookback_count]
    past_diagnosis_values = diagnosis_series.iloc[past_indices_to_select]
    past_times = time_series.iloc[past_indices_to_select]

    # Get the current time using the label from ordered_selection_indices
    current_time_label = ordered_selection_indices[current_point_selector_idx]
    current_time = time_series.iloc[current_time_label]

    # The most_recent_diagnosis for comparison is mr_dx, which we already extracted.
    milder_occurred_flag, time_since_milder = _calculate_milder_state_features(
        past_diagnosis_values=past_diagnosis_values,
        past_times=past_times,
        most_recent_diagnosis=mr_dx,  # This is the most_recent_value from base_long_features
        current_time=current_time,
    )

    return dx_features_base + [milder_occurred_flag, time_since_milder]


def gen_feature_names(label):
    """
    Generates a list of feature names based on the input label.

    Provides specific names for the 'dx' label and generic names otherwise,
    matching the outputs of create_dx_features and create_long_features.

    Args:
        label: A string indicating the type of feature (e.g., 'dx', 'ADAS13').

    Returns:
        A list of strings representing the feature names.
    """
    # Special handling for 'dx' features
    if label == "dx":
        return [
            "mr_dx",
            "time_since_mr_dx",
            "best_dx",
            "time_since_best_dx",
            "worst_dx",
            "time_since_worst_dx",
            "milder",
            "time_since_milder",
        ]
    # Generic feature names for other labels
    return [
        f"mr_{label}",
        f"time_since_mr_{label}",
        f"mr_change_{label}",
        f"low_{label}",
        f"time_since_low_{label}",
        f"high_{label}",
        f"time_since_high_{label}",
    ]


# For step 4
categorical_dict = {
    "apoe": 3,
    "male": 2,
    "married": 2,
    "mr_CDR": 5,
    "high_CDR": 5,
    "low_CDR": 5,
    "mr_dx": 3,
    "best_dx": 3,
    "worst_dx": 3,
    "milder": 2,
}

cdr_mapping = {0: 0, 0.5: 1, 1: 2, 2: 3, 3: 4}
