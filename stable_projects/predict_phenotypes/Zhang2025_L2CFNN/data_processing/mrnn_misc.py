#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import time
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def time_from(start):
    """Return duration from *start* to now"""
    duration = relativedelta(seconds=time.time() - start)
    return "%dm %ds" % (duration.minutes, duration.seconds)


def str2date(string):
    """Convert string to datetime object"""
    if isinstance(string, str):
        if string != "":
            return datetime.strptime(string, "%Y-%m-%d")
    return float("NaN")


def has_data_mask(frame):
    """
    Check whether rows has any valid value (i.e. not NaN)
    Args:
        frame: Pandas data frame
    Return:
        (ndarray): boolean mask with the same number of rows as *frame*
        True implies row has at least 1 valid value
    """
    return ~frame.isnull().apply(np.all, axis=1)


def get_data_dict(frame, features):
    """
    From a frame of all subjects, return a dictionary of frames
    The keys are subjects' ID
    The data frames are:
        - sorted by *Month_bl* (which are integers)
        - have empty rows dropped (empty row has no value in *features* list)
    Args:
        frame (Pandas data frame): data frame of all subjects
        features (list of string): list of features
    Return:
        (Pandas data frame): prediction frame
    """
    ret = {}
    frame_ = frame.copy()
    frame_["Month_bl"] = frame_["Month_bl"].round().astype(int)
    for subj in np.unique(frame_.RID):
        subj_data = frame_[frame_.RID == subj].sort_values(
            "Month_bl"
        )  # by default ascending and NaN at last
        subj_data = subj_data[has_data_mask(subj_data[features])]

        subj_data = subj_data.set_index("Month_bl", drop=True)
        ret[subj] = subj_data.drop(["RID"], axis=1)
    return ret


def month_between(end, start):
    """Get duration (in months) between *end* and *start* dates"""
    # assert end >= start
    diff = relativedelta(end, start)
    months = 12 * diff.years + diff.months
    to_next = relativedelta(
        end + relativedelta(months=1, days=-diff.days), end
    ).days
    to_prev = diff.days
    return months + (to_next < to_prev)


def make_date_col(starts, duration):
    """
    Return a list of list of dates
    The start date of each list of dates is specified by *starts*
    """
    date_range = [relativedelta(months=i) for i in range(duration)]
    ret = []
    for start in starts:
        ret.append([start + d for d in date_range])

    return ret


def get_index(fields, keys):
    """Get indices of *keys*, each of which is in list *fields*"""
    assert isinstance(keys, list)
    assert isinstance(fields, list)
    return [fields.index(k) for k in keys]


def to_categorical(y, nb_classes):
    """Convert list of labels to one-hot vectors"""
    if len(y.shape) == 2:
        y = y.squeeze(1)

    ret_mat = np.full((len(y), nb_classes), np.nan)
    good = ~np.isnan(y)

    ret_mat[good] = 0
    ret_mat[good, y[good].astype(int)] = 1.0

    return ret_mat


def PET_conv(value):
    """Convert PET measures from string to float"""
    try:
        return float(value.strip().strip(">"))
    except ValueError:
        return float("NaN")


def Diagnosis_conv(value):
    """Convert diagnosis from string to float"""
    if value == "CN":
        return 0.0
    if value == "MCI":
        return 1.0
    if value == "AD":
        return 2.0
    return float("NaN")


def DX_conv(value):
    """Convert change in diagnosis from string to float"""
    if isinstance(value, str):
        if value.endswith("Dementia"):
            return 2.0
        if value.endswith("MCI"):
            return 1.0
        if value.endswith("NL"):
            return 0.0

    return float("NaN")


def add_ci_col(values, ci, lo, hi):
    """Add lower/upper confidence interval to prediction"""
    return np.clip(np.vstack([values, values - ci, values + ci]).T, lo, hi)


def censor_d1_table(_table, site):
    """Remove problematic rows"""

    # Add errors="ignore" to avoid KeyError when _table is short (unit_test/example)
    if site == "ADNI":
        _table.drop(
            2984, inplace=True, errors="ignore"
        )  # RID 2190, Month = m03, Month_bl = 0.45
        _table.drop(
            4703, inplace=True, errors="ignore"
        )  # RID 4579, Month = m03, Month_bl = 0.32
        _table.drop(
            14415, inplace=True, errors="ignore"
        )  # RID 6014, Month = m0, Month_bl = 0.16
    elif site == "OASIS":
        _table.drop(
            44, inplace=True, errors="ignore"
        )  # RID OAS30004, duplicate Month_bl = 74
        _table.drop(
            3280, inplace=True, errors="ignore"
        )  # RID OAS30484, duplicate Month_bl = 2
        _table.drop(
            6810, inplace=True, errors="ignore"
        )  # RID OAS30970, duplicate Month_bl = 8
        _table.drop(
            8350, inplace=True, errors="ignore"
        )  # RID OAS31203, duplicate Month_bl = 8


def load_table(csv, columns, site):
    """Load CSV, only include *columns*"""
    conv = CONVERTERS if site != "OASIS" else CONVERTERS_OASIS
    table = pd.read_csv(
        csv, converters=conv, usecols=columns, low_memory=False
    )

    censor_d1_table(table, site)
    return table


# Converters for columns with non-numeric values
CONVERTERS_OASIS = {}
CONVERTERS = {
    "SCANDATE": str2date,
    "EXAMDATE": str2date,
    "Diagnosis": Diagnosis_conv,
    "PTAU_UPENNBIOMK9_04_19_17": PET_conv,
    "TAU_UPENNBIOMK9_04_19_17": PET_conv,
    "ABETA_UPENNBIOMK9_04_19_17": PET_conv,
}


def get_baseline_prediction_start(frame, site):
    """Get baseline dates and dates when prediction starts"""
    one_month = relativedelta(months=1) if site != "OASIS" else 1
    baseline = {}
    start = {}
    for subject in np.unique(frame.RID):
        dates = frame.loc[frame.RID == subject, "EXAMDATE"]
        baseline[subject] = min(dates)
        start[subject] = max(dates) + one_month

    return baseline, start


def get_mask(csv_path, use_validation, site, in_domain):
    """Get masks from CSV file"""
    if in_domain:
        columns = ["RID", "EXAMDATE", "train", "val", "test"]
        frame = load_table(csv_path, columns, site)
        train_mask = frame.train == 1
        if use_validation:
            pred_mask = frame.val == 1
        else:
            pred_mask = frame.test == 1
    else:
        columns = ["RID", "EXAMDATE", "test"]
        frame = load_table(csv_path, columns, site)
        train_mask = None
        pred_mask = frame.test == 1

    return train_mask, pred_mask, frame[pred_mask]
