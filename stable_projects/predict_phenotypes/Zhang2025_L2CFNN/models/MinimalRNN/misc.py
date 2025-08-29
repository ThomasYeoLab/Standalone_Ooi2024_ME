#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def build_pred_frame(prediction, outpath=""):
    """
    Construct the forecast spreadsheet following TADPOLE format
    Args:
        prediction (dictionary): contains the following key/value pairs:
            dates: dates of predicted timepoints for each subject
            subjects: list of subject IDs
            DX: list of diagnosis prediction for each subject
            ADAS13: list of ADAS13 prediction for each subject
            Ventricles: list of ventricular volume prediction for each subject
        outpath (string): where to save the prediction frame
        If *outpath* is blank, the prediction frame is not saved
    Return:
        (Pandas data frame): prediction frame
    """
    table = pd.DataFrame()
    dates = prediction["dates"]
    table["RID"] = np.repeat(prediction["subjects"], [len(x) for x in dates])
    table["Forecast Month"] = np.concatenate(
        [np.arange(len(x)) + 1 for x in dates]
    )
    table["Forecast Date"] = np.concatenate(dates)

    diag = np.concatenate(prediction["DX"])
    table["CN relative probability"] = diag[:, 0]
    table["MCI relative probability"] = diag[:, 1]
    table["AD relative probability"] = diag[:, 2]

    mmse = np.concatenate(prediction["MMSE"])
    table["MMSE"] = mmse[:, 0]
    table["MMSE 50% CI lower"] = mmse[:, 1]
    table["MMSE 50% CI upper"] = mmse[:, 2]

    vent = np.concatenate(prediction["Ventricles"])
    table["Ventricles_ICV"] = vent[:, 0]
    table["Ventricles_ICV 50% CI lower"] = vent[:, 1]
    table["Ventricles_ICV 50% CI upper"] = vent[:, 2]

    assert len(diag) == len(mmse) == len(vent)
    if outpath:
        table.to_csv(outpath, index=False)

    return table


def make_date_col(starts, duration, site):
    """
    Return a list of list of dates
    The start date of each list of dates is specified by *starts*
    """
    if site.split("_")[0] == "OASIS":
        date_range = list(range(duration))
    else:
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


def add_ci_col(values, ci, lo, hi):
    """Add lower/upper confidence interval to prediction"""
    return np.clip(np.vstack([values, values - ci, values + ci]).T, lo, hi)
