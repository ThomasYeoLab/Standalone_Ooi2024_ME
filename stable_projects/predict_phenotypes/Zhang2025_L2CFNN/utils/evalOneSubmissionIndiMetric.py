#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Chen Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""


import numpy as np
import pandas as pd

"""
This version computes the accuracy and MAE for each individual separately
This way, we can perform the significance tests (pair-wise, individual level)
"""


def parseData(d4Df, forecastDf, site, model):
    """Parse prediction and ground truth dataframes into a format
        suitable for metric calculation.
        Recall we predict 120 months into the future. To evaluate the performance
        at a specific future ground timepoint, we need to find the prediction
        closest in time to the ground truth date.

    Args:
        d4Df: dataframe containing the ground truth
        forecastDf: dataframe containing the future predictions
        site: test dataset name
        model: the model that was used to generate the predictions

    Return:
        subj_dict: a dictionary containing the RID, EXAMDATE, SCANDATE,
        as well as the prediction and ground truth for each individual's future visits
    """
    nrSubj = d4Df.shape[0]

    invalidFlag = False
    # for each subject in D4 match the closest user forecasts
    subj_dict = {
        "RID": [],
        "EXAMDATE": [],
        "SCANDATE": [],
        "DX_pred": [],
        "DX_gt": [],
        "VENT_pred": [],
        "VENT_gt": [],
        "MMSE_pred": [],
        "MMSE_gt": [],
    }
    for s in range(nrSubj):
        currSubjMask = d4Df["RID"].iloc[s] == forecastDf["RID"]
        currSubjData = forecastDf[currSubjMask]

        # if subject is missing
        if currSubjData.shape[0] == 0:
            print(
                "WARNING: Subject RID %s missing from user forecasts"
                % d4Df["RID"].iloc[s]
            )
            invalidFlag = True
            continue

        # if not all forecast months are present
        if (
            currSubjData.shape[0] < 5 * 12
        ):  # check if at least 5 years worth of forecasts exist
            print(
                "WARNING: Missing forecast months for subject with RID %s"
                % d4Df["RID"].iloc[s]
            )
            invalidFlag = True
            continue

        currSubjData = currSubjData.reset_index(drop=True)
        subj_dict["RID"].append(d4Df["RID"].iloc[s])
        timeDiffsScanCog = [
            d4Df["CognitiveAssessmentDate"].iloc[s] - d
            for d in currSubjData["Forecast Date"]
        ]
        subj_dict["EXAMDATE"].append(d4Df["CognitiveAssessmentDate"].iloc[s])
        indexMin = np.argsort(np.abs(timeDiffsScanCog))[0]

        pCN = currSubjData["CN relative probability"].iloc[indexMin]
        pMCI = currSubjData["MCI relative probability"].iloc[indexMin]
        pAD = currSubjData["AD relative probability"].iloc[indexMin]

        # normalise the relative probabilities by their sum
        pSum = (pCN + pMCI + pAD) / 3
        pCN /= pSum
        pMCI /= pSum
        pAD /= pSum

        subj_dict["DX_pred"].append(np.argmax([pCN, pMCI, pAD]))
        subj_dict["DX_gt"].append(d4Df["Diagnosis"].iloc[s])

        subj_dict["MMSE_pred"].append(currSubjData["MMSE"].iloc[indexMin])
        subj_dict["MMSE_gt"].append(d4Df["MMSE"].iloc[s])

        # for the mri scan find the forecast closest to the scan date,
        # which might be different from the cognitive assessment date
        timeDiffsScanMri = [
            d4Df["ScanDate"].iloc[s] - d for d in currSubjData["Forecast Date"]
        ]
        subj_dict["SCANDATE"].append(d4Df["ScanDate"].iloc[s])
        if site == "OASIS":
            if np.isnan(d4Df["ScanDate"].iloc[s]):
                indexMinMri = indexMin
            else:
                indexMinMri = np.argsort(np.abs(timeDiffsScanMri))[0]
        else:
            if d4Df["ScanDate"].iloc[s] is pd.NaT:
                indexMinMri = indexMin
            else:
                indexMinMri = np.argsort(np.abs(timeDiffsScanMri))[0]

        subj_dict["VENT_pred"].append(
            currSubjData["Ventricles_ICV"].iloc[indexMinMri]
        )
        subj_dict["VENT_gt"].append(d4Df["Ventricles"].iloc[s])

    if invalidFlag:
        if model != "AD_Map":  # relax this requirement for AD_Map
            # if at least one subject was missing or if
            raise ValueError("Submission was incomplete. Please resubmit")

    return subj_dict


def evalOneSubNew(d4Df, forecastDf, site, model=None):
    """Evaluates one submission. Calculation was performed
        through Pandas on a dataframe generated from the
        dictionary returned by parseDate() function.

    Args:
        d4Df: dataframe containing the ground truth
        forecastDf: dataframe containing the future predictions
        site: test dataset name
        model: the model that was used to generate the predictions

    Returns:
        result: a dictionary containing the following metrics:
            dxACC - subject level diagnosis prediction accuracy
            ventMAE - subject level Ventricles Mean Aboslute Error
            mmseMAE - subject level MMSE Mean Aboslute Error
        subj_metric: a dict containing the raw correctness of diagnosis
            and raw error between ground truth and prediction for each
            individual's future visits (before averaging to get the final metrics)
        subj_dfs: a dict containing the raw dataframes generated from
            the dictionary returned by parseDate() function. Mainly for debugging.
    """
    if model == "AD_Map":
        # Cast ID type to string to prevent leapsy bug
        d4Df["RID"] = d4Df["RID"].astype(str)
    if site != "OASIS":
        forecastDf["Forecast Date"] = pd.to_datetime(
            forecastDf["Forecast Date"]
        )
        d4Df["CognitiveAssessmentDate"] = pd.to_datetime(
            d4Df["CognitiveAssessmentDate"]
        )
        d4Df["ScanDate"] = pd.to_datetime(d4Df["ScanDate"])
    if isinstance(d4Df["Diagnosis"].iloc[0], str):
        mapping = {"CN": 0, "MCI": 1, "AD": 2}
        d4Df.replace({"Diagnosis": mapping}, inplace=True)

    subj_dict = parseData(d4Df, forecastDf, site, model)
    subj_df = pd.DataFrame(subj_dict)

    # DX
    dx_df = subj_df.loc[
        subj_df.DX_gt.notnull(), ["RID", "DX_pred", "DX_gt"]
    ].copy()
    dx_df["correct"] = dx_df["DX_pred"] == dx_df["DX_gt"]
    dx_return = pd.DataFrame(
        dx_df.groupby("RID")["correct"].mean()
    ).reset_index()

    # MMSE
    mmse_df = subj_df.loc[
        subj_df.MMSE_gt.notnull(), ["RID", "MMSE_pred", "MMSE_gt"]
    ].copy()
    mmse_df["diff"] = (mmse_df["MMSE_pred"] - mmse_df["MMSE_gt"]).abs()
    mmse_return = pd.DataFrame(
        mmse_df.groupby("RID")["diff"].mean()
    ).reset_index()

    # VENT
    vent_df = subj_df.loc[
        subj_df.VENT_gt.notnull(), ["RID", "VENT_pred", "VENT_gt"]
    ].copy()
    vent_df["diff"] = (vent_df["VENT_pred"] - vent_df["VENT_gt"]).abs()
    vent_return = pd.DataFrame(
        vent_df.groupby("RID")["diff"].mean()
    ).reset_index()

    result = {
        "dxACC": dx_return["correct"].mean(),
        "mmseMAE": mmse_return["diff"].mean(),
        "ventsMAE": vent_return["diff"].mean(),
    }

    subj_metric = {
        "dxACC": dx_return["correct"].values,
        "mmseMAE": mmse_return["diff"].values,
        "ventsMAE": vent_return["diff"].values,
    }
    # return result, subj_metric

    subj_dfs = {
        "dxACC": dx_return,
        "mmseMAE": mmse_return,
        "ventsMAE": vent_return,
    }
    return result, subj_metric, subj_dfs
