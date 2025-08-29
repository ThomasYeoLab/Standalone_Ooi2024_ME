#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Chen Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import warnings

import numpy as np
import pandas as pd
from utils.MAUC import MAUC

# Suppress RuntimeWarnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def calcBCA(estimLabels, trueLabels, nrClasses):

    # Balanced Classification Accuracy
    bcaAll = []
    for c0 in range(nrClasses):
        for c1 in range(c0 + 1, nrClasses):
            # c0 = positive class  &  c1 = negative class
            TP = np.sum((estimLabels == c0) & (trueLabels == c0))
            TN = np.sum((estimLabels == c1) & (trueLabels == c1))
            FP = np.sum((estimLabels == c1) & (trueLabels == c0))
            FN = np.sum((estimLabels == c0) & (trueLabels == c1))

            # sometimes the sensitivity of specificity can be NaN, if the user doesn't forecast one of the classes.
            # In this case we assume a default value for sensitivity/specificity
            if (TP + FN) == 0:
                sensitivity = 0.5
            else:
                sensitivity = (1.0 * TP) / (TP + FN)

            if (TN + FP) == 0:
                specificity = 0.5
            else:
                specificity = (1.0 * TN) / (TN + FP)

            bcaCurr = 0.5 * (sensitivity + specificity)
            bcaAll += [bcaCurr]
            # print('bcaCurr %f TP %f TN %f FP %f FN %f' % (bcaCurr, TP, TN, FP, FN))

    return np.mean(bcaAll)


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
        Various arrays for prediction and ground truth for each individual's future visits
    """

    trueDiag = d4Df["Diagnosis"]
    trueMMSE = d4Df["MMSE"]
    trueVents = d4Df["Ventricles"]

    nrSubj = d4Df.shape[0]

    zipTrueLabelAndProbs = []

    hardEstimClass = -1 * np.ones(nrSubj, int)
    mmseEstim = -1 * np.ones(nrSubj, float)
    mmseEstimLo = -1 * np.ones(nrSubj, float)  # lower margin
    mmseEstimUp = -1 * np.ones(nrSubj, float)  # upper margin
    ventriclesEstim = -1 * np.ones(nrSubj, float)
    ventriclesEstimLo = -1 * np.ones(nrSubj, float)  # lower margin
    ventriclesEstimUp = -1 * np.ones(nrSubj, float)  # upper margin

    invalidFlag = False

    # for each subject in D4 match the closest user forecasts
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

        timeDiffsScanCog = [
            d4Df["CognitiveAssessmentDate"].iloc[s] - d
            for d in currSubjData["Forecast Date"]
        ]
        # print('Forecast Date 2',currSubjData['Forecast Date'])
        indexMin = np.argsort(np.abs(timeDiffsScanCog))[0]
        # print('timeDiffsScanMri', indexMin, timeDiffsScanMri)

        pCN = currSubjData["CN relative probability"].iloc[indexMin]
        pMCI = currSubjData["MCI relative probability"].iloc[indexMin]
        pAD = currSubjData["AD relative probability"].iloc[indexMin]

        # normalise the relative probabilities by their sum
        pSum = (pCN + pMCI + pAD) / 3
        pCN /= pSum
        pMCI /= pSum
        pAD /= pSum

        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])

        mmseEstim[s] = currSubjData["MMSE"].iloc[indexMin]
        mmseEstimLo[s] = currSubjData["MMSE 50% CI lower"].iloc[indexMin]
        mmseEstimUp[s] = currSubjData["MMSE 50% CI upper"].iloc[indexMin]

        # for the mri scan find the forecast closest to the scan date,
        # which might be different from the cognitive assessment date
        timeDiffsScanMri = [
            d4Df["ScanDate"].iloc[s] - d for d in currSubjData["Forecast Date"]
        ]

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

        ventriclesEstim[s] = currSubjData["Ventricles_ICV"].iloc[indexMinMri]
        ventriclesEstimLo[s] = currSubjData[
            "Ventricles_ICV 50% CI lower"
        ].iloc[indexMinMri]
        ventriclesEstimUp[s] = currSubjData[
            "Ventricles_ICV 50% CI upper"
        ].iloc[indexMinMri]

        if not np.isnan(trueDiag.iloc[s]):
            zipTrueLabelAndProbs += [(trueDiag.iloc[s], [pCN, pMCI, pAD])]

    if invalidFlag:
        if model != "AD_Map":  # relax this requirement for AD_Map
            # if at least one subject was missing or if
            raise ValueError("Submission was incomplete. Please resubmit")

    # If there are NaNs in D4, filter out them along with the corresponding user forecasts
    # This can happen if rollover subjects don't come for visit in ADNI3.
    notNanMaskDiag = np.logical_not(np.isnan(trueDiag))
    trueDiagFilt = trueDiag[notNanMaskDiag]
    hardEstimClassFilt = hardEstimClass[notNanMaskDiag]

    notNanMaskMMSE = np.logical_not(np.isnan(trueMMSE))
    trueMMSEFilt = trueMMSE[notNanMaskMMSE]
    mmseEstim = mmseEstim[notNanMaskMMSE]
    mmseEstimLo = mmseEstimLo[notNanMaskMMSE]
    mmseEstimUp = mmseEstimUp[notNanMaskMMSE]

    notNanMaskVents = np.logical_not(np.isnan(trueVents))
    trueVentsFilt = trueVents[notNanMaskVents]
    ventriclesEstim = ventriclesEstim[notNanMaskVents]
    ventriclesEstimLo = ventriclesEstimLo[notNanMaskVents]
    ventriclesEstimUp = ventriclesEstimUp[notNanMaskVents]

    assert trueDiagFilt.shape[0] == hardEstimClassFilt.shape[0]
    assert (
        trueMMSEFilt.shape[0]
        == mmseEstim.shape[0]
        == mmseEstimLo.shape[0]
        == mmseEstimUp.shape[0]
    )
    assert (
        trueVentsFilt.shape[0]
        == ventriclesEstim.shape[0]
        == ventriclesEstimLo.shape[0]
        == ventriclesEstimUp.shape[0]
    )

    return (
        zipTrueLabelAndProbs,
        hardEstimClassFilt,
        mmseEstim,
        mmseEstimLo,
        mmseEstimUp,
        ventriclesEstim,
        ventriclesEstimLo,
        ventriclesEstimUp,
        trueDiagFilt,
        trueMMSEFilt,
        trueVentsFilt,
    )


def evalOneSub(d4Df, forecastDf, site, model=None):
    """Evaluates one submission

    Args:
        d4Df: dataframe containing the ground truth
        forecastDf: dataframe containing the future predictions
        site: test dataset name
        model: the model that was used to generate the predictions

    Returns:
        result: a dictionary containing the following metrics:
            mAUC - multiclass Area Under Curve
            bca - balanced classification accuracy
            mmseMAE - MMSE Mean Aboslute Error
            ventsMAE - Ventricles Mean Aboslute Error
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

    (
        zipTrueLabelAndProbs,
        hardEstimClass,
        mmseEstim,
        mmseEstimLo,
        mmseEstimUp,
        ventriclesEstim,
        ventriclesEstimLo,
        ventriclesEstimUp,
        trueDiagFilt,
        trueMMSEFilt,
        trueVentsFilt,
    ) = parseData(d4Df, forecastDf, site, model)
    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)

    # >>>>>>>>> compute metrics for the clinical status <<<<<<<<<<

    # Multiclass AUC (mAUC)
    true_labels = np.array([val[0] for val in zipTrueLabelAndProbs])
    nrClasses = np.unique(true_labels).shape[0]
    mAUC = MAUC(zipTrueLabelAndProbs, num_classes=nrClasses)

    # Balanced Classification Accuracy (BCA)
    trueDiagFilt = trueDiagFilt.astype(int)
    bca = calcBCA(hardEstimClass, trueDiagFilt, nrClasses=nrClasses)

    # compute metrics for Ventricles and MMSE (MAE)
    mmseMAE = np.mean(np.abs(mmseEstim - trueMMSEFilt))
    ventsMAE = np.mean(np.abs(ventriclesEstim - trueVentsFilt))

    result = {
        "mAUC": mAUC,
        "bca": bca,
        "mmseMAE": mmseMAE,
        "ventsMAE": ventsMAE,
    }
    return result
