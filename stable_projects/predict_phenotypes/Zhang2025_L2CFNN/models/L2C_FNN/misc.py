#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import pandas as pd


def build_pred_frame(sub_path, pred, tadpole, csv_path, args=None):
    """Function to structure predictions into proper format for evaluation

    Args:
        sub_path: path to submission_df csv file, placeholder for predictions
        pred: dictionary storing the predictions for each target variable
        tadpole: raw input data, we need the last column which stores the subject RID
        args: controlling additional input arguments if needed

    Returns:
        None
    """
    submission = pd.read_csv(sub_path)
    rid = submission["RID"].unique()
    DX_cols = submission.columns[3:6]

    for ii in rid:
        lab_pred = tadpole[:, -1] == ii
        lab_sub = submission["RID"] == ii

        submission.loc[lab_sub, "MMSE"] = pred["mmse"][lab_pred]
        submission.loc[lab_sub, "MMSE 50% CI lower"] = (
            pred["mmse"][lab_pred] / 2
        )
        submission.loc[lab_sub, "MMSE 50% CI upper"] = (
            pred["mmse"][lab_pred] * 2
        )

        submission.loc[lab_sub, "Ventricles_ICV"] = pred["vent"][lab_pred]
        submission.loc[lab_sub, "Ventricles_ICV 50% CI lower"] = (
            pred["vent"][lab_pred] / 2
        )
        submission.loc[lab_sub, "Ventricles_ICV 50% CI upper"] = (
            pred["vent"][lab_pred] * 2
        )

        submission.loc[lab_sub, DX_cols] = pred["clin"][lab_pred]

    submission.to_csv(csv_path, index=False)


def inverse_transform(df_dict, tgt):
    """
    Predictions for MMSE and ventricles are in normalized scales,
    reverse them to get back original prediction scale
    """
    df = pd.DataFrame(df_dict)
    df = df.sort_values("src")  # sort
    pos = df["tgt"].searchsorted(tgt, side="left")  # search

    N = df.shape[0]
    pos[pos >= N] = N - 1
    pos[pos - 1 <= 0] = 0

    x1 = df["tgt"].values[pos]
    x2 = df["tgt"].values[pos - 1]
    y1 = df["src"].values[pos]
    y2 = df["src"].values[pos - 1]

    relative = (tgt - x2) / (x1 - x2)
    return (1 - relative) * y2 + relative * y1
