#!/usr/bin/env python
# Create the baseline for each patients in the dataset. One patient
# per row. This is used for generating the long and d4 csv files later.
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


import argparse
from os import makedirs, path

import data_processing.misc as misc
import numpy as np
import pandas as pd

from config import global_config


def get_constant_feature(const_frame, fields, site):
    """
    Get cross-sectional features for each subject.
    Features are in order [subid, visit, age, edu, gender, maritalsta, APOE]
    They are [ordinal, ordinal, cat, cat, cat, cat] in nature.
    AIBL doesn't have education and marital_status from ADNI LONI.
    """
    if site == "OASIS":
        baseConst = const_frame[const_frame.EXAMDATE == 0]
    elif site == "MACC":
        baseConst = const_frame[const_frame.VISCODE == "BL"]
    else:  # for ADNI, AIBL, and fakeADNI, all follow this
        baseConst = const_frame[const_frame.VISCODE == "bl"]

    return baseConst[fields]


def main(args):
    """Main execution function."""

    np.random.seed(args.seed)

    # load spreadsheet using proper converters
    if args.site == "OASIS":
        marry_conv = misc.OASIS_marriage_conv
    elif args.site == "MACC":
        marry_conv = misc.MACC_marriage_conv
    elif args.site == "AIBL":
        marry_conv = misc.AIBL_marriage_conv
    else:  # for ADNI and fakeADNI, use this
        marry_conv = misc.ADNI_marriage_conv

    CONST_CONVERTERS = {
        "APOE": (
            misc.APOE_conv if args.site in ("AIBL", "OASIS", "MACC") else None
        ),
        "PTGENDER": misc.Gender_conv,
        "PTMARRY": marry_conv,
    }

    frame = pd.read_csv(
        f"{global_config.raw_data_path}/{args.site}.csv",
        converters=CONST_CONVERTERS,
        low_memory=False,
    )

    if args.site != "OASIS":
        fields = ["RID", "VISCODE", "PTEDUCAT", "PTGENDER", "PTMARRY", "APOE"]
    else:
        fields = ["RID", "EXAMDATE", "PTEDUCAT", "PTGENDER", "PTMARRY", "APOE"]

    const_feat = get_constant_feature(frame, fields, args.site)

    if args.site != "OASIS":
        patient_baselines = const_feat.drop("VISCODE", axis=1)
    else:
        patient_baselines = const_feat.drop("EXAMDATE", axis=1)
    patient_baselines.rename(
        {
            "RID": "patients",
            "PTEDUCAT": "educ",
            "PTGENDER": "isMale",
            "PTMARRY": "married",
        },
        axis=1,
        inplace=True,
    )

    # Generate the save directory
    save_path = path.join(global_config.baseline_path, f"{args.site}")
    makedirs(save_path, exist_ok=True)

    patient_baselines.to_csv(
        path.join(save_path, "patient_baselines.csv"), index=False
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--site", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
