#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


class model_config:
    """
    Store the necessary configurations for AD-Map to function
    """

    # columns to load while reading csv
    use_columns = [
        "RID",
        "MMSE",
        "CDR",
        "Ventricles",
        "Hippocampus",
        "Fusiform",
        "MidTemp",
        "WholeBrain",
        "ICV",
    ]

    # whether feature will increase / decrease as time goes by
    mri_features = {
        "Hippocampus": False,
        "Ventricles": True,
        "Fusiform": False,
        "MidTemp": False,
        "WholeBrain": False,
    }

    # columns to use when generating leapsy data
    feature_columns = [
        "ID",
        "TIME",
        "MMSE",
        "CDR",
        "Ventricles",
        "Hippocampus",
        "Fusiform",
        "MidTemp",
        "WholeBrain",
    ]

    # empty columns to add so we don't need to modify submission evaluation
    extra_sub_columns = [
        "CN relative probability",
        "MCI relative probability",
        "AD relative probability",
        "MMSE 50% CI lower",
        "MMSE 50% CI upper",
        "Ventricles_ICV 50% CI lower",
        "Ventricles_ICV 50% CI upper",
    ]

    # extra columns to add so we don't need to modify submission evaluation
    extra_sub_dx = [
        "MMSE 50% CI lower",
        "MMSE 50% CI upper",
        "Ventricles_ICV 50% CI lower",
        "Ventricles_ICV 50% CI upper",
    ]
