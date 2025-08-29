#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Written by Chen Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import os
from pathlib import Path


class global_config:
    # Determine root path: Check environment variable first, then default.
    _default_root = Path(
        __file__
    ).parent  # Directory containing this config.py
    root_path = Path(os.getenv("CBIG_L2C_ROOT_PATH", _default_root)).resolve()
    # print(
    #     f"Config using root path: {root_path} (Set by CBIG_L2C_ROOT_PATH env var or default)"
    # )

    # Define all other paths relative to the chosen root_path
    raw_data_path = root_path / "raw_data"
    split_path = root_path / "data" / "folds_split"
    baseline_path = root_path / "data" / "baselines"
    fold_gen_path = root_path / "data" / "fold_gen"
    frog_path = root_path / "data" / "frog_feature"
    processed_frog = root_path / "data" / "processed_frog"
    clean_gt_path = root_path / "data" / "folds_split" / "clean_gt"
    minrnn_path = root_path / "data" / "minrnn_data"

    checkpoint_dir = root_path / "checkpoints"
    prediction_dir = root_path / "predictions"
    reference_dir = root_path / "ref_results"

    # Only for predict_only
    pretrained_dir = root_path / "pretrained_model"

    feature_dict = {
        "FULL": [
            "ADAS13",
            "MMSE",
            "CDR",
            "FDG",
            "ADAS11",
            "RAVLT_immediate",
            "RAVLT_learning",
            "RAVLT_perc_forgetting",
            "Ventricles",
            "Fusiform",
            "WholeBrain",
            "Hippocampus",
            "MidTemp",
            "ICV",
        ],
        "PARTIAL": [
            "MMSE",
            "CDR",
            "Ventricles",
            "Fusiform",
            "WholeBrain",
            "Hippocampus",
            "MidTemp",
            "ICV",
        ],
        "MinFull": [
            "CDRSB",
            "ADAS11",
            "ADAS13",
            "MMSE",
            "RAVLT_immediate",
            "RAVLT_learning",
            "RAVLT_forgetting",
            "RAVLT_perc_forgetting",
            "MOCA",
            "FAQ",
            "Entorhinal",
            "Fusiform",
            "Hippocampus",
            "ICV",
            "MidTemp",
            "Ventricles",
            "WholeBrain",
            "AV45",
            "FDG",
            "ABETA_UPENNBIOMK9_04_19_17",
            "TAU_UPENNBIOMK9_04_19_17",
            "PTAU_UPENNBIOMK9_04_19_17",
        ],
    }
