#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


class model_config:
    """
    Store the necessary configurations for L2C-XGBw/nw to function
    """

    # Columns to exclude from being predictors
    # 'rid' and 'ptid' are participant IDs
    # Target variables ('curr_dx', 'curr_mmse') and other potential leaks are excluded
    not_predictors = [
        "ptid",
        "rid",
        "curr_dx",
        "curr_mmse",
        "curr_ventricles",
        "curr_icv",
    ]

    # For the test/validation set, 'forecast_month' is metadata, not a predictor itself
    not_predictors_test = ["ptid", "rid", "forecast_month"]

    # If you plan to ensemble multiple XGBoost models (e.g., with different seeds),
    # this variable would control that. For now, sticking to 1 model.
    num_ensemble_models = 1

    # Number of boosted runs are different for different targets
    num_boost_round_dict = {
        "clin": 500,
        "mmse": 750,
        "mmse_CN": 350,
        "vent": 900,
    }

    # Minimum sample requirement with CN or MCI diagnosis for training CN specific model
    min_cn_mci_samples = 1200

    # Partition boundaries for L2C-XGBw
    partition_dict = {
        "clin": [0, 8, 15, 27, 39, 60, 500],
        "mmse": [0, 9, 15, 27, 39, 54, 500],
        "vent": [0, 9, 15, 30, 500],
    }
