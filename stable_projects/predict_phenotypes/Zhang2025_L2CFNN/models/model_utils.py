#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from config import global_config


def _calc_score(mauc, bca, mmse, vent):
    """
    Calculate trial score for hyperparameter tuning
    Based on 4 metrics, lower is better
    """
    total = -mauc - bca + mmse + vent
    return total if np.isfinite(total) else 100000


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--util", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--site", type=str, default="ADNI")
    return parser


def clear_temporary(args):
    """
    Remove temporary files generated during Optuna hyperparameter tuning
    """

    def clear_helper(model_dir, file_suffix, model):
        for fold in range(20):
            fold_dir = model_dir / f"model.f{fold}"
            config_file = fold_dir / "config.json"

            try:
                config = json.load(config_file.open())
                trial = config["best_trial"]
            except FileNotFoundError:
                # print(f"Warning: config.json not found in {fold_dir}. Skipping fold {fold}.")
                continue
            except json.JSONDecodeError:
                print(
                    f"Warning: Error decoding config.json in {config_file}. Skipping fold {fold}."
                )
                continue
            except KeyError:
                print(
                    f"Warning: 'best_trial' key not found in {config_file}. Skipping fold {fold}."
                )
                continue

            for file_path in fold_dir.iterdir():
                # Check if the path is a directory, AD_Map has temporary folders to delete
                if file_path.is_dir():
                    try:
                        shutil.rmtree(
                            file_path
                        )  # <-- Recursively delete the directory
                    except OSError as e:
                        # Catch potential errors during deletion (e.g., permission issues)
                        print(
                            f"  Error deleting directory {file_path.name}: {e}"
                        )
                    # Continue to the next item in fold_dir after handling the directory
                    continue

                # If it's not a directory, proceed with file checks (safer to check is_file())
                elif file_path.is_file():
                    if model == "AD_Map":
                        if file_path.name.endswith("_log.json"):
                            expected_log_name = f"trial{trial}_log.json"  # log files are like trialX_log.json
                            if file_path.name != expected_log_name:
                                file_path.unlink()
                        else:
                            if file_path.name.startswith("trial"):
                                expected_model_name = f"trial{trial}.json"  # for AD_Map, model is saved as trialX.json
                                if file_path.name != expected_model_name:
                                    file_path.unlink()
                    else:
                        if file_path.suffix == file_suffix:
                            file_trial = int(file_path.stem[5:])
                            if file_trial != trial:
                                file_path.unlink()
                        if file_path.name.endswith("_log.json"):
                            expected_log_name = f"trial{trial}_log.json"  # log files are like trialX_log.json
                            if file_path.name != expected_log_name:
                                file_path.unlink()

    if args.model in ["L2C_XGBw", "L2C_XGBnw"]:
        for metric in ["clin", "mmse", "vent"]:
            model_dir = (
                Path(global_config.checkpoint_dir)
                / args.model
                / args.site
                / metric
            )
            clear_helper(model_dir, ".pkl", args.model)
    else:
        model_dir = Path(global_config.checkpoint_dir) / args.model / args.site
        clear_helper(model_dir, ".pt", args.model)


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    if args.util == "clear_temp":
        clear_temporary(args)
