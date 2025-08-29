#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import argparse
import os
import random
import shutil
import subprocess

import numpy as np

from config import global_config


class ExampleConfigs:
    """
    Example configuration class with default values.
    Users can override values by passing arguments to the constructor.
    Note:
        All values are stored as strings since they are passed directly to
        subprocess.call() as command-line arguments, which require string inputs.
    """

    def __init__(
        self,
        site="SampleData",
        n_ensemble="20",
        seed="3",
        batch_size="512",
        input_dim="101",
        num_folds="1",
        device="None",
        total_count="None",
    ):
        self.site = site
        self.n_ensemble = n_ensemble
        self.seed = seed
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_folds = num_folds
        self.device = device
        self.total_count = total_count


def test_data_processing(example_configs):
    """
    function to process test data
    """

    # Step 1: split generation
    subprocess.call(
        [
            "python",
            "-m",
            "data_processing.step1_fold_split",
            "--site",
            example_configs.site,
            "--independent_test",
        ]
    )
    print(">>>>>>>>> Data processing step 1 finished")

    # Step 2: baseline generation
    subprocess.call(
        [
            "python",
            "-m",
            "data_processing.step2_baseline_gen",
            "--site",
            example_configs.site,
        ]
    )
    print(">>>>>>>>> Data processing step 2 finished")

    # Step 3: data per fold generation
    subprocess.call(
        [
            "python",
            "-m",
            "data_processing.step3_fold_gen",
            "--site",
            example_configs.site,
            "--independent_test",
        ]
    )
    print(">>>>>>>>> Data processing step 3 finished")

    # Step 4: L2C feature generation
    subprocess.call(
        [
            "python",
            "-m",
            "data_processing.step4_frog_gen",
            "--site",
            example_configs.site,
            "--independent_test",
        ]
    )
    print(">>>>>>>>> Data processing step 4 finished")

    # Step 5: process L2C features for L2C-FNN (N folds for N ensembles)
    for fold in range(int(example_configs.n_ensemble)):
        subprocess.call(
            [
                "python",
                "-m",
                "data_processing.step5_predict_only_data_process",
                "--site",
                example_configs.site,
                "--fold",
                f"{fold}",
                "--independent_test",
            ]
        )
    print(">>>>>>>>> Data processing step 5 finished")

    # Step 6: filter out unused ground truth data
    subprocess.call(
        [
            "python",
            "-m",
            "data_processing.step6_gt_filter",
            "--site",
            example_configs.site,
        ]
    )
    print(">>>>>>>>> Data processing step 6 finished")


def test_L2C_FNN(example_configs):
    """
    function to test the performance of L2C_FNN ensemble performance
    """
    subprocess.call(
        [
            "python",
            "-m",
            "models.L2C_FNN.predict_only_evaluation",
            "--site",
            example_configs.site,
            "--batch_size",
            example_configs.batch_size,
            "--input_dim",
            example_configs.input_dim,
            "--n_ensemble",
            str(example_configs.n_ensemble),
            "--device",
            example_configs.device,
        ]
    )
    print(">>>>>>>>> L2C-FNN evaluation finished")


def plot_individual(example_configs):
    """
    function to test the performance of L2C_FNN ensemble performance
    """

    p1 = subprocess.Popen(
        [
            "python",
            "-m",
            "predict_only.plotting_utils",
            "--site",
            example_configs.site,
            "--total_count",
            example_configs.total_count,
            "--target",
            "MMSE",
        ]
    )

    p2 = subprocess.Popen(
        [
            "python",
            "-m",
            "predict_only.plotting_utils",
            "--site",
            example_configs.site,
            "--total_count",
            example_configs.total_count,
            "--target",
            "VENT",
        ]
    )

    p3 = subprocess.Popen(
        [
            "python",
            "-m",
            "predict_only.plotting_utils",
            "--site",
            example_configs.site,
            "--total_count",
            example_configs.total_count,
            "--target",
            "DX",
        ]
    )

    # Wait for all 3 training processes to finish
    p1.wait()
    p2.wait()
    p3.wait()

    print(">>>>>>>>> Individual trajectory plotting finished")


def check_results(example_configs):
    """
    function to compare example run prediction against reference results
    """

    subprocess.call(
        [
            "python",
            "-m",
            "utils.check_reference_results",
            "--models",
            "L2C_FNN",
            "--site",
            example_configs.site,
            "--num_folds",
            example_configs.num_folds,
            "--predict_only",
        ]
    )
    print(">>>>>>>>> Results comparison finished")


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n_ensemble", type=int, default=20)
    parser.add_argument("--site", type=str, default="SampleData")
    parser.add_argument("--batch_size", type=int, default="512")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot individual trajectories",
    )
    parser.add_argument(
        "--total_count",
        default="None",
        type=str,
        help="Total number of individuals to plot (e.g., 10 or 'all')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="None",
        help="Device to use: e.g., cpu, cuda, cuda:0, mps",
    )
    return parser


def example_wrapper(args):
    """
    example wrapper function
    """
    # Modify site name slightly to avoid conflict
    if args.site in ("AIBL", "MACC", "OASIS"):
        ori_data_name = global_config.raw_data_path / f"{args.site}.csv"
        new_data_name = global_config.raw_data_path / f"My{args.site}.csv"

        # Only copy if target doesn't already exist
        if not os.path.exists(new_data_name):
            try:
                shutil.copy(ori_data_name, new_data_name)
                print(
                    f"Copied {args.site}.csv to My{args.site}.csv to avoid potential namespace conflict"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Original file not found: {ori_data_name}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to copy file: {e}")
        else:
            print(f"{new_data_name} already exists. Skipping copy.")

        args.site = f"My{args.site}"

    # Ensure n_ensemble is valid
    assert (args.n_ensemble > 0) and (
        args.n_ensemble <= 20
    ), "Please make sure n_ensemble is within 1-20"

    # Ensure batch_size is valid
    assert args.batch_size > 0, "batch_size must be a positive integer."

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    example_configs = ExampleConfigs(
        site=args.site,
        n_ensemble=str(args.n_ensemble),
        seed=str(args.seed),
        batch_size=str(args.batch_size),
        device=args.device,
        total_count=args.total_count,
    )

    # step 1
    print(">>>>>> Start data processing <<<<<<")
    test_data_processing(example_configs)

    # step 2
    print(">>>>>> Start L2C-FNN train and evaluation <<<<<<")
    test_L2C_FNN(example_configs)

    # step 3
    if args.plot:
        print(">>>>>> Start plotting individual trajectories <<<<<<")
        plot_individual(example_configs)

    # step 4
    if args.site == "SampleData":
        # Only compare with reference when using sample data
        print(">>>>>> Compare results against reference <<<<<<")
        check_results(example_configs)


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    example_wrapper(args)
