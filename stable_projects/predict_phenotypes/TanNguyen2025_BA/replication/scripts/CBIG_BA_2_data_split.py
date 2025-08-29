'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script performs train-validation-test data splitting.
It can perform splitting for the full training & validation set size
(i.e., -sample_size "full") or the varied training & validation set sizes
(i.e., -sample_size "vary"), for each train-validation-test split
(i.e., --num_split), for each classification task (i.e., --task).

By default, splits for both `ad_classification` & `mci_progression`
tasks, for both `full` and `vary` sample sizes, and for all 50
train-val-test data splits.

Expected output(s):
1. $TANNGUYEN2025_BA_DIR/data/data_split

Example:
    conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR};
    python -m replication.scripts.CBIG_BA_2_data_split;
'''

import os
import sys
import subprocess
import argparse
TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError("ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from CBIG_BA_config import global_config
from utils.CBIG_BA_complete_step import generate_complete_flag


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run data splitting.")
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Specify task: ad_classification, mci_progression, or both")
    parser.add_argument("-s",
                        "--sample_size",
                        type=str,
                        default=None,
                        help="Specify sample size: full, vary")
    parser.add_argument("-n",
                        "--num_split",
                        type=int,
                        default=50,
                        help="Specify the number of splits (default is 50)")
    args = parser.parse_args()

    task = args.task
    sample_size = args.sample_size
    num_split = args.num_split

    # Check if task is provided, otherwise default to both tasks
    if task is None:
        task = "ad_classification mci_progression"
    elif task not in [
            "ad_classification", "mci_progression",
            "ad_classification mci_progression"
    ]:
        print(
            "ERROR: task must include only ad_classification or mci_progression"
        )
        sys.exit(1)

    # Setup ROOT_DIR from CBIG_BA_config.py
    ROOTDIR = global_config.ROOT_DIR
    os.chdir(ROOTDIR)

    # Default to running both 'full' and 'vary' if no sample_size is provided
    if sample_size is None:
        sample_size = "full vary"

    # Call relevant scripts for each task
    for i in task.split():
        # Check for sample_size argument and run accordingly
        if "full" in sample_size:
            print(f"Running data split for full dataset for {i}...")
            subprocess.run(["python", "-m", "data_split.CBIG_BA_data_split_full", \
                "--task", i, "--num_split", str(num_split)], check=True)
            print("")

        if "vary" in sample_size:
            print(f"Running data split for sample sizes experiment for {i}...")
            if i == "ad_classification":
                full_trainval_size = "997"
            elif i == "mci_progression":
                full_trainval_size = "448"
            else:
                print(f"ERROR: {i} value NOT accepted as task!")
                sys.exit(1)

            subprocess.run([
                "python", "-m", "data_split.CBIG_BA_data_split_vary", "--task",
                i, "--num_split",
                str(num_split), "--full_trainval_size", full_trainval_size
            ],
                           check=True)

    # Generate complete flags for next step to commence
    generate_complete_flag(os.path.join(ROOTDIR, 'data'),
                           append='2_data_split')

    print("Finished data splitting!")

if __name__ == "__main__":
    main()
