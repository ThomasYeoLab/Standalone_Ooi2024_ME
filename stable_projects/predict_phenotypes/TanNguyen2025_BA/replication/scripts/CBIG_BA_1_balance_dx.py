'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script performs age & sex matching between binary class samples.
By default, matches for both `ad_classification` & `mci_progression`
tasks.

Expected output(s):
1. $TANNGUYEN2025_BA_DIR/data/matched

Example:
    conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR};
    python -m replication.scripts.CBIG_BA_1_balance_dx;

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
    # Read input arguments
    parser = argparse.ArgumentParser(description="Run matching for tasks.")
    parser.add_argument("-t", "--task", type=str, help="Specify task: ad_classification, mci_progression, or both")
    args = parser.parse_args()

    task = args.task

    # Check input arguments
    if task is None:
        task = "ad_classification mci_progression"
    elif task not in ["ad_classification", "mci_progression", "ad_classification mci_progression"]:
        print("ERROR: task must include only ad_classification or mci_progression")
        sys.exit(1)

    # Setup - Get ROOT_DIR from CBIG_BA_config.py
    ROOTDIR = global_config.ROOT_DIR
    os.chdir(ROOTDIR)

    # Run the matching for each task
    for t in task.split():
        print(f"Running matching for {t}...")
        try:
            subprocess.run(["python", "-m", "data_split.CBIG_BA_matching", "--task", t], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running matching for {t}: {e}")
            sys.exit(1)
        print("")

    # Generate complete flags for next step to commence
    generate_complete_flag(os.path.join(ROOTDIR, 'data'), append='1_balance_dx')

    print("Finished matching!")

if __name__ == "__main__":
    main()
