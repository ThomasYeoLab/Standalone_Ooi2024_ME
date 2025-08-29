'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script performs the following tasks:
1. Cleans up clinical data by running a Python script (`uniform_data`).
2. Copies over hyperparameter files.

Expected output(s):
1. $TANNGUYEN2025_BA_DIR/data/raw
2. $TANNGUYEN2025_BA_DIR/data/params

Example:
    conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR};
    python -m replication.scripts.CBIG_BA_0_prepare_data;
'''

import os
import shutil
import subprocess
import sys
import requests
import h5py

TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)
from CBIG_BA_config import global_config
from utils.CBIG_BA_complete_step import generate_complete_flag


def clean_up_clinical_data():
    """Run the Python script to clean up clinical data."""
    try:
        subprocess.run(["python", "-m", "data_split.CBIG_BA_uniform_data"],
                       check=True)
        print("Clinical data cleaned up successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during cleaning up clinical data: {e}")
        raise


def copy_hyperparameters(REPLICATION_DATA, ROOT_DIR):
    """Copy the hyperparameters from the REPLICATION_DATA path."""
    params_src = os.path.join(REPLICATION_DATA, 'params')
    params_dst = os.path.join(ROOT_DIR, 'data', 'params')

    try:
        # Use shutil to copy the hyperparameters folder
        shutil.copytree(params_src, params_dst, dirs_exist_ok=True)
        print(f"Hyperparameters copied from {params_src} to {params_dst}.")
    except Exception as e:
        print(f"Error during copying hyperparameters: {e}")
        raise


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    with requests.Session() as session:
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def get_default_model_dir():
    root = os.path.join(os.path.expanduser('~'), '.pyment', 'models')
    os.makedirs(root, exist_ok=True)
    return root


def is_valid_hdf5(filepath):
    try:
        with h5py.File(filepath, 'r'):
            return True
    except Exception as e:
        print(f"[✗] Invalid HDF5 file: {filepath}")
        print("Error:", e)
        return False


def download_pyment_weights(output_dir=None):
    if output_dir is None:
        output_dir = get_default_model_dir()

    files = {
        "regression_sfcn_brain_age_weights.hdf5":
        "1-_nWsovjIFhfegmmnzYQpTXqeRGn0H0G",
        "regression_sfcn_brain_age_weights_no_top.hdf5":
        "1-X0d0NBXFOCwbVu0Rn6WbtyFkZZ2tjI9"
    }

    for filename, file_id in files.items():
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            print(f"[!] {filename} already exists. Verifying...")

            if is_valid_hdf5(output_path):
                print(f"[✓] {filename} is valid. Skipping download.")
            else:
                print(f"[✗] {filename} is corrupted. Re-downloading...")
                os.remove(output_path)
                download_file_from_google_drive(file_id, output_path)

                if is_valid_hdf5(output_path):
                    print(f"[✓] {filename} downloaded and verified.")
                else:
                    print(f"[✗] Download failed again: {filename}")
                    sys.exit(1)
        else:
            print(f"Downloading {filename} to {output_path}...")
            download_file_from_google_drive(file_id, output_path)

            if is_valid_hdf5(output_path):
                print(f"[✓] {filename} downloaded and verified.")
            else:
                print(f"[✗] Download failed or corrupted: {filename}")
                os.remove(output_path)
                sys.exit(1)

    print("Download complete.")


def main():
    # Access paths from CBIG_BA_config.py
    REPLICATION_DATA = global_config.REPLICATION_DATA
    ROOT_DIR = global_config.ROOT_DIR

    # Change working dir to ROOT_DIR
    os.chdir(ROOT_DIR)

    # Set the directory paths
    data_dir = os.path.join(ROOT_DIR, 'data')

    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Clean up clinical data
    clean_up_clinical_data()

    # Copy hyperparameters
    copy_hyperparameters(REPLICATION_DATA, ROOT_DIR)

    # Download Pyment weights
    download_pyment_weights()

    # Generate complete flags for next step to commence
    generate_complete_flag(data_dir, append='0_prepare_data')


if __name__ == "__main__":
    main()
