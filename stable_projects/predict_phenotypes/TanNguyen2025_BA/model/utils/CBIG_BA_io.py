"""
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

Utility functions for loading and saving data.
"""

import os
import json
import time
from datetime import datetime


def list2txt(data_list, out_file, allow_overwrite=False):
    """
    Write a list to a text file, one item per line.

    Args:
        data_list (list): List of items to write to the file.
        out_file (str): Path to the output .txt file.
        allow_overwrite (bool): Whether to overwrite the file if it already exists.
    """
    #if not allow_overwrite and os.path.exists(out_file):
        #raise FileExistsError(f'{out_file} exists!')

    with open(out_file, 'w') as f:
        for item in data_list:
            f.write('{}\n'.format(item))


def txt2list(txt_file):
    """
    Read a text file into a list. Converts items to float if possible.

    Args:
        txt_file (str): Path to the input .txt file.

    Returns:
        list: List of strings or floats from file.
    """
    outlist = []
    with open(txt_file, 'r') as f:
        for line in f:
            try:
                tmp = float(line.strip())
                outlist.append(tmp)
            except ValueError:
                outlist.append(line.strip())

    return outlist


def read_json(json_file, variables=None):
    """
    Read a JSON file and optionally extract selected variables.

    Args:
        json_file (str): Path to the JSON file.
        variables (list, optional): Keys to extract from the JSON. If None, returns all.

    Returns:
        dict: Parsed JSON content or subset.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    if variables:
        return {k: data[k] for k in variables}
    else:
        return data


def create_output_folder(base_out_dir, n_try=10):
    """
    Create an output folder named by timestamp in the base directory.
    Retries creation if folder exists (due to parallel jobs).

    Args:
        base_out_dir (str): Path to the base output directory.
        n_try (int): Number of retry attempts (default = 10).

    Returns:
        tuple: (timestamp string, created output directory path)
    """

    for _ in range(n_try):
        try:
            start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            output_dir = os.path.join(base_out_dir, start_time)
            os.makedirs(output_dir, exist_ok=True)
            str_error = None
        except Exception as e:
            str_error = e
            pass

        if str_error:
            time.sleep(5)
        else:
            break

    # Make sure the output directory is created
    assert os.path.exists(output_dir), 'Output directory does not exist.'

    return start_time, output_dir
