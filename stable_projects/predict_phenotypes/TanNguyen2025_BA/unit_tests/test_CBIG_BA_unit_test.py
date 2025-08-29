'''
Written by Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script runs unit tests to check the training and testing
of the logistic regression model.

Example:
    cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/TanNguyen2025_BA/unit_tests
    conda activate CBIG_BA
    python test_CBIG_BA_unit_test.py
'''

import sys
import os
import time
import unittest
import subprocess
import json
import re

TANNGUYEN2025_BA_DIR = os.getenv('TANNGUYEN2025_BA_DIR')
if not TANNGUYEN2025_BA_DIR:
    raise ValueError(
        "ERROR: TANNGUYEN2025_BA_DIR environment variable not set.")
sys.path.append(TANNGUYEN2025_BA_DIR)


def load_json_auc(filepath, auc_type):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data[f'{auc_type}_auc']


class TestBA(unittest.TestCase):

    def test_model(self):
        # Training & test model step
        # Define convenience variables
        out_dir = os.path.join(TANNGUYEN2025_BA_DIR, 'unit_tests', 'output')
        ref_dir = os.path.join(TANNGUYEN2025_BA_DIR, 'examples', 'ref_output')
        cmd = (f"cd {TANNGUYEN2025_BA_DIR};"
               "source CBIG_init_conda; conda activate CBIG_BA;"
               f"python -m model.log_reg.CBIG_BA_logistic_regression \
                --input_dir {TANNGUYEN2025_BA_DIR}/examples/input \
                    --output_dir {out_dir} --model_name log_reg \
                        --examples y;"
               "conda deactivate;")

        # Submit job & wait for job completion
        subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Compare results to reference
        ref_json = os.path.join(ref_dir, 'params_test.json')
        res_json = os.path.join(out_dir, 'params_test.json')
        ref = load_json_auc(ref_json, 'test')
        res = load_json_auc(res_json, 'test')
        self.assertAlmostEqual(
            res,
            ref,
            delta=1e-6,
            msg=
            f"Model results are different. Generated AUC: {res} vs Reference AUC: {ref}"
        )


if __name__ == '__main__':
    unittest.main()
