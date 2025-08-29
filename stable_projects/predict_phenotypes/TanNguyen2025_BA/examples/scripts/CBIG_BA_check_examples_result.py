'''
Written by Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script runs examples to check the training and testing
of the logistic regression model.

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m examples.scripts.CBIG_BA_check_examples_result
'''

import os
import json
import sys
import math


def load_test_auc(filepath):
    '''
    Extracts `test_auc` field from `params_test.json`.
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['test_auc']


def compare_test_auc(generated_path, reference_path):
    '''
    Compares if generated `params_test.json` has identical
    `test_auc` value to that of reference `params_test.json`.
    '''
    gen_auc = load_test_auc(generated_path)
    ref_auc = load_test_auc(reference_path)

    if math.isclose(gen_auc, ref_auc, abs_tol=1e-6):
        return True, gen_auc, ref_auc
    else:
        return False, gen_auc, ref_auc


def main():
    base_dir = os.environ.get('TANNGUYEN2025_BA_DIR')
    if not base_dir:
        print("Environment variable TANNGUYEN2025_BA_DIR is not set.")
        sys.exit(1)

    # Define and check paths exist
    gen_path = os.path.join(base_dir, 'examples', 'output',
                            'params_test.json')
    ref_path = os.path.join(base_dir, 'examples', 'ref_output',
                            'params_test.json')
    if not os.path.exists(gen_path):
        print(f"Generated file not found: {gen_path}")
        sys.exit(1)
    if not os.path.exists(ref_path):
        print(f"Reference file not found: {ref_path}")
        sys.exit(1)

    # Compare test AUC
    passed, gen_auc, ref_auc = compare_test_auc(gen_path, ref_path)

    # Determine pass/fail
    if not passed:
        print(f"ERROR: test_auc mismatch!")
        print(f"Generated: {gen_auc}")
        print(f"Reference: {ref_auc}")
        print("Model's results do not match the reference outputs.")
    else:
        print("Passed! Results are identical to the examples reference outputs.")


if __name__ == "__main__":
    main()
