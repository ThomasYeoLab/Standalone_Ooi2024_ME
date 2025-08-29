#!/usr/bin/env python
# coding: utf-8
'''
This python script checks the sav file of the generated file and example file.
Inputs are the paths to the .sav files.

Written by Leon Ooi and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
'''

########################################################
# import packages
########################################################
import numpy as np
import sys
import os
import pickle

script_dir = os.path.join(
    os.environ.get('CBIG_CODE_DIR'), 'stable_projects', 'predict_phenotypes',
    'Ooi2024_ME', 'curve_fitting')
sys.path.insert(0, script_dir)
# add no qa to prevent pep8 error since this requires environment to
# be loaded before importing
import CBIG_ME_fns as orsp  # noqa

########################################################
# initialize paths and inputs
########################################################
reference = sys.argv[1]
example = sys.argv[2]

# set number of phenotypes based on dataset
test_length = 27
b = 60

# load reference files
pickle_path = reference
pickle_f = open(pickle_path, 'rb')
ref_file = pickle.load(pickle_f)
pickle_f.close()

# load example files
pickle_path = example
pickle_f = open(pickle_path, 'rb')
example_file = pickle.load(pickle_f)
pickle_f.close()

# test conditions
r_diff = np.sum(
    np.array(ref_file['w_r_sav']) - np.array(example_file['w_r_sav']))
assert r_diff < 1e-6, "Reliability model weights differ!"
pa_diff = np.sum(
    np.array(ref_file['w_pa_sav']) - np.array(example_file['w_pa_sav']))
assert pa_diff < 1e-6, "Accuracy model weights differ!"
log_diff = np.sum(
    np.array(ref_file['zk_sav']) - np.array(example_file['zk_sav']))
assert log_diff < 1e-6, "Log model weights differ!"
r_cod = np.sum(
    np.array(ref_file['loss_n_r']) - np.array(example_file['loss_n_r']))
assert r_cod < 1e-6, "Reliability model COD values differ!"
pa_cod = np.sum(
    np.array(ref_file['loss_n_pa']) - np.array(example_file['loss_n_pa']))
assert pa_cod < 1e-6, "Accuracy model COD values differ!"
log_cod = np.sum(
    np.array(ref_file['loss_log']) - np.array(example_file['loss_log']))
assert log_cod < 1e-6, "Log model COD values differ!"
