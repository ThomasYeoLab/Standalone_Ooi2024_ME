#!/usr/bin/env python
# coding: utf-8
'''
This python script fits the logarithmic model and theoretical model for
prediction accuracy and reliability in Ooi2024_ME.

This code assumes that you have saved the prediction accuracies or
reliabilities in a NxT mat file. This can be done by running the readKRR
function in the utilities folder. The model will be fit from the
3rd T datapoint onwards.

Inputs:
    - dataset
      A string specifying the dataset that is being processed.
      Settings are based on the dataset.

    - behav_num
      An integer with the index (0-indexed) of the phenotype being analysed.

    - vers
      The analysis outstem. Can be "full", "random", "no_censoring",
      "uncesored_only", "full_1000parcels", "full_mixdays".

    - data_type
      The type of file to be read. Can be "predacc", "tstats", "Haufe"
      or "pfrac"

    - rep_dir
      The replication directory. Where the input and output data for
      the prediction models reside.

Outputs:
    - sav_file
      A sav file of the weights, and COD of the model fit for the model fit
      to each T minutes.

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
# read required inputs
dataset = sys.argv[1]
behav_num = int(sys.argv[2])
vers = sys.argv[3]
data_type = sys.argv[4]
rep_dir = sys.argv[5]

# modify variables related to the dataset
if dataset == 'HCP':
    test_length = 30
elif dataset == 'ABCD':
    if ("MID" in vers):
        test_length = 11
    elif ("NBACK" in vers):
        test_length = 10
    elif ("SST" in vers):
        test_length = 12
    else:
        test_length = 11
elif dataset == 'SINGER':
    test_length = 10
elif dataset == 'TCP':
    test_length = 14
elif dataset == 'ADNI':
    test_length = 9
elif dataset == 'MDD':
    test_length = 12
else:
    print("Dataset not recognized!")
    exit(1)

# modify variables related to the analysis
if data_type == 'predacc':
    output_vers = 'output'
    mat_variable = 'acc_landscape'
else:
    output_vers = 'output_splithalf'
    if data_type == 'tstats':
        mat_variable = 'tstats_icc_landscape'
    elif data_type == 'pfrac':
        mat_variable = 'p_frac_landscape'
    elif data_type == 'Haufe':
        mat_variable = 'fi_icc_landscape'
# set up paths
input_dir = rep_dir
output_dir = os.path.join(rep_dir, dataset, output_vers, vers, 'curve_fit')

#################################################
# load data
#################################################
scores = np.genfromtxt(
    os.path.join(input_dir, dataset, 'input',
                 dataset + '_variables_to_predict_real_names.txt'),
    dtype=str,
    delimiter='\n')
img_dir, res, X, Y, extent, scan_duration = orsp.load_data(
    dataset, data_type, rep_dir, vers=vers)
behav_all = np.flip(np.flip(res[mat_variable][:, :, behav_num].T), 1)

# print inputs for reference in log file
print("---------------------")
print("dataset:" + dataset)
print("behav_num:" + str(behav_num))
print("behav_name:" + str(scores[behav_num]))
print("vers:" + vers)
print("data_type:" + data_type)
print("Generating results in:" + output_dir)
print("---------------------")

#################################################
# fit models
#################################################

# prepare variables to be saved
w_r_sav = []
w_pa_sav = []
zk_sav = []
loss_n_pa = []
loss_n_r = []
loss_log = []

# fit equation from 3rd point to max scan time
for limit in range(3, test_length):
    # set up according to limit
    behav = behav_all[:, :limit]
    curr_scan = scan_duration[:, :limit]

    #################################################
    # theoretical models
    #################################################
    # reliability equation
    w_r = orsp.gd_rel(X[:limit], behav.flatten(), Y)
    w_r_sav.append(w_r)
    print("mins:", X[limit - 1], "w (reliability):", w_r)
    # measure loss
    curve_val = w_r[0] / (
        w_r[0] + (1 / (np.repeat(Y, limit) / 2)) *
        (1 - 2 * w_r[1] /
         (1 + (w_r[2] / np.squeeze(np.tile(X[:limit], (1, len(Y))))))))
    mse, COD = orsp.calc_loss(behav.flatten(), curve_val)
    loss_n_r.append(COD)

    # prediction accuracy equation
    w_pa = orsp.gd_pred_acc(X[:limit], behav.flatten(), Y)
    w_pa_sav.append(w_pa)
    print("mins:", X[limit - 1], "w (pred acc):", w_pa)
    # measure loss
    curve_val = w_pa[0] * np.sqrt(1 / (1 + (w_pa[1] / np.repeat(Y, limit)) +
                                       (w_pa[2] / (curr_scan.flatten()))))
    mse, COD = orsp.calc_loss(behav.flatten(), curve_val)
    loss_n_pa.append(COD)

    #################################################
    # log model
    #################################################
    # use least squares
    z, k = orsp.lst_sq_log(curr_scan.flatten(), behav.flatten())
    zk_sav.append([z, k])
    # measure loss
    curve_val = z * (np.log(curr_scan.flatten()) / np.log(2)) + k
    mse, COD = orsp.calc_loss(behav.flatten(), curve_val)
    loss_log.append(COD)

# save variables
file = open(
    os.path.join(output_dir,
                 data_type + '_behav' + str(behav_num) + '_results.sav'), 'wb')
curve_fit = {
    "w_r_sav": w_r_sav,
    "w_pa_sav": w_pa_sav,
    "zk_sav": zk_sav,
    "loss_n_r": loss_n_r,
    "loss_n_pa": loss_n_pa,
    "loss_log": loss_log
}
pickle.dump(curve_fit, file)
file.close()
