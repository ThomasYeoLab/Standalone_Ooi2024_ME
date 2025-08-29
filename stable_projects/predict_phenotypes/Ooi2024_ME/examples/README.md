## Example of running KRR
This example runs the kernel ridge regression code for simulated data.

### Input
Data for this example can be found in the `example_data/sim_data` folder. 

A description of the data for the subjects are as follows.
1. `y.mat`: A mat file of the variables to be predicted.
2. `RSFC.mat`: A mat file of the features to be used for prediction.
3. `covariates.mat`: A mat file of covariates to control for.
4. `no_relative_2_fold_sub_list.mat`: a mat file of the cross validation structure.

### Scripts
1. `CBIG_ME_KRR_example_wrapper.m`: Runs the KRR prediction.
2. `CBIG_ME_KRR_check_example_results.m`: Checks whether the output of the example are the same as the reference results.

## Example of fitting theoretical models for the NxT matrix of a given phenotype
This example fits the logarithmic and theoretical models in Ooi2024_ME to the NxT matrix of prediction 
accuracies for the cognition factor phenotype in the Human Connectome Project. Alternative to the MATLAB scripts, a Python version can be found here `Ooi2024_ME/unit_tests/test_CBIG_ME_unit_test.py`. The run time is approximately 30 minutes.

### Input
Data for this example can be found in the `example_data/HCP` folder. This contains the data 
in the folder structure that will be generated after the prediction scripts & 
`utilities/regr_utils/CBIG_ME_HCP_readKRR` are run. 

A description of the data for the subjects are as follows.
1. `HCP_variables_to_predict_real_names.txt`: A text file containing the variable names in the HCP.
2. `acc_KRR_corr_landscape.mat`: A NxTx#phenotypes mat file containing the accuracies of all the tested
phenotypes in the HCP. The cognition factor is in index 60 (matlab-indexed) / 59 (python-indexed).

### Scripts
1. `CBIG_ME_curve_example_wrapper.m`: Runs the curve fitting and saves the fitted parameters, and loss from the fitted curves. Please activate the `Ooi2024_ME` python environment before running the scripts. 
2. `CBIG_ME_check_curve_example_results.m` or `CBIG_ME_check_example_savfile.py`: Checks whether the output of the example are the same as the reference results.