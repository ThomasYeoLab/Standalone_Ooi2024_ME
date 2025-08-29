# Fit logarithmic and theoretical models to the NxT matrix of prediction accuracies
Functions in this folder provides the python scripts used to fit the logarithmic and
theoretical models for each NxT matrix from a given phenotype. 

## Usage
Ensure that you have run the KRR scripts in the `prediction` folder already and collated the
results in a NxT matrix using the collate results functions in `utilities/regr_utils`. The scripts
will read the matrix and fit the models using gradient descent.

* `CBIG_ME_submit_job.sh`: Wrapper script to run curve fitting for all phenotypes of a given dataset
* `CBIG_ME_fit_all.py`: Python function to fit logarithmic and theoretical models for a NxT matrix
* `CBIG_ME_fns.py`: Function wrapper with all functions for plotting and fitting curves

## Notes
* Please ensure you have the python dependencies installed. You can install `CBIG_ME_python_env.yml` under the `replication/config` folder.