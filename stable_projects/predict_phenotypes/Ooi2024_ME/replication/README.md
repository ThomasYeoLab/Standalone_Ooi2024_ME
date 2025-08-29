# Run replication
Functions in this folder run the replication results. 

## Usage
The replication functions are split by dataset (e.g. `ABCD` and `HCP`), but the process remains similar.
The steps are outlined as follows:

1. `CBIG_ME_<dataset>_genFC_wrapper`: Generate FC matrices using different conditions and different amounts of frames
2. `CBIG_ME_<dataset>_runRegressions_wrapper`: Run regressions for dataset, with the different conditions and analysis modes 
3. `CBIG_ME_<dataset>_runReliability_wrapper`: Generate reliability results for feature importance

Once these steps are completed, you can collate the results using the `readKRR` or `readLRR` scripts in the 
`utilities` folder to collate the prediction accuracies into a NxT matrix. These can further be used to 
fit the logarithmic and theoretical models. Instructions to fit the models can be found in `curve_fitting`.
In the aforementioned folder, the function `CBIG_ME_submit_job.sh` can be used as a wrapper script to fit
the models and save the model parameters, and goodness-of-fit.