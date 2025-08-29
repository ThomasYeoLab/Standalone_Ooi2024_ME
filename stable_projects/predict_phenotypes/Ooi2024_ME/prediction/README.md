# Run regression models for prediction tasks
This folder provides functions to run the regression models. 

## Usage
We provide scripts to run KRR and LRR respective folders. For the `KRR` and `LRR` folders,
the scripts specific to each dataset (e.g. `ABCD` and `HCP`) are placed in a sub-folder with the 
corresponding dataset name. Scripts with the suffix _splithalf_ are used for the reliability testing
procedure where the dataset is split into 2. Their functionality is the same as the scripts without
the _splithalf_ suffix.

## Descriptions of prediction workflows
* ABCD workflows:
1. `CBIG_ME_ABCD_KRR`: default KRR workflow for running a leave-n-site-out cross validation
2. `CBIG_ME_ABCD_KRR_random_FC`: randomize order of runs when calculating FC as input features
3. `CBIG_ME_ABCD_subcorticalKRR`: use 19x419 subcortical connections as when calculating FC as input features
4. `CBIG_ME_ABCD_taskKRR`: workflow to run for task data
5. `CBIG_ME_ABCD_LRR_fr`: run LRR workflow

* HCP workflows:
1. `CBIG_ME_HCP_KRR`: default workflow for running a standard cross validation, able to accommodate random order as well
2. `CBIG_ME_HCP_subcorticalKRR`: use 19x419 subcortical connections when calculating FC as input features
3. `CBIG_ME_HCP_LRR_fr`: run LRR workflow