# Analyse reliability of feature importance for prediction tasks
This folder provides functions to analyse the reliability of interpreting predictive models. 

## Usage
We provide scripts to run the reliability analysis specific to each dataset (e.g. `ABCD` and `HCP`).
The scripts are placed in a sub-folder with the corresponding dataset name. 

## Descriptions of reliability workflows
1. `CBIG_ME_<dataset>_BWA`: analyses the univariate feature importance by calculating the t-statistic between 
 each FC edge and the phenotype of interest.
2. `CBIG_ME_<dataset>_Haufe`: analyses the multivariate feature importance by calculating the Haufe-transformed regression weights between 
 each FC edge and the phenotype of interest.