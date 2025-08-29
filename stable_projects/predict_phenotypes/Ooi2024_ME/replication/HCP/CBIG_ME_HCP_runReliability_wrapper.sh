#!/bin/sh

#####
# This is a wrapper script to run reliability analysis for the split-half predictions in the HCP dataset.
# "BWA" refers to the univariate reliability analysis and "Haufe" refers to the multivariate reliability analysis.
#
# EXAMPLE: 
#    CBIG_ME_HCP_runReliability_wrapper.sh
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh
script_dir=$ME_CODE_DIR/utilities/interp_utils

### run reliability analysis for original run order
$script_dir/CBIG_ME_HCP_schedule_interpretation.sh full BWA
$script_dir/CBIG_ME_HCP_schedule_interpretation.sh full Haufe
### run reliability analysis for randomized run order
$script_dir/CBIG_ME_HCP_schedule_interpretation.sh random BWA
$script_dir/CBIG_ME_HCP_schedule_interpretation.sh random Haufe
