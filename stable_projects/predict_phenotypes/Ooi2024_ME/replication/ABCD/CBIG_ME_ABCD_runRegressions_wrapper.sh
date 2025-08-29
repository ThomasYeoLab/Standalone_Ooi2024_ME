#!/bin/sh

#####
# This is a wrapper script to run prediction algorithms in the ABCD dataset.
#
# EXAMPLE: 
#    CBIG_ME_ABCD_runRegressions_wrapper.sh
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh
script_dir=$ME_CODE_DIR/utilities/regr_utils

$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR_sh_random_ss random
exit

### Prediction code for main prediction accuracy analysis
## run KRR and LRR for 419x419-resolution FC matrices using original run order
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR full
$script_dir/CBIG_ME_ABCD_schedule_regression.sh LRR full
## run KRR 419x419-resolution FC matrices with "no_censoring", "uncensored_only" settings
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR no_censoring
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR uncensored_only
## run KRR 419x419-resolution FC matrices with randomized run order
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR random
## run KRR for 1019x1019-resolution FC matrices using original run order
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR full_1000parcels
## run KRR 419x19 FC matrices subcortical connections using original run order
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR_sc full
## run KRR and LRR for 419x419-resolution FC matrices for task data
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR_task full_MID
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR_task full_NBACK
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR_task full_SST

### Prediction code for reliability analysis
## run splithalf KRR for 419x419-resolution FC matrices using original and randomized run order
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR_sh full
$script_dir/CBIG_ME_ABCD_schedule_regression.sh KRR_sh random





