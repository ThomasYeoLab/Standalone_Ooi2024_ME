#!/bin/bash

#####
# This wrapper script submits job to the schduler to run curve fitting to each dataset.
# Dataset, and type of analysis needs to be specified. A .sav file will be the output with 
# the theoretical and log curves fitted from the 3rd T point onwards (e.g. if ABCD is 2min 
# to 20mins, the curves will be fitted from 6min to 20 mins).
#
# Input:
#    - dataset:
#      An string indicating the dataset. Can be "HCP", "ABCD", "SINGER", "TCP", "MDD and "ADNI".
#
#    -datatype:
#     The type of analysis to run. Can be "predacc" for prediction accuracy analysis, "tstats" for 
#     univariate reliability, and "Haufe" for multivariate reliability. "pfrac" is possible, but is 
#     not used in the paper.
#
#    -vers:
#     The manner in which FC was calculated (See FC generation scripts). Can be "full", "no_censoring",
#     "uncensored_only" or "random".
#
# EXAMPLE: 
#    CBIG_ME_submit_job.sh 'ABCD' 'predacc' 12 full
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
scriptdir=$ME_CODE_DIR/curve_fitting
# modify name of python environment if needed
py_env="Ooi2024_ME" 

### read function inputs
dataset=$1
datatype=$2
vers=$3

# set variables based on input
if [ $dataset == 'HCP' ]; then
    behavs=$(seq 0 1 60)
elif [ $dataset == 'ABCD' ]; then
    behavs=$(seq 0 1 38)
elif [ $dataset == 'SINGER' ]; then
    behavs=$(seq 0 1 18)
elif [ $dataset == 'TCP' ]; then
    behavs=$(seq 0 1 18)
elif [ $dataset == 'ADNI' ]; then
    behavs=$(seq 0 1 6)
elif [ $dataset == 'MDD' ]; then
    behavs=$(seq 0 1 19)
else
    echo "Input not recognised, exiting..."
    exit
fi

if [ $datatype == 'predacc' ]; then
    output_vers=output
else
     output_vers=output_splithalf
fi

### Create log file and save params
outdir=$ME_rep_dir/$dataset/$output_vers/$vers/curve_fit
if [ ! -d $out_dir ]; then mkdir -p $out_dir; fi
if [ ! -d $out_dir/logs ]; then mkdir -p $out_dir/logs; fi

### run curve fitting
# loop over phenotypes
for behav in $behavs; do
        log_f=$outdir/logs/${dataset}_${datatype}_behav${behav}.log
        cmd="source activate $py_env; python ${scriptdir}/CBIG_ME_fit_all.py \
            $dataset $behav $vers $datatype $common_dir"
        ssh headnode "$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd '${cmd}' \
                -walltime 1:30:00 -name '${dataset}_${vers}_fitcurve' \
                -mem '8G' -joberr '$outdir/logs' -jobout '$log_f'" < /dev/null
done