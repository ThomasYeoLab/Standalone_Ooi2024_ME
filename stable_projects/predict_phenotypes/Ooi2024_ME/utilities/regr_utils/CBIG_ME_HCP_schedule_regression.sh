#!/bin/sh

#####
# This wrapper script submits job to the schduler to run the specified type of regression, using
# the specified input FCs. These scripts are for the HCP dataset.
#
# Input:
#    -regression: 
#     The type of regression to use. Can be "KRR" or "LRR". 
#     Append "_sh" for the split-half analysis used for assessing reliability (e.g. "KRR_sh"). 
#     Append "_task" for running task fMRI. Type of task will have to be specified in `vers` as "full_taskname".
#     Append "_sc" for running predictions based on subcortical connections.
#
#    -vers:
#     The manner in which FC was calculated (See FC generation scripts). Can be "full", "no_censoring",
#     "uncensored_only" or "random".
#
# EXAMPLE: 
#    CBIG_ME_HCP_schedule_regression.sh KRR full
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh
KRRscriptdir=$ME_CODE_DIR/prediction/KRR/HCP
LRRscriptdir=$ME_CODE_DIR/prediction/LRR/HCP
projectdir=$ME_rep_dir/HCP
FCdir=$projectdir/input/FC

### read function inputs	
regression=$1
vers=$2

### set up scheduling variables based on type of regression
if [ $regression == "KRR" ]; then
    # predictions using original and randomized run order
    base_cmd="$KRRscriptdir/CBIG_ME_HCP_KRR.sh "
    output_stem='output'
    # change time and memory based on version
    if [ $vers == "random" ]; then
        mem="40GB"
        time_allowed="06:00:00"
    else
        mem="50GB"
        time_allowed="04:00:00"
    fi
elif [ $regression == "LRR" ]; then
    # predictions using original run order
    base_cmd="$LRRscriptdir/CBIG_ME_HCP_LRR_fr.sh "
    time_allowed="10:00:00"
    output_stem='output'
    scheduler_tag="HCP_LRR"
    mem="10GB"
elif [ $regression == "KRR_sc" ]; then
    # prediction using subcortical connections
    base_cmd="$KRRscriptdir/CBIG_ME_HCP_subcorticalKRR.sh "
    scheduler_tag="HCP_subcorticalKRR"
    output_stem='output'
    mem="20GB"
    time_allowed="04:00:00"
elif [ $regression == "KRR_sh" ]; then
    # predictions using split-half set up using original and randomized run order
    base_cmd="$KRRscriptdir/CBIG_ME_HCP_KRR_splithalf.sh "
    scheduler_tag="HCP_KRR_splithalf"
    output_stem='output_splithalf'
    if [ $vers == "random" ]; then
        mem="40GB"
        time_allowed="06:00:00"
    else
        mem="16GB"
        time_allowed="04:00:00"
    fi
else
    echo "ERROR: type of regression not recognised, exiting..."
    exit
fi

### set up scheduling variables based on version
if [ $vers == "uncesored_only" ]; then
    min_array=($(seq 2 2 40))
else
    min_array=($(seq 2 2 58))
fi

### other settings and paths
split_array=$(seq 1 1 50) # number of random splits
outdir=$projectdir/$output_stem/$vers
logdir=$outdir/scheduler_logs
mkdir -p $logdir
cov_file=$outdir/subject792_covariates.mat 
pred_file=$outdir/subject792_variables_to_predict.mat

### define run job function
run_jobs(){

# run regression in scheduler
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "$time_allowed" -name "$scheduler_tag" \
    -mem "$mem" -joberr "$logdir" -jobout "$logdir"

# covariate, y variable and input FC files that is used by all splits will be generated 
# wait for the files to be generated
FC_f=$FCdir/$vers/${which_min}min_FC.mat
if [ ! -f $cov_file ] || [ ! -f $pred_file ]; then
    sleep 3m   
else
    sleep 2s
fi

}

### submit all required jobs to scheduler
# for each split
for which_split in ${split_array[@]}; do
    # append input variables based on regression type
    if [ $regression == "KRR" ] || [ $regression == "KRR_sc" ] || [ $regression == "KRR_sh" ]; then
        # for each of T FCs
        for which_min in ${min_array[@]}; do
             cmd="$base_cmd $which_split $which_min $vers"
             run_jobs
        done	
    elif [ $regression == "LRR" ]; then
        # for each of T FCs
        for which_min in ${min_array[@]}; do
             cmd="$base_cmd $which_split $which_min $vers "
             run_jobs
        done
    fi
done
