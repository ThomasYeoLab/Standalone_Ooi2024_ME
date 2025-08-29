#!/bin/sh

#####
# This wrapper script submits job to the schduler to run the specified type of regression, using
# the specified input FCs. These scripts are for the ABCD dataset.
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
#    CBIG_ME_ABCD_schedule_regression.sh KRR full
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh
KRRscriptdir=$ME_CODE_DIR/prediction/KRR/ABCD
LRRscriptdir=$ME_CODE_DIR/prediction/LRR/ABCD
projectdir=$ME_rep_dir/ABCD
FCdir=$projectdir/input/FC

### read function inputs	
regression=$1
vers=$2

### set up scheduling variables based on type of regression
if [ $regression == "KRR" ]; then 
    # predictions using original and randomized run order
    base_cmd="$KRRscriptdir/CBIG_ME_ABCD_KRR.sh "
    time_allowed="20:00:00"
    scheduler_tag="ABCD_KRR"
    output_stem='output'
    # change time and memory based on version
    if [ $vers == "random" ]; then
        mem="20GB"
        time_allowed="84:00:00"
    else
        mem="100GB"
        time_allowed="20:00:00"
    fi
elif [ $regression == "LRR" ]; then
    # predictions using original run order
    base_cmd="$LRRscriptdir/CBIG_ME_ABCD_LRR_fr.sh "
    time_allowed="48:00:00"
    output_stem='output'
    scheduler_tag="ABCD_LRR"
    mem="20GB"
    # additionally need to specify sample sizes for script
    sample_array=(200 400 600 800 1000 1200 1400 1600) 
elif [ $regression == "KRR_sc" ]; then
    # prediction using subcortical connections
    base_cmd="$KRRscriptdir/CBIG_ME_ABCD_subcorticalKRR.sh "
    time_allowed="20:00:00"
    scheduler_tag="ABCD_subcortKRR"
    output_stem='output'
    mem="40GB"
    time_allowed="20:00:00"
elif [ $regression == "KRR_task" ]; then
    # predictions using task data
    base_cmd="$KRRscriptdir/CBIG_ME_ABCD_taskKRR.sh "
    time_allowed="20:00:00"
    scheduler_tag="ABCD_taskKRR"
    output_stem='output'
    mem="30GB"
    time_allowed="20:00:00"
elif [ $regression == "KRR_sh" ]; then
    # predictions using split-half set up using original and randomized run order
    base_cmd="$KRRscriptdir/CBIG_ME_ABCD_KRR_splithalf.sh "
    time_allowed="20:00:00"
    scheduler_tag="ABCD_KRR_splithalf"
    output_stem='output_splithalf'
    if [ $vers == "random" ]; then
        mem="20GB"
        time_allowed="84:00:00"
    else
        mem="20GB"
        time_allowed="48:00:00"
    fi
elif [ $regression == "KRR_sh_random_ss" ]; then
    # predictions using split-half set for randomized run order
    # can be used to speed up process by running 1 job for each subsample size
    # not run by default
    base_cmd="$KRRscriptdir/CBIG_ME_ABCD_KRR_splithalf_subsample.sh "
    scheduler_tag="ABCD_KRR_ss_splithalf"
    output_stem='output_splithalf'
    mem="20GB"
    time_allowed="84:00:00"
    sample_array=(200 400 600 800 1000) 
else
    echo "ERROR: type of regression not recognised, exiting..."
    exit
fi

### set up scheduling variables based on version
if [ $vers == "uncensored_only" ]; then
    min_array=($(seq 2 2 14) "15")
elif [ $regression == "KRR_task" ]; then
    min_array=($(seq 1 1 12))
else
    min_array=($(seq 2 2 20))
fi

### other settings and paths
outdir=$projectdir/$output_stem/$vers
logdir=$outdir/scheduler_logs_tmp
mkdir -p $logdir
if [ $regression == "KRR_task" ]; then
    cov_file=$outdir/subject2262_variables_to_predict.mat 
    pred_file=$outdir/subject2262_covariates.mat
else
    cov_file=$outdir/subject2565_covariates.mat 
    pred_file=$outdir/subject2565_variables_to_predict.mat
fi

### define run job function
run_jobs(){

# run regression in scheduler
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "$time_allowed" -name "$scheduler_tag" \
    -mem "$mem" -joberr "$logdir" -jobout "$logdir"

# covariate, y variable and input FC files that is used by all splits will be generated 
# wait for the files to be generated
if [ ! -f $cov_file ] || [ ! -f $pred_file ]; then
    sleep 3m   
else
    sleep 2s
fi

}

### submit all required jobs to scheduler
# for each of T FCs
for which_min in ${min_array[@]}; do
    # append input variables based on regression type
    if [ $regression == "KRR" ] || [ $regression == "KRR_task" ] || \
        [ $regression == "KRR_sc" ] || [ $regression == "KRR_sh" ]; then
        cmd="$base_cmd $which_min $vers"
        run_jobs
    elif [ $regression == "KRR_sh_random_ss" ]; then
        for sample in ${sample_array[@]}; do
            cmd="$base_cmd $which_min $vers $sample "
            run_jobs
        done
    elif [ $regression == "LRR" ]; then
        for sample in ${sample_array[@]}; do
            cmd="$base_cmd $which_min $vers $sample "
            run_jobs
        done
    fi
done
