#!/bin/sh

#####
# This is a wrapper script to generate all the FCs used in the analysis for the ABCD dataset. 
#
# For rest-FC matrices there are 3 types of FC matrices that are generated. It can be run for at the 
# 419x419 FC resolution or 1019x1019 FC resolution. It is run from 2 mins to 20 mins in intervals of 2 mins
# (which we refer to as T mins).
#
# "full": 
# FC matrices are generated from T mins. Censoring occurs after T mins are grabbed from the time series.
# "no_censoring": 
# FC matrices are generated from T mins. No censoring is used.
# "uncesored_only": 
# FC matrices are generated from T mins. Censoring occurs before T mins are grabbed from the time series.
#
# For task-FC matrices only the "full" version is run at the 419x419 FC resolution for the MID, NBACK
# and SST tasks. It is run from 2 mins to 12 mins in intervals of 1 min.
#
# EXAMPLE: 
#    CBIG_ME_ABCD_genFC_wrapper.sh
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh
script_dir=$ME_CODE_DIR/utilities/FC_utils
input_dir=$ABCD_input_dir
# text files with list of subjects
rest_sub=$input_dir/subject_list_2565.txt
task_sub=$input_dir/TRBPC_2262_list.txt


### Generate FC for 419x419 FC matrices for rest-fMRI
## Original run order
vers='full'
min_array=($(seq 2 2 20))
perms=($(seq 1 1 24))
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    for perm in ${perms[@]}; do
        output_dir=$input_dir/FC/${vers}_perm${perm}
        logdir=$output_dir/logs
        if [ ! -d $logdir ];then mkdir -p $logdir; fi
        for batch in $(seq 1 1 3); do
            cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
                $script_dir/CBIG_ME_calc_ABCD_FC.sh $output_dir $mins $vers $rest_sub $perm $batch 400"
            # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
            $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "96:00:00" -name "ABCD_ME_genFC_full" \
                -mem "20GB" -joberr "$logdir" -jobout "$logdir"	
        done
    done
done

## No frames are censored
vers='no_censoring'
min_array=($(seq 2 2 20))
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    output_dir=$input_dir/FC/${vers}_perm24
    logdir=$output_dir/logs
    if [ ! -d $logdir ];then mkdir -p $logdir; fi
    for batch in $(seq 1 1 3); do
        cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
            $script_dir/CBIG_ME_calc_ABCD_FC.sh $output_dir $mins $vers $rest_sub 24 $batch 400"
        # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
        $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "96:00:00" -name "ABCD_ME_genFC_nc" \
            -mem "20GB" -joberr "$logdir" -jobout "$logdir"	
    done
done

## Do not use censored frames for to calculate the time
vers='uncensored_only'
min_array=($(seq 2 2 14) "15")
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    output_dir=$input_dir/FC/${vers}_perm24
    logdir=$output_dir/logs
    if [ ! -d $logdir ];then mkdir -p $logdir; fi
    for batch in $(seq 1 1 3); do
        cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
            $script_dir/CBIG_ME_calc_ABCD_FC.sh $output_dir $mins $vers $rest_sub 24 $batch 400"
        # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
        $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "96:00:00" -name "ABCD_ME_genFC_uo" \
            -mem "20GB" -joberr "$logdir" -jobout "$logdir"	
    done
done

### Generate FC for 1019x1019 FC matrices for rest-fMRI
## Original run order
vers='full'
min_array=($(seq 2 2 20))
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    output_dir=$input_dir/FC/${vers}_1000parcels
    logdir=$output_dir/logs
    if [ ! -d $logdir ];then mkdir -p $logdir; fi
    for batch in $(seq 1 1 3); do
        cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
            $script_dir/CBIG_ME_calc_ABCD_FC.sh $output_dir $mins $vers $rest_sub 24 $batch 1000"
        # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
        $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "96:00:00" -name "ABCD_ME_genFC_1000parcel" \
            -mem "20GB" -joberr "$logdir" -jobout "$logdir"
    done
done

### Generate FC for 419x419 FC matrices for task-fMRI
## Original run order
vers='full'
min_array=($(seq 1 1 12))
task_array=("MID" "NBACK" "SST")
# submit jobs to scheduler
for task in ${task_array[@]}; do
    for mins in ${min_array[@]}; do
        output_dir=$input_dir/FC/${vers}_${task}
        logdir=$output_dir/logs
        if [ ! -d $logdir ];then mkdir -p $logdir; fi
        for batch in $(seq 1 1 3); do
            cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
                $script_dir/CBIG_ME_calc_ABCD_taskFC.sh $output_dir $mins $vers $task_sub 2 $batch $task"
            # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
            $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "96:00:00" -name "ABCD_ME_genFC_${task}" \
                -mem "20GB" -joberr "$logdir" -jobout "$logdir"
        done
    done
done
