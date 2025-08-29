#!/bin/sh

#####
# This is a wrapper script to generate all the FCs used in the analysis for the HCP dataset. It generates
#
# For rest-FC matrices there are 5 types of FC matrices that are generated. It can be run for at the 
# 419x419 FC resolution or 1019x1019 FC resolution. It is run from 2 mins to 58 mins in intervals of 2 mins 
# (which we refer to as T mins).
#
# "full": 
# FC matrices are generated from T mins. Censoring occurs after T mins are grabbed from the time series.
# "no_censoring": 
# FC matrices are generated from T mins. No censoring is used.
# "uncesored_only": 
# FC matrices are generated from T mins. Censoring occurs before T mins are grabbed from the time series.
# "full_day1": 
# FC matrices are generated from T mins from runs from day 1 only. Censoring follows practice from "full".
# "full_day2: 
# FC matrices are generated from T mins from runs from day 2 only. Censoring follows practice from "full".
#
# EXAMPLE: 
#    CBIG_ME_HCP_genFC_wrapper.sh
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh
script_dir=$ME_CODE_DIR/utilities/FC_utils
input_dir=$HCP_input_dir
sub_txt=$input_dir/subject_list_792.txt

### settings for generating 400-resolution FC matrices using the "full" setting
vers='full'
min_array=($(seq 2 2 58))
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    perms=($(seq 1 1 25))
    for perm in ${perms[@]}; do
        output_dir=$input_dir/FC/${vers}_perm${perm}
        logdir=$output_dir/logs
        if [ ! -d $logdir ];then mkdir -p $logdir; fi
        cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
            $script_dir/CBIG_ME_calc_HCP_FC.sh $output_dir $mins $vers $sub_txt $perm 400"
        # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
        $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "24:00:00" -name "HCP_ME_genFC_full" \
            -mem "16GB" -joberr "$logdir" -jobout "$logdir"	
    done
done

### settings for generating 400-resolution FC matrices using the "no_censoring" setting
vers='no_censoring'
min_array=($(seq 2 2 58))	
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    output_dir=$input_dir/FC/${vers}_perm24
    logdir=$output_dir/logs
    if [ ! -d $logdir ];then mkdir -p $logdir; fi
    cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
        $script_dir/CBIG_ME_calc_HCP_FC.sh $output_dir $mins $vers $sub_txt 25 400"
    # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "24:00:00" -name "HCP_ME_genFC_nc" \
        -mem "16GB" -joberr "$logdir" -jobout "$logdir"	
done

### settings for generating 400-resolution FC matrices using the "uncensored_only" setting
vers='uncensored_only'
min_array=($(seq 2 2 40))  	
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    output_dir=$input_dir/FC/${vers}_perm24
    logdir=$output_dir/logs
    if [ ! -d $logdir ];then mkdir -p $logdir; fi
    cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
        $script_dir/CBIG_ME_calc_HCP_FC.sh $output_dir $mins $vers $sub_txt 25 400"
    # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "24:00:00" -name "HCP_ME_genFC_uo" \
        -mem "16GB" -joberr "$logdir" -jobout "$logdir"	
done

### settings for generating 1000-resolution FC matrices using the "full" setting
vers='full'
min_array=($(seq 2 2 58))	
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    output_dir=$input_dir/FC/${vers}_1000parcels
    logdir=$output_dir/logs
    if [ ! -d $logdir ];then mkdir -p $logdir; fi
    cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
        $script_dir/CBIG_ME_calc_HCP_FC.sh $output_dir $mins $vers $sub_txt 25 1000"
    # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "24:00:00" -name "HCP_ME_genFC_1000parcel" \
        -mem "16GB" -joberr "$logdir" -jobout "$logdir"	
done

### settings for generating 400-resolution FC matrices using the "full" setting for day 1 and day 2 separately
vers='full_day1'
min_array=($(seq 1 1 29))	
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    output_dir=$input_dir/FC/${vers}_perm${perm}
    logdir=$output_dir/logs
    if [ ! -d $logdir ];then mkdir -p $logdir; fi
    cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
        $script_dir/CBIG_ME_calc_HCP_FC.sh $output_dir $mins $vers $sub_txt 2 400"
    # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "24:00:00" -name "HCP_ME_genFC_full" \
        -mem "16GB" -joberr "$logdir" -jobout "$logdir"	
done

vers='full_day2'
min_array=($(seq 1 1 29))	
# submit jobs to scheduler
for mins in ${min_array[@]}; do
    output_dir=$input_dir/FC/${vers}_perm${perm}
    logdir=$output_dir/logs
    if [ ! -d $logdir ];then mkdir -p $logdir; fi
    cmd="source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh; \
        $script_dir/CBIG_ME_calc_HCP_FC.sh $output_dir $mins $vers $sub_txt 2 400"
    # define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "24:00:00" -name "HCP_ME_genFC_full" \
        -mem "16GB" -joberr "$logdir" -jobout "$logdir"	
done