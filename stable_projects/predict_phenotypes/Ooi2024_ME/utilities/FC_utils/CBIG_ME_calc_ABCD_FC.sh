#!/bin/sh

#####
# This script calls the matlab function to run generate FC matrices in the ABCD dataset.
#
# Input:
#    -output_dir: 
#     Path to store FC matrices
#
#    -mins:
#     The scan duration (in minutes) that is used to calculate the FC
#
#    -vers: 
#     The manner in which to calculate the the first t frames of data used to generate the FC. Can be 
#     one of the following
#     "full":           FC matrices are generated from T mins. Censoring occurs after T mins are grabbed
#                       from the time series.
#     "no_censoring":   FC matrices are generated from T mins. No censoring is used.
#     "uncesored_only": FC matrices are generated from T mins. Censoring occurs before T mins are grabbed 
#                       from the time series.
#
#    -sub_txt:
#     A text file containing the participant IDs for participants used in the analysis
#
#    -perm_order: 
#     A number between 1 and (#runs)!. Since there are 4 runs in the ABCD this should be between 1-24.                  
#
#    -batch: 
#     Which batch (in 1000s) to process the FC for. This is to allow for submission of mulitple jobs
#     to speed up processing.
#
#    -res: 
#    An integer representing parcellation resolution. Can be either 400 or 1000.
#
# EXAMPLE: 
#    CBIG_ME_calc_ABCD_FC.sh /home/leon_ooi/storage/optimal_prediction/replication/ABCD/input 20 \
#        full /home/leon_ooi/storage/optimal_prediction/replication/ABCD/input/subject_list_2565.txt 1 400
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
script_dir=$(dirname "$(readlink -f "$0")")

### read function inputs
output_dir=$1
mins=$2
vers=$3
sub_txt=$4
perm_order=$5
batch=$6
res=$7

### Create log file 
log_dir=$output_dir/logs
if [ ! -d $log_dir ];then mkdir -p $log_dir; fi
LF="$log_dir/${mins}mins_FC_batch${batch}.log"
if [ -f $LF ]; then rm $LF; fi

### Call matlab function
matlab -nodesktop -nosplash -nodisplay -r " try addpath('$script_dir'); \
    CBIG_ME_calc_ABCD_FC( '$sub_txt', $mins, '$vers', $perm_order, $batch, $res); catch ME; \
    display(ME.message); end; exit; " >> $LF 2>&1
