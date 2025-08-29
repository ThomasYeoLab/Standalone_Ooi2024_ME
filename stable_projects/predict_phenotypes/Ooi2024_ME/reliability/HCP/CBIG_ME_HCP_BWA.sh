#!/bin/sh

#####
# This script calls the matlab function to run the univariate t-stats for KRR models. 
# Specify the input directory with FC matrices, the output directory with the results,
# the version of the analysis that was run (either full or random), the number of minutes,
# sample size and index of the phenotype.
# 
# EXAMPLE: 
#	CBIG_ME_HCP_BWA.sh /my_input_dir/HCP/input/ /my_results_dir/HCP/output/ full
#                                "2_mins" "200_subjects" 1
#	
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh
scriptdir=$ME_CODE_DIR/reliability/HCP

# set params
input_dir=$1
results_dir=$2
vers=$3
num_mins=$4
num_subs=$5
behav_num=$6

# Create log file and save params
output_dir=$results_dir/interpretation/logs
if [ ! -d $output_dir ];then mkdir -p $output_dir; fi
LF="$output_dir/BWA_${num_mins}_${num_subs}_behav${behav_num}_interpretation.log"
if [ -f $LF ]; then rm $LF; fi
echo "input_dir = $input_dir" >> $LF
echo "results_dir = $results_dir" >> $LF
echo "num_mins = $num_mins" >> $LF
echo "num_subs = $num_subs" >> $LF

### Call matlab function
matlab -nodesktop -nosplash -nodisplay -r " try addpath('$scriptdir'); CBIG_ME_HCP_BWA( \
   '$input_dir','$results_dir', '$vers', '$num_mins', '$num_subs', $behav_num); catch ME; display(ME.message); \
   end; exit; " >> $LF 2>&1