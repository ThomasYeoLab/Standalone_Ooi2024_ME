#!/bin/sh

#####
# This script submits job to the schduler to run the specified type of reliability interpretation. 
# These scripts are for the ABCD dataset. 
#
# Input:
#     -vers:
#      Indicates the version of the prediction procedure that was run. Can be either "full" or "random"
#
#     -method: 
#      Indicates the analysis method of calculating reliability. Can be "BWA" for univariate and "Haufe"
#      for multivariate.
#
# EXAMPLE: 
#	CBIG_ME_ABCD_schedule_interpretation.sh BWA full
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
source $ME_CODE_DIR/replication/config/CBIG_ME_tested_config.sh
scriptdir=$ME_CODE_DIR/reliability/ABCD

### read function inputs
vers=$1
method=$2
# modify directories dependent on input	
input_dir=$ME_rep_dir/ABCD/input
outdir=$ME_rep_dir/ABCD/output_splithalf/$vers/
logdir=$ME_rep_dir/ABCD/scheduler_logs_tmp
if [ ! -d $logdir ]; then mkdir -p $logdir; fi
# other settings and paths
min_array=($(seq 2 2 20))
subs_array=(200 400 600 800 1000 2565)
pheno_array=($(seq 1 1 37))
reg='KRR'

### submit jobs to scheduler
for min in ${min_array[@]}; do
	for subs in ${subs_array[@]}; do
		min_str="${min}min"
		sub_str="${subs}_subjects"
                for b in ${pheno_array[@]}; do
                    # command based on method
                    if [ $method == "BWA" ]; then
                        cmd="$scriptdir/CBIG_ME_ABCD_BWA.sh $input_dir $outdir $vers $min_str $sub_str $b"
                        jobname="ABCD_BWA"
                        output_file=$outdir/interpretation/$min_str/$sub_str/KRR_tstats_mat_behav${b}.mat
                    elif [ $method == "Haufe" ]; then
                        cmd="$scriptdir/CBIG_ME_ABCD_Haufe.sh $input_dir $outdir $vers $min_str $sub_str $reg $b"
                        jobname="ABCD_Haufe"
                        output_file=$outdir/interpretation/$min_str/$sub_str/${reg}_cov_mat_behav${b}.mat
                fi
		### define run job function: MODIFY THIS IF RUNNING ON YOUR OWN SCHEDULER
                # check if output file exists
                if [ ! -f $output_file ]; then
		    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "${cmd}" -walltime "72:00:00" -name "$jobname" \
		        -mem "24GB" -joberr "$logdir" -jobout "$logdir"	
                else
                    echo "Final output file exists, skipping..."
                fi
            done
	done
done