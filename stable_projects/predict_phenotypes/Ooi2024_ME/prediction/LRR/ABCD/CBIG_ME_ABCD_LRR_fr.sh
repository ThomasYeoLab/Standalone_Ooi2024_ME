#!/bin/sh

#####
# This wrapper script submits job to the scheduler to run LRR in the ABCD dataset.
# This runs the prediction procedure (10 site choose 3 cross validation).
#
# Input:
#    -min:
#     An integer indicating which FC to run regression for.
#
#    -vers:
#     The manner in which FC was calculated (See FC generation scripts). For LRR, we only run
#     the "full" analysis.
#
# EXAMPLE: 
#    CBIG_ME_ABCD_LRR.sh 20 full
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
scriptdir=$ME_CODE_DIR/prediction/LRR/ABCD
projectdir=$ME_rep_dir/ABCD
REPDIR=$CBIG_REPDATA_DIR/stable_projects/predict_phenotypes/Ooi2022_MMP/MMP_ABCD_data/data/behaviors

### read function inputs
min=$1
vers=$2
sample=$3

### set other params
num_sites=3
innerFolds=10
subtxt='subject_list_2565.txt'
subcsv='ABCD_leaveout_2565subs_componentscores_240604.csv'
predvar=$REPDIR/MMP_ABCD_y_variables.txt
covtxt=$REPDIR/MMP_ABCD_covariates.txt
ymat='subject2565_variables_to_predict.mat'
covmat='subject2565_covariates.mat'
outstem="LRR_$min"

# echo settings for job submission log file
echo "Project Directory set to: $projectdir"
echo "outstem: $outstem"

### Create log file and save params
mkdir -p $projectdir/output/$vers/logs
LF="$projectdir/output/$vers/logs/${outstem}_sample${sample}.log"
if [ -f $LF ]; then rm $LF; fi

# echo settings for script log file
echo "num_sites = $num_sites" >> $LF
echo "innerFolds = $innerFolds" >> $LF
echo "min = $min" >> $LF
echo "subtxt = $subtxt" >> $LF
echo "subcsv = $subcsv" >> $LF
echo "predvar = $predvar" >> $LF
echo "covtxt = $covtxt" >> $LF
echo "ymat = $ymat" >> $LF
echo "covmat = $covmat" >> $LF
echo "sample = $sample" >> $LF

### Call matlab function
matlab -nodesktop -nosplash -nodisplay -r " try addpath('$scriptdir'); CBIG_ME_ABCD_LRR_fr($num_sites, \
   $innerFolds, $min, '$vers', '$projectdir', '$subtxt', '$subcsv', \
   '$predvar', '$covtxt', '$ymat','$covmat', $sample); catch ME; display(ME.message); \
   end; exit; " >> $LF 2>&1
