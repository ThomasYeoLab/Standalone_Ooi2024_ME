#!/bin/sh

#####
# This wrapper script submits job to the schduler to run KRR in the ABCD dataset.
# This runs the reliability procedure (10 site choose 5 cross validation).
#
# Input:
#    -min:
#     An integer indicating which FC to run regression for.
#
#    -vers:
#     The manner in which FC was calculated (See FC generation scripts). Can be "full", "no_censoring",
#     "uncensored_only" or "random".
#
# EXAMPLE: 
#    CBIG_ME_ABCD_KRR_splithalf.sh 20 full
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#####

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
scriptdir=$ME_CODE_DIR/prediction/KRR/ABCD
projectdir=$ME_rep_dir/ABCD
REPDIR=$CBIG_REPDATA_DIR/stable_projects/predict_phenotypes/Ooi2022_MMP/MMP_ABCD_data/data/behaviors

### read function inputs
min=$1
vers=$2

### set other params
num_sites=5
innerFolds=10
subtxt='subject_list_2565.txt'
subcsv='ABCD_leaveout_2565subs_componentscores_240604.csv'
predvar=$REPDIR/MMP_ABCD_y_variables.txt
covtxt=$REPDIR/MMP_ABCD_covariates.txt
ymat='subject2565_variables_to_predict.mat'
covmat='subject2565_covariates.mat'
outstem="KRR_$min"

# echo settings for job submission log file
echo "Project Directory set to: $projectdir"
echo "outstem: $outstem"

### Create log file and save params
mkdir -p $projectdir/output_splithalf/$vers/logs
LF="$projectdir/output_splithalf/$vers/logs/${outstem}.log"
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

### Call matlab function
if [ $vers == "random" ]; then
    matlab -nodesktop -nosplash -nodisplay -r " try addpath('$scriptdir'); CBIG_ME_ABCD_KRR_randomFC_splithalf( \
    $num_sites, $innerFolds, $min, '$vers', '$projectdir', '$subtxt', '$subcsv', \
    '$predvar', '$covtxt', '$ymat','$covmat'); catch ME; display(ME.message); \
    end; exit; " >> $LF 2>&1
else
    matlab -nodesktop -nosplash -nodisplay -r " try addpath('$scriptdir'); CBIG_ME_ABCD_KRR_splithalf( \
    $num_sites, $innerFolds, $min, '$vers', '$projectdir', '$subtxt', '$subcsv', \
    '$predvar', '$covtxt', '$ymat','$covmat'); catch ME; display(ME.message); \
    end; exit; " >> $LF 2>&1
fi
