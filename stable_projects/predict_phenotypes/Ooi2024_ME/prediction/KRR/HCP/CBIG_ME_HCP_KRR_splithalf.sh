#!/bin/sh

#####
# This wrapper script submits job to the schduler to run KRR in the HCP dataset.
# This runs the reliability procedure (2 fold cross validation).
#
# Input:
#    -seed: 
#     An integer of the the random seed used to split the folds
#
#    -min:
#     An integer indicating which FC to run regression for.
#
#    -vers:
#     The manner in which FC was calculated (See FC generation scripts). Can be "full" or "random".
#
# EXAMPLE: 
#    CBIG_ME_HCP_KRR_splithalf.sh 1 58 full
#
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

### set up data directories
ME_CODE_DIR=$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Ooi2024_ME
scriptdir=$ME_CODE_DIR/prediction/KRR/HCP
projectdir=$ME_rep_dir/HCP
REPDIR=$CBIG_REPDATA_DIR/stable_projects/predict_phenotypes/Ooi2022_MMP/MMP_HCP_data/data/behaviors

### read function inputs
seed=$1
min=$2
vers=$3

### set other params
innerFolds=10
outerFolds=2
subtxt='subject_list_792.txt'
scorecsv=$projectdir/input/HCP_leaveout_componentscores_240604.csv
restrictedcsv=$REPDIR/RESTRICTED_jingweili_4_12_2017_1200subjects_fill_empty_zygosityGT_by_zygositySR.csv
predvar=$REPDIR/MMP_HCP_y_variables.txt
covtxt=$REPDIR/MMP_HCP_covariates.txt
ymat='subject792_variables_to_predict.mat'
covmat='subject792_covariates.mat'
outstem="KRR_$min"

# echo settings for job submission log file
echo "Project Directory set to: $projectdir"
echo "seed: $seed"
echo "outstem: $outstem"

### Create log file and save params
mkdir -p $projectdir/output_splithalf/$vers/seed_${seed}/logs
LF="$projectdir/output_splithalf/$vers/seed_${seed}/logs/${outstem}_${seed}.log"
if [ -f $LF ]; then rm $LF; fi

# echo settings for script log file
echo "innerFolds = $innerFolds" >> $LF
echo "outerFolds = $outerFolds" >> $LF
echo "seed = $seed" >> $LF
echo "min = $min" >> $LF
echo "subtxt = $subtxt" >> $LF
echo "scorecsv = $scorecsv" >> $LF
echo "restrictedcsv = $restrictedcsv" >> $LF
echo "predvar = $predvar" >> $LF
echo "covtxt = $covtxt" >> $LF
echo "ymat = $ymat" >> $LF
echo "covmat = $covmat" >> $LF

### Call matlab function
matlab -nodesktop -nosplash -nodisplay -r " try addpath('$scriptdir'); CBIG_ME_HCP_KRR_splithalf( \
    $outerFolds, $innerFolds, $seed, $min, '$vers', '$projectdir', '$subtxt', '$scorecsv', \
    '$restrictedcsv', '$predvar', '$covtxt', '$ymat','$covmat'); catch ME; display(ME.message); \
    end; exit; " >> $LF 2>&1
