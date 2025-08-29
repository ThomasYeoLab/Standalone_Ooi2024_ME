#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_py38
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

# ADNI requires 20 splits
declare -a sites=("ADNI")
for i in "${sites[@]}"; do
    echo "Data generation step 1, site: $i"
    python -m data_processing.step1_fold_split \
            --nb_folds 20 --site $i --need_split
done

# declare an array of sites for external test datasets
declare -a sites=("AIBL" "MACC" "OASIS")
for i in "${sites[@]}"; do
    echo "Data generation step 1, site: $i"
    python -m data_processing.step1_fold_split \
            --site $i --independent_test &
done

# wait for all background processes to complete
wait
echo "All commands completed"