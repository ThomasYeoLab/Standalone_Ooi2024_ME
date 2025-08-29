#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_py38
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

# ADNI reads L2C features from 20 folds
declare -a sites=("ADNI")
for i in "${sites[@]}"; do
    echo "Data generation step 5, site: $i"
    python -m data_processing.step5_gauss_data_process \
            --fold $1 --train_site ADNI --site $i
done

# declare an array of sites for external test datasets
declare -a sites=("AIBL" "MACC" "OASIS")
for i in "${sites[@]}"; do
    echo "Data generation step 5, site: $i"
    python -m data_processing.step5_gauss_data_process \
            --fold $1 --train_site ADNI --site $i --independent_test &
done

# wait for all background processes to complete
wait
echo "All commands completed"