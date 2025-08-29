#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_py38
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

# Check if the second argument is "train"
if [[ "$1" == "train" ]]; then
    # ADNI requires 20 splits
    declare -a sites=("ADNI")
    for i in "${sites[@]}"; do
        echo "Data generation step 4_1, site: $i"
        python -m data_processing.step4_frog_gen \
                --site $i --fold $2 --need_split
    done
else
    # declare an array of sites for external test datasets
    declare -a sites=("AIBL" "MACC" "OASIS")
    for i in "${sites[@]}"; do
        echo "Data generation step 4_2, site: $i"
        python -m data_processing.step4_frog_gen \
                --site $i --independent_test &
    done

    # wait for all background processes to complete
    wait
fi
echo "All commands completed"