#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_py38
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

# ADNI training generate both val.pkl and test.pkl
declare -a sites=("ADNI")
for i in "${sites[@]}"; do
    echo "MinimalRNN data generation, site: $i"
    for fold in $(seq 0 19); do
        python -m data_processing.step7_mrnn_gen \
        --site $i --strategy model --fold $fold \
        --in_domain --validation

        python -m data_processing.step7_mrnn_gen \
        --site $i --strategy model --fold $fold \
        --in_domain
    done
done

# declare an array of sites for external test datasets
declare -a sites=("AIBL" "MACC" "OASIS")
for i in "${sites[@]}"; do
    echo "MinimalRNN data generation, site: $i"
    (
        for fold in $(seq 0 19); do
            python -m data_processing.step7_mrnn_gen \
            --site $i --strategy model --fold $fold 
        done
    ) &
done

# wait for all background processes to complete
wait
echo "All commands completed"