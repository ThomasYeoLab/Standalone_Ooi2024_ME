#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_py38
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

# Baseline same across 20 folds, so all sites together
declare -a sites=("ADNI" "MACC" "OASIS" "AIBL")
for i in "${sites[@]}"; do
    echo "Data generation step 2, site: $i"
    python -m data_processing.step2_baseline_gen --site $i &
done

# wait for all background processes to complete
wait
echo "All commands completed"