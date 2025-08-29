#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_py38
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

# For 20-split generation (train)
echo "Data generation step 6, site: ADNI"
python -m data_processing.step6_gt_filter --site ADNI --need_split &

# For external test datasets
declare -a sites=("MACC" "OASIS" "AIBL")
for i in "${sites[@]}"; do
    echo "Data generation step 6, site: $i"
    python -m data_processing.step6_gt_filter --site $i &
done

# wait for all background processes to complete
wait
echo "All commands completed"