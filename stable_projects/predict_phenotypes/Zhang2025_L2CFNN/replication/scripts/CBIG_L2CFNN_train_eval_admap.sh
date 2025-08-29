#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_ADMap
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

################################### Training ###################################
python -m models.AD_Map.train \
--fold $1 --seed 0 --site ADNI --trials 50

# Remove temporary checkpoints and logs that are not best hyperparameter setup
python -m models.model_utils --util clear_temp --model AD_Map --site ADNI

############################### Evaluation in-domain ###############################
python -m models.AD_Map.evaluation \
--fold $1 --site ADNI --in_domain &

############################### Evaluation out-domain ###############################
declare -a sites=("AIBL" "MACC" "OASIS")
for i in "${sites[@]}"; do
    python -m models.AD_Map.evaluation \
    --fold $1 --site $i &
done

# wait for all background processes to complete
wait
echo "All commands completed"