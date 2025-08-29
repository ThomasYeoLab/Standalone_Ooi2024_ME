#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_py38
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

################################### Training ###################################
for i in {0..0}; do
    python -m models.L2C_XGBnw.train --fold $i --start_fold $1 --trials 100 --seed 0 --site ADNI --target mmse &
    python -m models.L2C_XGBnw.train --fold $i --start_fold $1 --trials 100 --seed 0 --site ADNI --target vent &
    python -m models.L2C_XGBnw.train --fold $i --start_fold $1 --trials 100 --seed 0 --site ADNI --target clin &
done
# wait for all background processes to complete
wait
echo "All commands completed"

# Remove temporary checkpoints and logs that are not best hyperparameter setup
python -m models.model_utils --util clear_temp --model L2C_XGBnw --site ADNI

############################### Evaluation in-domain ###############################
(
    python -m models.L2C_XGBnw.eval --fold $1 --site ADNI --in_domain --target mmse
    python -m models.L2C_XGBnw.eval --fold $1 --site ADNI --in_domain --target clin
    python -m models.L2C_XGBnw.eval --fold $1 --site ADNI --in_domain --target vent
) &

############################### Evaluation out-domain ###############################
declare -a sites=("AIBL" "MACC" "OASIS")
for s in "${sites[@]}"; do
    (
        python -m models.L2C_XGBnw.eval --fold $1 --site $s --target mmse
        python -m models.L2C_XGBnw.eval --fold $1 --site $s --target clin
        python -m models.L2C_XGBnw.eval --fold $1 --site $s --target vent
    ) &
done

# wait for all background processes to complete
wait
echo "All commands completed"