#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
source activate CBIG_Zhang2025_py38
module load cuda/11.0
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"

# Make sure the results are not obtained on GPU server 4 (A5000)
echo "Running job on: ""$(hostname)"
if [ "$(hostname)" == "gpuserver4" ]; then
    exit
fi

################################### Training ###################################
for f in {0..3}; do
    python -m models.L2C_FNN.train \
    --fold $f --start_fold $1 \
    --epochs 100 --trials 100 --batch_size 512 \
    --seed 3 --input_dim 101 --site ADNI &
done
# wait for all background processes to complete
wait
echo "All commands completed"

# Remove temporary checkpoints and logs that are not best hyperparameter setup
python -m models.model_utils --util clear_temp --model L2C_FNN --site ADNI

############################### Evaluation in-domain ###############################
for f in {0..3}; do
    python -m models.L2C_FNN.evaluation \
      --fold $f --site ADNI --start_fold $1 --batch_size 512 \
      --input_dim 101 --in_domain &
done

# wait for all background processes to complete
wait
echo "All commands completed"

############################### Evaluation out-domain ###############################
declare -a sites=("AIBL" "MACC" "OASIS")
for f in {0..3}; do
    for i in "${sites[@]}"; do
        python -m models.L2C_FNN.evaluation \
        --fold $f --site $i --start_fold $1 --batch_size 512 \
        --input_dim 101 &
    done
    # wait for all background processes to complete
    wait
    echo "All commands completed"
done