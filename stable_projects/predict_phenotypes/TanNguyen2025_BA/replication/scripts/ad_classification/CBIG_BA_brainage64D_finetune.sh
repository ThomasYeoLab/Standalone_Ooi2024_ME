#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This model-level wrapper script runs any of the following steps
# (listed in order in which they should be ran):
# 1. train
# 2. test
# for AD classification task.
#
# To conveniently run all steps belonging to this model
# without inserting input arguments, please run
# ${TANNGUYEN2025_BA_DIR}/replication/CBIG_BA_3a_ad_classification.sh
#
# Arguments:
# -d    mode         : set model to "train" or "test" mode
#                      (str); example: "test"
# -s    split_list   : number of different train-val-test splits
#                      (str); example: "0 1 2 3 4"
# -z    sample_size  : number of participants for train+val (dev) set
#                      (str); example: "997"
# -t    walltime     : maximum time duration job can run
#                      (str); example: "100:00:00"
#
# Example:
# ssh headnode; ${TANNGUYEN2025_BA_DIR}/replication/scripts/ad_classification/CBIG_BA_brainage64D_finetune.sh \
# -d "train" -s "0 1 2 3 4" -z "997";

################################################################################
# Defining arguments & convenience variables
################################################################################
# Read input arguments
while getopts "d:s:z:t:" arg; do
    case $arg in
        d) mode=$OPTARG;;  # train or test
        s) split_list=$OPTARG;;
        z) sample_size=$OPTARG;;
        t) walltime=$OPTARG;;
        ?)
            echo "Unknown argument"
            exit 1;;
    esac
done

# Check mode argument is either train or test
if [[ $mode != "train" && $mode != "test" ]]; then 
    echo "Wrong -m, -m can only be train or test."
    exit 1 
fi 
# Set default values for input arguments
# `split_list` = different train-val-test splits
if [[ -z $split_list ]]; then
    # Run on 0 - 49 splits
    split_list=""
    for i in {0..49}; do 
        split_list+="$i "
    done
fi
# `sample_size` = different development set sizes
if [[ -z $sample_size ]]; then
    sample_size="997"
fi
if [[ -z "$walltime" ]]; then
    if [[ $mode == "test" ]]; then
        hours=10
    else
        # For training: estimated hours = number of splits * run time of each split
        run_time=6

        read -r -a split_arr <<< "$split_list"  # Read list of splits to run
        hours=$((${#split_arr[@]} * $run_time))
    fi
    walltime="$hours:00:00"
fi

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

################################################################################
# Defining arguments & convenience variables
################################################################################
# Preset variables
seed="SEEDID"  # Placeholder for the seed ID
task="ad_classification"
model="brainage64D_finetune"

# Input
DATA_SPLIT_DIR="${TANNGUYEN2025_BA_DIR}/data/data_split/${task}/${sample_size}/seed_${seed}"
TRAIN_CSV="${DATA_SPLIT_DIR}/train.csv"
VAL_CSV="${DATA_SPLIT_DIR}/val.csv"
TEST_CSV="${DATA_SPLIT_DIR}/test.csv"
CONFIG_CSV="${TANNGUYEN2025_BA_DIR}/data/params/${task}/${model}.csv"  # Hyperparameters
INITIALIZED_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/ad_classification/brainage64D/transfer_weights/"
INITIALIZED_DIR+="initialized_sfcn_fc/${sample_size}/seed_${seed}"
INITIALIZED_MODEL="${INITIALIZED_DIR}/SFCN_FC_initialized.pth" # SFCN_FC initialized with brain age weights
# Output
OUT_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}/${sample_size}/seed_${seed}"
LOG_DIR="$(dirname "$OUT_DIR")/log"
mkdir -p ${LOG_DIR}

################################################################################
# Defining job submission variables
################################################################################
# Config for job
ncpus="4"
ngpus="1"
memory="10gb"
job_name="brainage64D_finetune"
timestamp=`date +"%Y%m%d_%H%M%S"`

# Select only RTX3090 GPU models
num_used_gpu1=`pbsnodes -a | grep '^gpuserver\|resources_assigned.ngpus' | \
awk '/gpuserver1/{getline; if (/resources_assigned.ngpus/) print}' | cut -d'=' -f2`
num_used_gpu2=`pbsnodes -a | grep '^gpuserver\|resources_assigned.ngpus' | \
awk '/gpuserver2/{getline; if (/resources_assigned.ngpus/) print}' | cut -d'=' -f2`
num_used_gpu3=`pbsnodes -a | grep '^gpuserver\|resources_assigned.ngpus' | \
awk '/gpuserver3/{getline; if (/resources_assigned.ngpus/) print}' | cut -d'=' -f2`
# Send job to shortest GPU queue
if [ $num_used_gpu1 -le $num_used_gpu2 ] && [ $num_used_gpu1 -le $num_used_gpu3 ]; then
    node=1
elif [ $num_used_gpu2 -le $num_used_gpu1 ] && [ $num_used_gpu2 -le $num_used_gpu3 ]; then
    node=2
else
    node=3
fi

# Use same conda environment for all steps
cmd="cd $TANNGUYEN2025_BA_DIR;source CBIG_init_conda; conda activate CBIG_BA;"
if [[ $mode == "train" ]]; then 
    # Construct run command from the data paths and hyperparameters
    cmd+="python -m model.cnn.CBIG_BA_train --data_split ${split_list} --data_split_csv ${CONFIG_CSV}"
    cmd+=" --out_dir ${OUT_DIR} --train_csv ${TRAIN_CSV} --val_csv ${VAL_CSV} --task ${task}"
    cmd+=" --pretrain_file ${INITIALIZED_MODEL} --finetune_layers 'all'"
    cmd+=";conda deactivate"

    # Submit job
    echo $cmd
    echo "${cmd}" | qsub -V -q gpuQ \
    -l walltime="${walltime}" \
    -l select=1:ncpus="${ncpus}":mem="${memory}":ngpus="${ngpus}":host="gpuserver${node}" \
    -m ae \
    -N "${job_name}" \
    -e "${LOG_DIR}/${job_name}_train_${timestamp}.err" \
    -o "${LOG_DIR}/${job_name}_train_${timestamp}.out"
    sleep 3
else
    # Construct run command from the data paths
    cmd+="python -m model.cnn.CBIG_BA_test --data_split ${split_list} --model_dir ${OUT_DIR}"
    cmd+=" --test_csv ${TEST_CSV} --task ${task}"
    cmd+=";conda deactivate"

    # Submit job
    echo $cmd
    echo "${cmd}" | qsub -V -q gpuQ \
    -l walltime="${walltime}" \
    -l select=1:ncpus="${ncpus}":mem="${memory}":ngpus="${ngpus}":host="gpuserver${node}" \
    -m ae \
    -N "${job_name}" \
    -e "${LOG_DIR}/${job_name}_test_${timestamp}.err" \
    -o "${LOG_DIR}/${job_name}_test_${timestamp}.out"
    sleep 3
fi
