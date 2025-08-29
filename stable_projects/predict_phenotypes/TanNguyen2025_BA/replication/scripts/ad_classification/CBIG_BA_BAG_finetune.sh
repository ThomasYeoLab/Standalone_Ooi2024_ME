#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This model-level wrapper script runs any of the following steps
# (listed in order in which they should be ran):
# 1. transfer_weights
# 2. finetune
# 3. extract_features
# 4. BAG_classifier
# for AD classification task.
#
# To conveniently run all steps belonging to this model
# without inserting input arguments, please run
# ${TANNGUYEN2025_BA_DIR}/replication/CBIG_BA_3a_ad_classification.sh
#
# Arguments:
# -s    split_list   : number of different train-val-test splits
#                      (str); example: "0 1 2 3 4"
# -z    sample_size  : number of participants for train+val (dev) set
#                      (str); example: "997"
# -t    walltime     : maximum time duration job can run
#                      (str); example: "100:00:00"
# -f    function_flag: type of function to-be-run
#                      (str); example: "transfer_weights"
#
# Example:
# ssh headnode; ${TANNGUYEN2025_BA_DIR}/replication/scripts/ad_classification/CBIG_BA_BAG_finetune.sh \
# -s "0 1 2 3 4" -z "997" -f "transfer_weights";

################################################################################
# Defining arguments & convenience variables
################################################################################
# Read input arguments
while getopts "s:z:t:f:" arg; do
    case $arg in
        s) split_list=$OPTARG;;
        z) sample_size=$OPTARG;;
        t) walltime=$OPTARG;;
        f) function_flag=$OPTARG;;
        ?)
            echo "Unknown argument"
            exit 1;;
    esac
done

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
    if [[ $function_flag == "finetune" ]]; then
        hours=10
        # For training: estimated hours = number of splits * run time of each split
        run_time=6

        read -r -a split_arr <<< "$split_list"  # Read list of splits to run
        hours=$((${#split_arr[@]} * $run_time))
    else
        hours=10
    fi
    walltime="$hours:00:00"
fi

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

# Preset variables
seed="SEEDID"  # Placeholder for the seed ID
task="ad_classification"
model="BAG_finetune"

# Input
DATA_SPLIT_DIR="${TANNGUYEN2025_BA_DIR}/data/data_split/${task}/${sample_size}/seed_${seed}"
TRAIN_NCI_CSV="${DATA_SPLIT_DIR}/train_nci.csv"
VAL_NCI_CSV="${DATA_SPLIT_DIR}/val_nci.csv"
TEST_NCI_CSV="${DATA_SPLIT_DIR}/test_nci.csv"
CONFIG_CSV="${TANNGUYEN2025_BA_DIR}/data/params/${task}/${model}.csv"  # Hyperparameters
TRAINED_MODEL="${TRAINED_AD_DIR}/best_model/best_auc.pt"

# Output of transfer_weights
MODEL_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}"
INITIALIZED_DIR="${MODEL_DIR}/transfer_weights/initialized_sfcn"
INITIALIZED_MODEL="${INITIALIZED_DIR}/SFCN_initialized.pth" # SFCN initialized with brain age weights

# Output of extracting BAG
FEATURE_EXTRACTED_DIR="${MODEL_DIR}/feature_extract"

# Output of BAG_classifier
CLASSIFY_OUT_DIR="${MODEL_DIR}/BAG_classifier"

################################################################################
# Defining job submission variables
################################################################################
# Config for job
ncpus="4"
ngpus="1"
memory="10gb"
job_name="BAG_finetune"
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
cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"

if [[ "$function_flag" == "transfer_weights" ]]; then
    # Define output directories
    OUT_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}"
    LOG_DIR="$OUT_DIR/log"
    mkdir -p ${LOG_DIR}

    # Construct command
    cmd+="python -m model.pyment.CBIG_BA_transfer_weights --task ${task}"
    cmd+=" --initialized_sfcn_fc_dir ${INITIALIZED_DIR} --model_type SFCN; conda deactivate;"
    job_name="BAG_finetune_TW"

    # Submit job
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit \
    -cmd "$cmd" \
    -walltime "${walltime}" \
    -mem "${memory}" \
    -ncpus "${ncpus}" \
    -name "${job_name}" \
    -joberr "${LOG_DIR}/${job_name}_${timestamp}.err" \
    -jobout "${LOG_DIR}/${job_name}_${timestamp}.out"

elif [[ "$function_flag" == "finetune" ]]; then
    # Define output directories
    OUT_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}/${sample_size}/seed_${seed}"
    LOG_DIR="$(dirname "$OUT_DIR")/log"
    mkdir -p ${LOG_DIR}

    # Construct command
    cmd+="python -m model.cnn.CBIG_BA_train --data_split ${split_list} --data_split_csv ${CONFIG_CSV}"
    cmd+=" --out_dir ${OUT_DIR} --train_csv ${TRAIN_NCI_CSV} --val_csv ${VAL_NCI_CSV} --task ${task}"
    cmd+=" --pretrain_file ${INITIALIZED_MODEL} --finetune_layers 'all' --model_type 'SFCN' --score 'mae'"
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
elif [[ "$function_flag" == "extract_features" ]]; then
    # Define input directories
    TRAINED_MODEL_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}/${sample_size}/seed_${seed}"
    TRAINED_MODEL="${TRAINED_MODEL_DIR}/best_model/best_mae.pt"

    # Define output directories
    OUT_DIR="${FEATURE_EXTRACTED_DIR}/${sample_size}/seed_${seed}"
    LOG_DIR="$(dirname "$OUT_DIR")/log"
    mkdir -p ${LOG_DIR}

    # Construct command
    cmd+="python -m model.log_reg.CBIG_BA_extract_features --data_split ${split_list} --model_name ${model}"
    cmd+=" --input_dir ${DATA_SPLIT_DIR} --output_dir ${OUT_DIR} --trained_model ${TRAINED_MODEL}"
    cmd+=" --task ${task} --feature_dim 1;"
    job_name="BAG_finetune_EF"

    # Submit job
    echo $cmd
    echo "${cmd}" | qsub -V -q gpuQ \
    -l walltime="${walltime}" \
    -l select=1:ncpus="${ncpus}":mem="${memory}":ngpus="${ngpus}":host="gpuserver${node}" \
    -m ae \
    -N "${job_name}" \
    -e "${LOG_DIR}/${job_name}_extract_features_${timestamp}.err" \
    -o "${LOG_DIR}/${job_name}_extract_features_${timestamp}.out"

elif [[ "$function_flag" == "BAG_classifier" ]]; then
    # Define input directories
    PREPARE2CLASSIFY_OUT_DIR="${FEATURE_EXTRACTED_DIR}/997"

    # Define output directories
    LOG_DIR="$(dirname "$CLASSIFY_OUT_DIR")/log"
    mkdir -p ${LOG_DIR}

    # Construct command
    cmd+="python -m model.BAG_models.CBIG_BA_BAG_classifier --in_dir ${PREPARE2CLASSIFY_OUT_DIR}"
    cmd+=" --out_dir ${CLASSIFY_OUT_DIR} --model_name BAG_finetune; conda deactivate;"
    job_name="BAG_classifier"

    # Submit job
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit \
    -cmd "$cmd" \
    -walltime "${walltime}" \
    -mem "${memory}" \
    -ncpus "${ncpus}" \
    -name "${job_name}" \
    -joberr "${LOG_DIR}/${job_name}_${timestamp}.err" \
    -jobout "${LOG_DIR}/${job_name}_${timestamp}.out"
fi
