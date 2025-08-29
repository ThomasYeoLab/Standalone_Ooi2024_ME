#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This model-level wrapper script runs any of the following steps
# (listed in order in which they should be ran):
# 1. extract_features
# 2. transfer_weights
# 3. logistic_regression
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
# -f    function_flag: type of function to-be-run
#                      (str); example: "extract_features"
#
# Example:
# ssh headnode; ${TANNGUYEN2025_BA_DIR}/replication/scripts/ad_classification/CBIG_BA_brainage64D.sh \
# -s "0 1 2 3 4" -z "997" -f "extract_features";

################################################################################
# Defining arguments & convenience variables
################################################################################
# Read input arguments
while getopts "s:z:f:" arg; do
    case $arg in
        s) split_list=$OPTARG;;
        z) sample_size=$OPTARG;;
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

if [[ -z $function_flag ]]; then
    function_flag="extract_features"
fi

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

# Preset variables
seed="SEEDID"  # Placeholder for the seed ID
task="ad_classification"
model="brainage64D"
num_split=$(echo "$split_list" | grep -oE '[0-9]+' | wc -l)


# Set up relevant paths
MODEL_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}"
LOG_DIR="${MODEL_DIR}/log"
mkdir -p ${LOG_DIR} &> /dev/null

# Output of feature extraction
EXTRACT_OUT_DIR="${MODEL_DIR}/feature_extract"

# Output of transfer weights
TRANSFER_ROOT_DIR="${MODEL_DIR}/transfer_weights"
TRANSFER_LOGREG_IN_DIR="${TRANSFER_ROOT_DIR}/logreg_in"
TRANSFER_LOGREG_OUT_DIR="${TRANSFER_ROOT_DIR}/logreg_out"
TRANSFER_INIT_DIR="${TRANSFER_ROOT_DIR}/initialized_sfcn_fc"

# Output of logistic regression
LOGREG_OUT_DIR="${MODEL_DIR}/log_reg"
LOGREG_SIZE_OUT_DIR="${LOGREG_OUT_DIR}/${sample_size}/seed_${seed}"

################################################################################
# Defining job submission variables
################################################################################
# Config for job
ncpus="4"
memory="10gb"
walltime="5:00:00"
timestamp=`date +"%Y%m%d_%H%M%S"`

# Construct command
if [[ "$function_flag" == "extract_features" ]]; then
    cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate pyment;"
    cmd+="python -m model.pyment.CBIG_BA_extract_features --task ${task} --in_dir ${EXTRACT_OUT_DIR}"
    cmd+=" --out_dir ${EXTRACT_OUT_DIR} --feature_extract 1 --model_name ${model};"
    job_name="brainage64D_EF"
elif [[ "$function_flag" == "transfer_weights" ]]; then
    cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate pyment_v1;"
    cmd+="python -m model.pyment.CBIG_BA_transfer_weights --task ${task} --in_dir ${EXTRACT_OUT_DIR}"
    cmd+=" --logreg_in_dir ${TRANSFER_LOGREG_IN_DIR} --logreg_out_dir ${TRANSFER_LOGREG_OUT_DIR}"
    cmd+=" --initialized_sfcn_fc_dir ${TRANSFER_INIT_DIR} --logreg_pred ${LOGREG_OUT_DIR} --num_split ${num_split}"
    cmd+=" --step 50 --current_trainval_size ${sample_size} --full_trainval_size 997 --model_type SFCN_FC;"
    job_name="brainage64D_TW"
elif [[ "$function_flag" == "logistic_regression" ]]; then
    cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"
    cmd+="python -m model.log_reg.CBIG_BA_logistic_regression --input_dir ${LOGREG_SIZE_OUT_DIR}"
    cmd+=" --output_dir ${LOGREG_SIZE_OUT_DIR} --model_name ${model};"
    job_name="brainage64D_LR"
fi
cmd+="conda deactivate;"

# Submit job
$CBIG_CODE_DIR/setup/CBIG_pbsubmit \
-cmd "$cmd" \
-walltime "${walltime}" \
-mem "${memory}" \
-ncpus "${ncpus}" \
-name "${job_name}" \
-joberr "${LOG_DIR}/${job_name}_${timestamp}.err" \
-jobout "${LOG_DIR}/${job_name}_${timestamp}.out"
