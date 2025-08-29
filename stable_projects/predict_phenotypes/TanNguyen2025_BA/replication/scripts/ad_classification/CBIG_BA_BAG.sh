#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This model-level wrapper script runs any of the following steps
# (listed in order in which they should be ran):
# 1. extract_features
# 2. prepare2classify
# 3. BAG_classifier
# for AD classification task.
#
# To conveniently run all steps belonging to this model
# without inserting input arguments, please run
# ${TANNGUYEN2025_BA_DIR}/replication/CBIG_BA_3a_ad_classification.sh
#
# Arguments:
# -f    function_flag: type of function to-be-run
#                      (str); example: "extract_features"
#
# Example:
# ssh headnode; ${TANNGUYEN2025_BA_DIR}/replication/scripts/ad_classification/CBIG_BA_BAG.sh -f "extract_features";

################################################################################
# Defining arguments & convenience variables
################################################################################
# Read input arguments
while getopts "f:" arg; do
    case $arg in
        f) function_flag=$OPTARG;;
        ?)
            echo "Unknown argument"
            exit 1;;
    esac
done

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
model="BAG"

# Set up relevant paths
MODEL_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}"
LOG_DIR="${MODEL_DIR}/log"
mkdir -p ${LOG_DIR} &> /dev/null

# Output of feature extraction
EXTRACT_OUT_DIR="${MODEL_DIR}/feature_extract"
# Output of feature extraction
PREPARE2CLASSIFY_OUT_DIR="${MODEL_DIR}/prepare2classify"
# Output of BAG_classifier
CLASSIFY_OUT_DIR="${MODEL_DIR}/BAG_classifier"

################################################################################
# Defining job submission variables
################################################################################
# Config for job
ncpus="4"
memory="10gb"
walltime="5:00:00"
timestamp=`date +"%Y%m%d_%H%M%S"`

# Use same conda environment for all steps
cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"

if [[ "$function_flag" == "extract_features" ]]; then
    # Construct command
    cmd+="python -m model.pyment.CBIG_BA_extract_features --task ${task} --in_dir ${EXTRACT_OUT_DIR}"
    cmd+=" --out_dir ${EXTRACT_OUT_DIR} --feature_extract 0 --model_name ${model};"
    job_name="BAG_EF"
elif [[ "$function_flag" == "prepare2classify" ]]; then
    # Construct command
    cmd+="python -m model.BAG_models.CBIG_BA_prepare2classify --in_dir ${EXTRACT_OUT_DIR}"
    cmd+=" --out_dir ${PREPARE2CLASSIFY_OUT_DIR} --task ${task} --full_trainval_size 997;"
    job_name="BAG_prepare2classify"
elif [[ "$function_flag" == "BAG_classifier" ]]; then
    # Construct command
    cmd+="python -m model.BAG_models.CBIG_BA_BAG_classifier --in_dir ${PREPARE2CLASSIFY_OUT_DIR}"
    cmd+=" --out_dir ${CLASSIFY_OUT_DIR} --model_name BAG;"
    job_name="BAG_classifier"
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
