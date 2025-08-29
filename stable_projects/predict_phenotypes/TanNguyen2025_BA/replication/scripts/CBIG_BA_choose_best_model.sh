#!/bin/bash
# Written by Kim-Ngan Nguyen and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This script calls Python function
# ${TANNGUYEN2025_BA_DIR}/model/utils/CBIG_BA_choose_best_model.py.
# `CBIG_BA_choose_best_model.py` creates a symbolic link pointing to
# the highest validation set performance.
#
# This script is not called directly on command line by user,
# but is called by the task-level wrapper scripts (e.g.,
# ${TANNGUYEN2025_BA_DIR}/replication/CBIG_BA_3a_ad_classification.sh and/or
# ${TANNGUYEN2025_BA_DIR}/replication/CBIG_BA_3b_mci_progression.sh).

# Read input arguments
while getopts "n:z:m:t:s:" arg; do
    case $arg in
        n) num_split=$OPTARG;;
        z) sample_size=$OPTARG;;
        m) model_name=$OPTARG;;
        t) task_name=$OPTARG;;
        s) summary_test=$OPTARG;;
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
# `size_list` = different development set sizes
if [[ -z $sample_size ]]; then
    sample_size="997"
fi

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

if [[ -z $summary_test ]]; then
    summary_test="y"
fi

# Define convenience variables
UTIL_DIR="${TANNGUYEN2025_BA_DIR}/model/utils"
REPLICATION_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task_name}"
MODEL_DIR="${REPLICATION_DIR}/${model_name}"
LOG_DIR="${MODEL_DIR}/choose_best_model_log"
mkdir -p ${LOG_DIR}

# Define job submission arguments
ncpus="2"
memory="8gb"
walltime="5:00:00"
timestamp=`date +"%Y%m%d_%H%M%S"`
job_name="${model_name}_choose_best_model"

# Construct job submission command
cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"
cmd+="python ${UTIL_DIR}/CBIG_BA_choose_best_model.py --base_dir="${MODEL_DIR}/${sample_size}""
cmd+=" --seed_max "${num_split}" --create_link="y" --summary_test="$summary_test";"
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
