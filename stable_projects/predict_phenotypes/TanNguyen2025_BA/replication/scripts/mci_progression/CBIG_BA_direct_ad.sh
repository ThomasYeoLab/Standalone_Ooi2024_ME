#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This model-level wrapper script runs any of the following steps
# (listed in order in which they should be ran):
# 1. extract_features
# 2. logistic_regression
# for MCI progression prediction task.
#
# To conveniently run all steps belonging to this model
# without inserting input arguments, please run
# ${TANNGUYEN2025_BA_DIR}/replication/CBIG_BA_3b_mci_progression.sh
#
# Arguments:
# -s    split_list   : number of different train-val-test splits
#                      (str); example: "0 1 2 3 4"
# -z    sample_size  : number of participants for train+val (dev) set
#                      (str); example: "448"
# -f    logreg_flag  : (int) if 0, run feature_extract.
#                            if 1, run logistic_regression.
#
# Example:
# ssh headnode; ${TANNGUYEN2025_BA_DIR}/replication/scripts/ad_classification/CBIG_BA_direct_ad.sh \
# -s "0 1 2 3 4" -z "448" -f 0;

################################################################################
# Defining arguments & convenience variables
################################################################################
# Read input arguments
while getopts "s:z:f:" arg; do
    case $arg in
        s) split_list=$OPTARG;;
        z) sample_size=$OPTARG;;
        f) logreg_flag=$OPTARG;;
        ?)
            echo "Unknown argument"
            exit 1;;
    esac
done

# Set default values for input arguments
if [[ -z $split_list ]]; then
    # Run on 0 - 49 splits
    split_list=""
    for i in {0..49}; do 
        split_list+="$i "
    done
fi
if [[ -z $sample_size ]]; then
    sample_size="448"
fi

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

# Preset variables
seed="SEEDID"  # Placeholder for the seed ID
task="mci_progression"
model="direct_ad"

# Set up relevant paths
DATA_SPLIT_DIR="${TANNGUYEN2025_BA_DIR}/data/data_split/${task}/${sample_size}/seed_${seed}"

# Trained model used to extract features
TRAINED_AD_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/ad_classification/direct/997/seed_${seed}"
TRAINED_MODEL="${TRAINED_AD_DIR}/best_model/best_auc.pt"

# Output of logistic regression
FEATURE_EXTRACTED_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}/${sample_size}/seed_${seed}"
OUT_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${task}/${model}/${sample_size}/seed_${seed}"

# Create log directory
LOG_DIR="$(dirname "$OUT_DIR")/log"
mkdir -p ${LOG_DIR}

################################################################################
# Defining job submission variables
################################################################################
# Config for job
ncpus="4"
ngpus="1"
memory="10gb"
walltime="5:00:00"
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

if [[ "$logreg_flag" -eq 1 ]]; then
    # Construct command
    cmd+="python -m model.log_reg.CBIG_BA_logistic_regression --input_dir ${FEATURE_EXTRACTED_DIR}"
    cmd+=" --output_dir ${OUT_DIR} --model_name ${model};"
    cmd+="conda deactivate;"
    job_name="direct_ad_logreg"

    # Submit job
    echo "${cmd}" | qsub -V -q gpuQ \
    -l walltime="${walltime}" \
    -l select=1:ncpus="${ncpus}":mem="${memory}":ngpus="${ngpus}":host="gpuserver${node}" \
    -m ae \
    -N "${job_name}" \
    -e "${LOG_DIR}/${job_name}_logistic_regression_${timestamp}.err" \
    -o "${LOG_DIR}/${job_name}_logistic_regression_${timestamp}.out"
    sleep 3    
else
    # Construct command
    cmd+="python -m model.log_reg.CBIG_BA_extract_features --data_split ${split_list} --model_name ${model}"
    cmd+=" --input_dir ${DATA_SPLIT_DIR} --output_dir ${FEATURE_EXTRACTED_DIR}"
    cmd+=" --trained_model ${TRAINED_MODEL} --task ${task};"
    cmd+="conda deactivate;"
    job_name="direct_ad_pre_logreg"

    # Submit job
    echo "${cmd}" | qsub -V -q gpuQ \
    -l walltime="${walltime}" \
    -l select=1:ncpus="${ncpus}":mem="${memory}":ngpus="${ngpus}":host="gpuserver${node}" \
    -m ae \
    -N "${job_name}" \
    -e "${LOG_DIR}/${job_name}_extract_features_${timestamp}.err" \
    -o "${LOG_DIR}/${job_name}_extract_features_${timestamp}.out"
    sleep 3    
fi
