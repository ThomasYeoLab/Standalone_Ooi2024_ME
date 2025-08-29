#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This task-level wrapper script will run each model-level wrapper script,
# to fully replicate all 3 MCI progression prediction models:
# 1. Direct (note this is different from AD classification Direct
#    in the sense that AD classification Direct was trained
#    directly on AD classification, but MCI progression prediction
#    Direct was trained directly on MCI progression prediction)
# 2. Direct-AD
# 3. Brainage64D-finetune-AD
#
# If user aims to fully replicate MCI progression prediction work,
# this script requires no arguments except for the date & time
# in string format for example "250423_1302" to represent
# 23rd April 2025, 1:02 p.m.
#
# Example:
# ssh headnode; ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3b_mci_progression.sh -d "250423_1302";

# Function to check if previous step completed successfully
# Essentially, this checks whether the empty .txt file (generated
# at the end of each step) was successfully generated.
# If the step is prematurely terminated, the empty .txt file
# will not be generated.
check_completed_step() {
    local OUT_DIR="$1"        # String: The output directory
    local append="$2"         # String: The append value
    local split_list=("${@:3}")   # Array: List of integers for split_list
    local size_list=("${@:4}")    # Array: List of integers for size_list

    split_count=$(echo "$split_list" | wc -w)
    size_count=$(echo "$size_list" | wc -w)
    correct_count=$((split_count * size_count))

    count=0
    while [[ "$count" -lt "$correct_count" ]]; do
        count=0
        for split in $split_list; do
            for sample_size in $size_list; do
                COMPLETE_FILE="${OUT_DIR}/${sample_size}/seed_${split}/complete_${append}"
                if [ -e "$COMPLETE_FILE" ]; then
                    ((count=count+1))
                fi
            done
        done
    done
}

# Read input arguments
while getopts "s:z:d:" arg; do
    case $arg in
        s) split_list=$OPTARG;;
        z) size_list=$OPTARG;;
        d) datetime=$OPTARG;;
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
if [[ -z $size_list ]]; then
    size_list="50 100 150 200 250 300 350 400 448"
fi

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

# Set convenience variables
num_split=$(echo "$split_list" | grep -oE '[0-9]+' | wc -l)
WRAPPER_DIR="${TANNGUYEN2025_BA_DIR}/replication/scripts"
MCIPROGRESSION_DIR="${WRAPPER_DIR}/mci_progression"
REPLICATION_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/mci_progression"
UTIL_DIR="${TANNGUYEN2025_BA_DIR}/model/utils"
WRAPPER_LOG="${TANNGUYEN2025_BA_DIR}/replication/output/${datetime}/replication_wrapper.log"
mkdir -p ${MCIPROGRESSION_DIR}

################################################################################
# Submit job to train direct model
################################################################################
${MCIPROGRESSION_DIR}/CBIG_BA_direct.sh -d "train" -s "$split_list" -z "448"
echo "Submitted job to train direct model" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job for direct_ad feature extraction (pre-logistic regression)
################################################################################
for size in $size_list; do
    ${MCIPROGRESSION_DIR}/CBIG_BA_direct_ad.sh -s "$split_list" -z "$size" -f 0
done
echo "Submitted job for direct_ad feature extraction" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job for brainage64D_finetune_ad feature extraction (pre-logistic regression)
################################################################################
for size in $size_list; do
    ${MCIPROGRESSION_DIR}/CBIG_BA_brainage64D_finetune_ad.sh -s "$split_list" -z "$size" -f 0
done
echo "Submitted job for brainage64D_finetune_ad feature extraction" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if direct_ad feature extraction complete
################################################################################
echo "Waiting to complete feature extraction for direct_ad" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/direct_ad" "direct_ad_pre_logreg" "$split_list" "$size_list"
echo "Completed feature extraction for direct_ad!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if brainage64D_finetune_ad feature extraction complete
################################################################################
echo "Waiting to complete feature extraction for brainage64D_finetune_ad" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/brainage64D_finetune_ad" \
"brainage64D_finetune_ad_pre_logreg" "$split_list" "$size_list"
echo "Completed feature extraction for brainage64D_finetune_ad!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job for logistic regression for direct_ad model
################################################################################
for size in $size_list; do
    ${MCIPROGRESSION_DIR}/CBIG_BA_direct_ad.sh -s "$split_list" -z "$size" -f 1
done
echo "Submitted job for logistic regression for direct_ad" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job for logistic regression for brainage64D_finetune_ad model
################################################################################
for size in $size_list; do
    ${MCIPROGRESSION_DIR}/CBIG_BA_brainage64D_finetune_ad.sh -s "$split_list" -z "$size" -f 1
done
echo "Submitted job for logistic regression for brainage64D_finetune_ad" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if logistic regression for direct_ad complete
################################################################################
echo "Waiting to complete logistic regression for direct_ad" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/direct_ad" "direct_ad_LR" "$split_list" "$size_list"
echo "Completed logistic regression for direct_ad!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if logistic regression for brainage64D_finetune_ad complete
################################################################################
echo "Waiting to complete logistic regression for brainage64D_finetune_ad" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/brainage64D_finetune_ad" \
"brainage64D_finetune_ad_LR" "$split_list" "$size_list"
echo "Completed logistic regression for brainage64D_finetune_ad!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if direct model training complete
################################################################################
echo "Waiting to complete training for direct model" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/direct" "train" "$split_list" "448"
echo "Completed training for direct model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to test direct model
################################################################################
${MCIPROGRESSION_DIR}/CBIG_BA_direct.sh -d "test" -s "$split_list" -z "448"
echo "Submitted job to test direct model" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if direct model testing complete
################################################################################
echo "Waiting to complete testing for direct model" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/direct" "test" "$split_list" "448"
echo "Completed testing for direct model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to choose best model for direct
################################################################################
${WRAPPER_DIR}/CBIG_BA_choose_best_model.sh \
-n "${num_split}" -z "448" -m "direct" -t "mci_progression"
echo "Submitted job to choose best model for direct" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if best model selection for direct complete
################################################################################
check_completed_step "${REPLICATION_DIR}/direct" "best_model" "$split_list" "448"
echo "Completed choose best model for direct!" | tee -a ${WRAPPER_LOG}

################################################################################
# Double-check all steps completed
################################################################################
if grep -q "Completed logistic regression for direct_ad!" "${WRAPPER_LOG}" \
    && grep -q "Completed logistic regression for brainage64D_finetune_ad!" "${WRAPPER_LOG}" \
    && grep -q "Completed testing for direct model!" "${WRAPPER_LOG}" \
    && grep -q "Completed choose best model for direct!" "${WRAPPER_LOG}"; then
    echo -e "\nMCI progression completed successfully!!" | tee -a ${WRAPPER_LOG}
fi
