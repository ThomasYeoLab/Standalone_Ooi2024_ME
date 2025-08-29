#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This task-level wrapper script will run each model-level wrapper script,
# to fully replicate all 5 AD classification models:
# 1. Direct
# 2. Brain Age Gap (BAG)
# 3. BAG-finetune
# 4. Brainage64D
# 5. Brainage64D-finetune
#
# If user aims to fully replicate AD classification work,
# this script requires no arguments except for the date & time
# in string format for example "250423_1302" to represent
# 23rd April 2025, 1:02 p.m.
#
# Example:
# ssh headnode; ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3a_ad_classification.sh -d "250423_1302";

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
    size_list="50 100 200 300 400 500 600 700 800 900 997"
fi

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

# Set convenience variables
num_split=$(echo "$split_list" | grep -oE '[0-9]+' | wc -l)
WRAPPER_DIR="${TANNGUYEN2025_BA_DIR}/replication/scripts"
ADCLASSIFICATION_DIR="${WRAPPER_DIR}/ad_classification"
REPLICATION_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/ad_classification"
UTIL_DIR="${TANNGUYEN2025_BA_DIR}/model/utils"
WRAPPER_LOG="${TANNGUYEN2025_BA_DIR}/replication/output/${datetime}/replication_wrapper.log"
mkdir -p "$(dirname "$WRAPPER_LOG")"

################################################################################
# Submit job to train direct model
################################################################################
for size in $size_list; do
    ${ADCLASSIFICATION_DIR}/CBIG_BA_direct.sh -d "train" -s "$split_list" -z "$size"
done
echo "Submitted job to train direct model" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to extract features for brainage64D model   
################################################################################
${ADCLASSIFICATION_DIR}/CBIG_BA_brainage64D.sh -s "$split_list" -f "extract_features"
echo "Submitted job to extract features for brainage64D" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to extract features for BAG model
################################################################################
${ADCLASSIFICATION_DIR}/CBIG_BA_BAG.sh -f "extract_features"
echo "Submitted job to extract features for BAG model" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to transfer weights for BAG_finetune model
################################################################################
${ADCLASSIFICATION_DIR}/CBIG_BA_BAG_finetune.sh -s "$split_list" -z "997" -f "transfer_weights"
echo "Submitted job to transfer weights for BAG_finetune model" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if brainage64D feature extraction complete
################################################################################
echo "Waiting to complete extract features for brainage64D" | tee -a ${WRAPPER_LOG}
while [ ! -e "${REPLICATION_DIR}/brainage64D/feature_extract/complete_brainage64D_EF" ]; do
    sleep 30
done
echo "Completed extract features for brainage64D!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if BAG feature extraction is complete
################################################################################
echo "Waiting to complete extract features for BAG model" | tee -a ${WRAPPER_LOG}
while [ ! -e "${REPLICATION_DIR}/BAG/feature_extract/complete_BAG_EF" ]; do
    sleep 30
done
echo "Completed extract features for BAG model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if BAG_finetune weight transfer is complete
################################################################################
echo "Waiting to complete weight transfer for BAG_finetune model" | tee -a ${WRAPPER_LOG}
while [ ! -e "${REPLICATION_DIR}/BAG_finetune/transfer_weights/initialized_sfcn/complete_BAG_finetune_TW" ]; do
    sleep 30
done
echo "Completed weight transfer for BAG_finetune model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to transfer weights for brainage64D model
################################################################################
for size in $size_list; do
    ${ADCLASSIFICATION_DIR}/CBIG_BA_brainage64D.sh -s "$split_list" -z "$size" -f "transfer_weights"
done
echo "Submitted job to transfer weights for brainage64D" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to prepare BAG model for classification
################################################################################
${ADCLASSIFICATION_DIR}/CBIG_BA_BAG.sh -f "prepare2classify"
echo "Submitted job to prepare BAG model for classification" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to finetune (on brain age prediction) BAG_finetune model
################################################################################
${ADCLASSIFICATION_DIR}/CBIG_BA_BAG_finetune.sh -s "$split_list" -z "997" -f "finetune"
echo "Submitted job to finetune BAG_finetune model" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if brainage64D weight transfer complete
################################################################################
echo "Waiting to complete transfer weights for brainage64D" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/brainage64D/log_reg" "brainage64D_TW" "$split_list" "$size_list"
echo "Completed transfer weights for brainage64D!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if BAG prepare2classify step is complete
################################################################################
echo "Waiting to complete prepare2classify step for BAG model" | tee -a ${WRAPPER_LOG}
while [ ! -e "${REPLICATION_DIR}/BAG/prepare2classify/complete_BAG_prepare2classify" ]; do
    sleep 30
done
echo "Completed prepare2classify step for BAG model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to run logistic regression for brainage64D model
################################################################################
for size in $size_list; do
    ${ADCLASSIFICATION_DIR}/CBIG_BA_brainage64D.sh -s "$split_list" -z "$size" -f "logistic_regression"
done
echo "Submitted job to run logistic regression for brainage64D" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to classify using BAG model
################################################################################
${ADCLASSIFICATION_DIR}/CBIG_BA_BAG.sh -f "BAG_classifier"
echo "Submitted job to classify using BAG model" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if BAG classification is complete
################################################################################
echo "Waiting to complete classification using BAG model" | tee -a ${WRAPPER_LOG}
while [ ! -e "${REPLICATION_DIR}/BAG/BAG_classifier/complete_BAG_classifier" ]; do
    sleep 30
done
echo "Completed classification using BAG model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if logistic regression for brainage64D model complete
################################################################################
echo "Waiting to complete logistic regression for brainage64D" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/brainage64D/log_reg" "brainage64D_LR" "$split_list" "$size_list"
echo "Completed logistic regression for brainage64D!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to train brainage64D_finetune model
################################################################################
for size in $size_list; do
    ${ADCLASSIFICATION_DIR}/CBIG_BA_brainage64D_finetune.sh -d "train" -s "$split_list" -z "$size"
done
echo -e "\nSubmitted job to train brainage64D_finetune model" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if direct model training complete
################################################################################
echo "Waiting to complete training for direct model" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/direct" "train" "$split_list" "$size_list"
echo "Completed training for direct model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to test direct model
################################################################################
for size in $size_list; do
    ${ADCLASSIFICATION_DIR}/CBIG_BA_direct.sh -d "test" -s "$split_list" -z "$size"
done
echo "Submitted job to test direct model" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if BAG_finetune finetuning is complete
################################################################################
echo "Waiting to complete finetuning for BAG_finetune model" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/BAG_finetune" "train" "$split_list" "997"
echo "Completed finetuning for BAG_finetune model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to choose best BAG_finetune model
################################################################################
${WRAPPER_DIR}/CBIG_BA_choose_best_model.sh -n "${num_split}" -z "997" -m "BAG_finetune" -t "ad_classification" -s "n"
echo "Submitted job to choose best model for BAG_finetune" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if best BAG_finetune model selection complete
################################################################################
check_completed_step "${REPLICATION_DIR}/BAG_finetune" "best_model" "$split_list" "997"
echo "Completed choose best model for BAG_finetune!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to extract features for BAG_finetune model
################################################################################
${ADCLASSIFICATION_DIR}/CBIG_BA_BAG_finetune.sh -s "$split_list" -z "997" -f "extract_features"
echo "Submitted job to extract features for BAG_finetune model" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if BAG_finetune feature extraction is complete
################################################################################
echo "Waiting to complete extract features for BAG_finetune model" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/BAG_finetune/feature_extract" "BAG_finetune_bag" "$split_list" "997"
echo "Completed extract features for BAG_finetune model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to classify using BAG_finetune model
################################################################################
${ADCLASSIFICATION_DIR}/CBIG_BA_BAG_finetune.sh -f "BAG_classifier"
echo "Submitted job to classify using BAG_finetune model" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if BAG_finetune classification is complete
################################################################################
echo "Waiting to complete classification using BAG_finetune model" | tee -a ${WRAPPER_LOG}
while [ ! -e "${REPLICATION_DIR}/BAG_finetune/BAG_classifier/complete_BAG_classifier" ]; do
    sleep 30
done
echo "Completed classification using BAG_finetune model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if brainage64D_finetune training complete
################################################################################
echo "Waiting to complete training for brainage64D_finetune" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/brainage64D_finetune" "train" "$split_list" "$size_list"
echo "Completed training for brainage64D_finetune!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to test brainage64D_finetune model
################################################################################
for size in $size_list; do
    ${ADCLASSIFICATION_DIR}/CBIG_BA_brainage64D_finetune.sh -d "test" -s "$split_list" -z "$size"
done
echo "Submitted job to test brainage64D_finetune" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if brainage64D_finetune testing complete
################################################################################
echo "Waiting to complete testing for brainage64D_finetune" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/brainage64D_finetune" "test" "$split_list" "$size_list"
echo "Completed testing for brainage64D_finetune!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if direct model testing complete
################################################################################
echo "Waiting to complete testing for direct model" | tee -a ${WRAPPER_LOG}
check_completed_step "${REPLICATION_DIR}/direct" "test" "$split_list" "$size_list"
echo "Completed testing for direct model!" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to choose best direct model
################################################################################
for size in $size_list; do
    ${WRAPPER_DIR}/CBIG_BA_choose_best_model.sh \
    -n "${num_split}" -z "${size}" -m "direct" -t "ad_classification"
done
echo -e "\nSubmitted job to choose best model for direct" | tee -a ${WRAPPER_LOG}

################################################################################
# Submit job to choose best brainage64D_finetune model
################################################################################
for size in $size_list; do
    ${WRAPPER_DIR}/CBIG_BA_choose_best_model.sh \
    -n "${num_split}" -z "${size}" -m "brainage64D_finetune" -t "ad_classification"
done
echo "Submitted job to choose best model for brainage64D_finetune" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if best direct model selection complete
################################################################################
check_completed_step "${REPLICATION_DIR}/direct" "best_model" "$split_list" "$size_list"
echo "Completed choose best model for direct!" | tee -a ${WRAPPER_LOG}

################################################################################
# Check if best brainage64D_finetune model selection complete
################################################################################
check_completed_step "${REPLICATION_DIR}/brainage64D_finetune" "best_model" "$split_list" "$size_list"
echo "Completed choose best model for brainage64D_finetune!" | tee -a ${WRAPPER_LOG}

################################################################################
# Double-check all steps completed
################################################################################
if grep -q "Completed extract features for brainage64D!" "${WRAPPER_LOG}" \
    && grep -q "Completed extract features for BAG model!" "${WRAPPER_LOG}" \
    && grep -q "Completed weight transfer for BAG_finetune model!" "${WRAPPER_LOG}" \
    && grep -q "Completed transfer weights for brainage64D!" "${WRAPPER_LOG}" \
    && grep -q "Completed prepare2classify step for BAG model!" "${WRAPPER_LOG}" \
    && grep -q "Completed classification using BAG model!" "${WRAPPER_LOG}" \
    && grep -q "Completed logistic regression for brainage64D!" "${WRAPPER_LOG}" \
    && grep -q "Completed training for direct model!" "${WRAPPER_LOG}" \
    && grep -q "Completed testing for direct model!" "${WRAPPER_LOG}" \
    && grep -q "Completed finetuning for BAG_finetune model!" "${WRAPPER_LOG}" \
    && grep -q "Completed extract features for BAG_finetune model!" "${WRAPPER_LOG}" \
    && grep -q "Completed classification using BAG_finetune model!" "${WRAPPER_LOG}" \
    && grep -q "Completed training for brainage64D_finetune!" "${WRAPPER_LOG}" \
    && grep -q "Completed testing for brainage64D_finetune!" "${WRAPPER_LOG}" \
    && grep -q "Completed choose best model for direct!" "${WRAPPER_LOG}" \
    && grep -q "Completed choose best model for brainage64D_finetune!" "${WRAPPER_LOG}"; then
    echo -e "\nAD classification completed successfully!!" | tee -a ${WRAPPER_LOG}
fi
