#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This highest-level wrapper script will run all steps for replication:
# 1. ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_0_prepare_data.py
# 2. ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_1_balance_dx.py
# 3. ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_2_data_split.py
# 4. ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3a_ad_classification.sh
# 5. ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3b_mci_progression.sh
# 6. ${TANNGUYEN2025_BA_DIR}/replication/scripts/4_check_replication_results.py
#
# Example:
# ssh headnode; ${TANNGUYEN2025_BA_DIR}/replication/CBIG_BA_replication_wrapper.sh;

################################################################################
# Define variables general to all steps
################################################################################
# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

# Create log base directory
DATETIME=$(date +'%y%m%d_%H%M')
LOG_BASE_DIR="${TANNGUYEN2025_BA_DIR}/replication/output/${DATETIME}"
mkdir -p ${LOG_BASE_DIR}
LOG_FILE="${LOG_BASE_DIR}/replication_wrapper.log"

################################################################################
# Run Step 0: preparing data
################################################################################
# Submit
job_name="0_prepare_data"
cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"
cmd+="python -m replication.scripts.CBIG_BA_0_prepare_data; conda deactivate;"
walltime="00:10:00"
mem="2G"
ncpus="1"
LOG_DIR="${LOG_BASE_DIR}/${job_name}"
mkdir -p ${LOG_DIR}
job_out="${LOG_DIR}/${job_name}.out"
job_err="${LOG_DIR}/${job_name}.err"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit \
-cmd "$cmd" -name "$job_name" -walltime "$walltime" -mem "$mem" -ncpus "$ncpus" -joberr "$job_err" -jobout "$job_out"
echo "Submitted 0_prepare_data" | tee -a ${LOG_FILE}

# Check
echo "Waiting to complete 0_prepare_data" | tee -a ${LOG_FILE}
while [ ! -e "${TANNGUYEN2025_BA_DIR}/data/complete_0_prepare_data" ]; do
    sleep 30
done
echo "Completed 0_prepare_data!" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

################################################################################
# Run Step 1: balancing diagnoses
################################################################################
# Submit
job_name="1_balance_dx"
cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"
cmd+="python -m replication.scripts.CBIG_BA_1_balance_dx; conda deactivate;"
walltime="00:10:00"
mem="2G"
ncpus="1"
LOG_DIR="${LOG_BASE_DIR}/${job_name}"
mkdir -p ${LOG_DIR}
job_out="${LOG_DIR}/${job_name}.out"
job_err="${LOG_DIR}/${job_name}.err"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit \
-cmd "$cmd" -name "$job_name" -walltime "$walltime" -mem "$mem" -ncpus "$ncpus" -joberr "$job_err" -jobout "$job_out"
echo "Submitted 1_balance_dx" | tee -a ${LOG_FILE}

# Check
echo "Waiting to complete 1_balance_dx" | tee -a ${LOG_FILE}
while [ ! -e "${TANNGUYEN2025_BA_DIR}/data/complete_1_balance_dx" ]; do
    sleep 30
done
echo "Completed 1_balance_dx!" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

################################################################################
# Run Step 2: dividing data into train-validation-test splits
################################################################################
# Submit
job_name="2_data_split"
cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"
cmd+="python -m replication.scripts.CBIG_BA_2_data_split; conda deactivate;"
walltime="00:10:00"
mem="2G"
ncpus="1"
LOG_DIR="${LOG_BASE_DIR}/${job_name}"
mkdir -p ${LOG_DIR}
job_out="${LOG_DIR}/${job_name}.out"
job_err="${LOG_DIR}/${job_name}.err"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit \
-cmd "$cmd" -name "$job_name" -walltime "$walltime" -mem "$mem" -ncpus "$ncpus" -joberr "$job_err" -jobout "$job_out"
echo "Submitted 2_data_split" | tee -a ${LOG_FILE}

# Check
echo "Waiting to complete 2_data_split" | tee -a ${LOG_FILE}
while [ ! -e "${TANNGUYEN2025_BA_DIR}/data/complete_2_data_split" ]; do
    sleep 30
done
echo "Completed 2_data_split!" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

################################################################################
# Run Step 3A: AD classification
################################################################################
echo "Running 3a_ad_classification" | tee -a ${LOG_FILE}
${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3a_ad_classification.sh -d "$DATETIME"
echo "Completed 3a_ad_classification!" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

################################################################################
# Run Step 3B: MCI progression prediction
################################################################################
echo "Running 3b_mci_progression" | tee -a ${LOG_FILE}
${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3b_mci_progression.sh -d "$DATETIME"
echo "Completed 3b_mci_progression!" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

################################################################################
# Run Step 4: Check if replication results identical to reference results
################################################################################
# Submit
job_name="4_check_reference_results"
cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"
cmd+="python -m replication.scripts.CBIG_BA_4_check_reference_results; conda deactivate;"
walltime="00:10:00"
mem="2G"
ncpus="1"
LOG_DIR="${LOG_BASE_DIR}/${job_name}"
mkdir -p ${LOG_DIR}
job_out="${LOG_DIR}/${job_name}.out"
job_err="${LOG_DIR}/${job_name}.err"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit \
-cmd "$cmd" -name "$job_name" -walltime "$walltime" -mem "$mem" -ncpus "$ncpus" -joberr "$job_err" -jobout "$job_out"
echo "Submitted 4_check_reference_results" | tee -a ${LOG_FILE}

# Check reference results
echo "Waiting to complete 4_check_reference_results" | tee -a ${LOG_FILE}
while [ ! -e "${TANNGUYEN2025_BA_DIR}/replication/output/complete_4_check_reference_results_ad_classification" ] || \
      [ ! -e "${TANNGUYEN2025_BA_DIR}/replication/output/complete_4_check_reference_results_mci_progression" ]; do
    sleep 30
done
echo "Completed 4_check_reference_results!" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
