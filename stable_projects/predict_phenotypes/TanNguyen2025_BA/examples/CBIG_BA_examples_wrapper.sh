#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This examples wrapper script runs the logistic regression model.
#
# Example:
# ${TANNGUYEN2025_BA_DIR}/examples/CBIG_BA_examples_wrapper.sh;

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

# Set convenience variables
EXAMPLES_DIR="${TANNGUYEN2025_BA_DIR}/examples"
EXAMPLES_SCRIPTS_DIR="$EXAMPLES_DIR/scripts"
EXAMPLES_OUTPUT_DIR="$EXAMPLES_DIR/output"
WRAPPER_LOG="$EXAMPLES_DIR/output/examples_wrapper.log"
mkdir -p "$(dirname "$WRAPPER_LOG")"

################################################################################
# Submit job for logistic regression model
################################################################################
echo "Running examples ..." | tee -a ${WRAPPER_LOG}
${EXAMPLES_SCRIPTS_DIR}/CBIG_BA_examples.sh

################################################################################
# Check if logistic regression model complete
################################################################################
while [ ! -e "${EXAMPLES_OUTPUT_DIR}/complete_log_reg_LR" ]; do
    sleep 30
done

################################################################################
# Check results against reference outputs
################################################################################
source CBIG_init_conda; conda activate CBIG_BA;
python ${EXAMPLES_SCRIPTS_DIR}/CBIG_BA_check_examples_result.py | tee -a ${WRAPPER_LOG}
conda deactivate;
