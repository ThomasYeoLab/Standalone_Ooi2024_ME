#!/bin/bash
# Written by Trevor Tan and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
#
# This examples wrapper script runs the logistic regression model type.
# This logistic regression model type was used by Brainage64D,
# Brainage64D-finetune-AD and Direct-AD.
#
# Example:
# ${TANNGUYEN2025_BA_DIR}/examples/scripts/CBIG_BA_examples.sh;

################################################################################
# Defining arguments & convenience variables
################################################################################
# Set convenience variables
model="log_reg"

# Check if environment variable is set
if [[ -z "${TANNGUYEN2025_BA_DIR}" ]]; then
    echo "ERROR: TANNGUYEN2025_BA environment variable not set." >&2
    exit 1
fi

# Set up relevant paths
EXAMPLES_DIR="${TANNGUYEN2025_BA_DIR}/examples"
INPUT_DIR="$EXAMPLES_DIR/input"
OUTPUT_DIR="$EXAMPLES_DIR/output"

################################################################################
# Run logistic regression model
################################################################################
# Construct command
cmd="cd $TANNGUYEN2025_BA_DIR; source CBIG_init_conda; conda activate CBIG_BA;"
cmd+="python -m model.log_reg.CBIG_BA_logistic_regression --input_dir ${INPUT_DIR}"
cmd+=" --output_dir ${OUTPUT_DIR} --model_name ${model} --examples 'y';"
cmd+="conda deactivate;"

# Run command
eval "$cmd"
