#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
cd $ROOTDIR
export PYTHONPYCACHEPREFIX="${HOME}/.cache/Python"
export PYTHONPATH=$ROOTDIR:$PYTHONPATH 

# --- Set the Environment Variable ---
# Ensure the variable name matches what Python config code expects (e.g., CBIG_L2C_ROOT_PATH)
export CBIG_L2C_ROOT_PATH=$ROOTDIR"/examples"

echo "Running script using environment variable CBIG_L2C_ROOT_PATH=$CBIG_L2C_ROOT_PATH"
# Optional: Create the directory if it doesn't exist
mkdir -p "$CBIG_L2C_ROOT_PATH"

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"  # Works for most setups
else
    echo "Conda is not installed or not in PATH."
    exit 1
fi

conda activate CBIG_Zhang2025_py38
python -m examples.CBIG_L2CFNN_example --step 1
conda deactivate

conda activate CBIG_Zhang2025_ADMap
python -m examples.CBIG_L2CFNN_example --step 2