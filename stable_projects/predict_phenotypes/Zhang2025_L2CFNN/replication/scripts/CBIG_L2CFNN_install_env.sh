#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Navigate to the project directory
cd "$CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"

# Step 2: Create the conda environment from the YAML file
conda env create -f replication/config/CBIG_L2CFNN_ADMap_env.yml

# Step 3: Activate the new conda environment
# Note: Activating conda environments in scripts can require special handling
# if run in a non-interactive shell
eval "$(conda shell.bash hook)"
conda activate CBIG_Zhang2025_ADMap

# Step 4: Uninstall pip-installed numpy to avoid conflicts
pip uninstall -y numpy

# Step 5: Install numpy-base and other dependencies using conda
conda install -y numpy-base=1.24.3=py39h31eccc5_0 "blas=1.0=mkl" pandas=1.5.3 zipp=3.20.2 scipy=1.9.3 -c defaults

# Step 6: Verify numpy installation
env | grep -i python

conda list | grep numpy

# Step 7: Install the specific numpy version with MKL
conda install -y numpy=1.24.3 "blas=1.0=mkl" pandas=1.5.3 zipp=3.20.2 scipy=1.9.3 -c defaults