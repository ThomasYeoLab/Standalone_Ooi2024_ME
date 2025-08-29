#! /bin/sh
# Last successfully run on 12 March 2024 with git repository version v0.29.8-CBIG2022_DiffProc-updates
# Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# DO NOT CHANGE: This clears old freesurfer variables if they previously exists
if [ -n "$FREESURFER_HOME" ]; then
    $FREESURFER_HOME/bin/clear_fs_env.csh 
fi

# PLEASE CHANGE: Please specify location of CBIG repository
export CBIG_CODE_DIR=$HOME/storage/CBIG_private

# PLEASE CHANGE: define locations for these libraries
export FREESURFER_HOME=/apps/freesurfer/5.3.0
export CBIG_MATLAB_DIR=/apps/matlab/R2018b
export CBIG_SPM_DIR=/apps/spm/spm12
export CBIG_AFNI_DIR=/apps/afni/AFNI_2011_12_21_1014/linux_openmp_64
export CBIG_ANTS_DIR=/apps/ants/ants_v2.2.0/BUILD/bin/
export CBIG_WB_DIR=/apps/HCP/workbench-1.1.1/
export CBIG_FSLDIR=/apps/fsl/5.0.10
# PLEASE CHANGE: define locations for data
export ABCD_data_dir=/mnt/isilon/CSC2/Yeolab/Data/ABCD
export HCP_data_dir=/mnt/isilon/CSC1/Yeolab/Data/HCP
export ABCD_task_dir=/mnt/nas/CSC7/Yeolab/Data/ABCD/task_preproc
# PLEASE CHANGE: define locations for input data for regression algorithms
export ME_rep_dir=a_placeholder_dir #e.g. /home/leon_ooi/storage/optimal_prediction/replication
export ABCD_input_dir=$ME_rep_dir/ABCD/input
export HCP_input_dir=$ME_rep_dir/HCP/input

# DO NOT CHANGE: define locations for unit tests data and replication data
export CBIG_TESTDATA_DIR=/mnt/isilon/CSC1/Yeolab/CodeMaintenance/UnitTestData
export CBIG_REPDATA_DIR=/mnt/isilon/CSC1/Yeolab/CodeMaintenance/ReplicationData

# DO NOT CHANGE: define scheduler location
export CBIG_SCHEDULER_DIR=/opt/pbs/bin

# DO NOT CHANGE: set up your environment with the configurations above
SETUP_PATH=$CBIG_CODE_DIR/setup/CBIG_generic_setup.sh
source $SETUP_PATH

# DO NOT CHANGE: set up temporary directory for MRIread from FS6.0 for CBIG 
# members using the HPC, Other users should comment this out
export TMPDIR=/tmp

# Do NOT CHANGE: set up MATLABPATH so that MATLAB can find startup.m in our repo 
export MATLABPATH=$CBIG_CODE_DIR/setup

# specified the default Python environment.
# Please UNCOMMENT if you follow CBIG's set up for Python environments.
# We use Python version 3.5 as default.
# Please see $CBIG_CODE_DIR/setup/python_env_setup/README.md for more details.
# source activate CBIG_py3