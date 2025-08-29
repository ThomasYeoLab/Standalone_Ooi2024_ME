#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

# setup
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
REPDATADIR=$CBIG_REPDATA_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"
cd $ROOTDIR

# create folders
mkdir -p $ROOTDIR"/replication/raw_data"

# copy raw data to replication folder
rsync -r $REPDATADIR"/data/raw_data" $ROOTDIR"/replication"