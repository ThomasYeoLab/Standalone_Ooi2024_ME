#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

cd $CBIG_CODE_DIR/../
mkdir -p Standalone_Zhang2025_L2CFNN/
rsync -axD $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN/ Standalone_Zhang2025_L2CFNN/
