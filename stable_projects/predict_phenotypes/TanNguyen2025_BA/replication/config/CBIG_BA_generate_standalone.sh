#!/bin/sh
# Written by CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

cd ~/storage
rsync -a --exclude .git CBIG/* Standalone_TanNguyen2025_BA
# remove useless stable projects
rm -r Standalone_TanNguyen2025_BA/stable_projects/brain_parcellation
rm -r Standalone_TanNguyen2025_BA/stable_projects/disorder_subtypes
rm -r Standalone_TanNguyen2025_BA/stable_projects/fMRI_dynamics
rm -r Standalone_TanNguyen2025_BA/stable_projects/harmonization
rm -r Standalone_TanNguyen2025_BA/stable_projects/meta-analysis
rm -r Standalone_TanNguyen2025_BA/stable_projects/preprocessing
rm -r Standalone_TanNguyen2025_BA/stable_projects/registration
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/An2022_gcVAE
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/Chen2024_MMM
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/ChenOoi2023_ICCW
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/ChenTam2022_TRBPC
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/He2019_KRDNN
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/He2022_MM
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/Kong2023_GradPar
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/Naren2024_MMT1
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/Nguyen2020_RNNAD
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/Ooi2022_MMP
rm -r Standalone_TanNguyen2025_BA/stable_projects/predict_phenotypes/Ooi2024_ME
