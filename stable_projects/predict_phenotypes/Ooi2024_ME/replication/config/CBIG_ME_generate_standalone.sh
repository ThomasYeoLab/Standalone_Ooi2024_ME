#!/bin/sh
# Written by CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

cd ~/storage
rsync -a --exclude .git CBIG/* Standalone_Ooi2024_ME
# remove useless stable projects
rm -r Standalone_Ooi2024_ME/stable_projects/brain_parcellation/Kong2019_MSHBM
rm -r Standalone_Ooi2024_ME/stable_projects/brain_parcellation/Kong2022_ArealMSHBM
rm -r Standalone_Ooi2024_ME/stable_projects/brain_parcellation/Xue2021_IndCerebellum
rm -r Standalone_Ooi2024_ME/stable_projects/brain_parcellation/Yan2023_homotopic
rm -r Standalone_Ooi2024_ME/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering
rm -r Standalone_Ooi2024_ME/stable_projects/disorder_subtypes
rm -r Standalone_Ooi2024_ME/stable_projects/fMRI_dynamics
rm -r Standalone_Ooi2024_ME/stable_projects/registration
rm -r Standalone_Ooi2024_ME/stable_projects/meta-analysis
rm -r Standalone_Ooi2024_ME/stable_projects/preprocessing
rm -r Standalone_Ooi2024_ME/stable_projects/predict_phenotypes/He2019_KRDNN
rm -r Standalone_Ooi2024_ME/stable_projects/predict_phenotypes/Nguyen2020_RNNAD
rm -r Standalone_Ooi2024_ME/stable_projects/predict_phenotypes/ChenTam2022_TRBPC
rm -r Standalone_Ooi2024_ME/stable_projects/predict_phenotypes/An2022_gcVAE
rm -r Standalone_Ooi2024_ME/stable_projects/predict_phenotypes/Ooi2022_MMP
rm -r Standalone_Ooi2024_ME/stable_projects/predict_phenotypes/He2022_MM
rm -r Standalone_Ooi2024_ME/stable_projects/predict_phenotypes/ChenOoi2023_ICCW
rm -r Standalone_Ooi2024_ME/stable_projects/predict_phenotypes/Kong2023_GradPar