# Comparison of brain age pretrained models versus models trained from scratch

## References
Tan, T. W. K. *, Nguyen, K.-N. *, Zhang, C., et al. (2024). [Mind the Gap: Does Brain Age Improve Alzheimer Disease Prediction?]. (https://doi.org/10.1101/2024.11.16.623903). _bioRxiv_, 2024.2011.2016.623903.

---

## Background
Large brain age models, trained to predict chronological age from brain MRI, are widely used as markers of general brain health. However, it is unclear whether these models can effectively predict specific outcomes like Alzheimer's Disease (AD) classification or Mild Cognitive Impairment (MCI) progression. In this work, we compare a state-of-the-art pretrained brain age model (Leonardsen et al., 2022) against models trained directly on AD-related outcomes using T1 MRI from ADNI, AIBL, and MACC Harmonization Cohort datasets. We find that direct models consistently match or outperform brain age models, even when data for direct models is over 1,000 times smaller. These results suggest that training directly on target outcomes of interest is more effective for specific clinical tasks, despite the appeal of large-scale pretraining.

---

## Folder structure
* `data_split`: This folder contains Python scripts to split into train-validation-test data.
* `model`: This folder contains Python scripts to run all models.
* `replication`: This folder contains the scripts to replicate all the analyses in this paper.
* `examples`: This folder contains the example demo data, scripts and reference outputs.
* `unit_tests`: This folder uses `examples` demo data as input to check code.

# Code Release
## Download stand-alone repository
Since the whole Github repository is too big, we provide a stand-alone version of only this project and its dependencies. To download this stand-alone repository, visit this link: [https://github.com/ThomasYeoLab/Standalone_TanNguyen2025_BA](https://github.com/ThomasYeoLab/Standalone_TanNguyen2025_BA)

## Download whole repository
Besides this project, if you want to use the code for other stable projects from our lab as well, you need to download the whole repository.

To download the version of the code that was last tested, you can either: 

- visit this link: [https://github.com/ThomasYeoLab/CBIG/releases/tag/v0.36.0-TanNguyen2025_BA](https://github.com/ThomasYeoLab/CBIG/releases/tag/v0.36.0-TanNguyen2025_BA)

or

- run the following command, if you have Git installed

```
git checkout -b TanNguyen2025_BA v0.36.0-TanNguyen2025_BA
```

# Usage
## Setup
1. Check `TANNGUYEN2025_BA_DIR` path in `replication/config/CBIG_BA_tested_config.sh` points to `$CBIG_CODE_DIR/stable_projects/predict_phenotypes/TanNguyen2025_BA` (or the full path pointing to wherever else your `TanNguyen2025_BA` folder may be located).
2. Install Python environments:
    ```
    cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/TanNguyen2025_BA/replication/config
    conda env create -f CBIG_BA_python_env.yml
    conda env create -f CBIG_BA_pyment_python_env.yml
    conda env create -f CBIG_BA_pyment_v1_python_env.yml
    ```
3. Install brain age pretrained model "pyment" Github repository (Leonardsen et al. 2022):
    ```
    cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/TanNguyen2025_BA
    git clone https://github.com/estenhl/pyment-public.git
    cd pyment-public
    git checkout 7eb604d2aa88a76a59fd9a1a18b148450d998d67
    ```
4. Modify `~/.bashrc` file:
    * Remove/comment any line that sources a CBIG config file. For example, you should remove/comment `source $CBIG_CODE_DIR/setup/CBIG_sample_config.sh`.
    * Please add line `source $CBIG_CODE_DIR/stable_projects/predict_phenotypes/TanNguyen2025_BA/replication/config/CBIG_BA_tested_config.sh` (please replace `$CBIG_CODE_DIR` with its full path).
    * To initialize environmental variables properly, please restart terminal (if using terminal), or restart VNC Viewer port (if using VNC Viewer).
5. Check `PYMENT_DIR` path in `$CBIG_CODE_DIR/stable_projects/predict_phenotypes/TanNguyen2025_BA/CBIG_BA_config.py` points to `$CBIG_CODE_DIR/stable_projects/predict_phenotypes/TanNguyen2025_BA/pyment-public` (or the full path pointing to wherever else you cloned the `pyment-public` folder to).

---

## Updates
* Release v0.36.0 (25/07/2025): Initial release for TanNguyen2025_BA project
* Release v0.35.1 (04/07/2025): README note for TanNguyen2025_BA project

---

## Bugs and questions
Please contact Trevor Tan at [trevortanweikiat97@gmail.com](mailto:trevortanweikiat97@gmail.com) and Thomas Yeo at [yeoyeo02@gmail.com](mailto:yeoyeo02@gmail.com).
