# Overview
We compared a state-of-the-art pretrained brain age model (Leonardsen et al., 2022) against models trained directly on 1. AD classification and 2. MCI progression prediction — both of which are binary classification tasks. AD classification involves distinguishing between cognitively normal (CN) versus participants diagnosed with Alzheimer's disease (AD). MCI progression prediction involves distinguishing between stable mild cognititive impairment (sMCI) versus progressive MCI (pMCI) participants. Models were compared across varying development set sizes. Each model (per development set size) was trained and evaluated based on 50 random training-validation-test splits.

There were 5 AD classification models (i.e., 1. BAG, 2. BAG-finetune, 3. Brainage64D, 4. Brainage64D-finetune, and 5. Direct). BAG means "brain age gap". Therefore Models 1-4 were based on the pretrained brain age model. In contrast, Direct model is named as such because it was directly trained on the target from scratch. In terms of prediction, BAG used the brain age gap as a feature. BAG-finetune first finetuned the pretrained brain age model on our dataset to predict brain age, then extracted the brain age gap as a feature. Brainage64D extracted the 64-dimensional (64D) intermediate representations from the pretrained brain age model as features to train and evaluate a logistic regression model. Brainage64D-finetune finetuned all layers of the pretrained brain age model to classify AD. BAG and BAG-finetuned both performed poorly on the largest development size of 997 — would probably perform worse on smaller development sizes — thus replication will only be run for this development size for BAG and BAG-finetune models.  Replication will be run for all development sizes for Brainage64D, Brainage64D-finetune and Direct.

There were 3 MCI progression prediction models (i.e., 1. Brainage64D-finetune, 2. Direct-AD, and 3. Direct). Direct is trained from scratch on distinguishing sMCI versus pMCI participants — note that MCI progression Direct should not be confused as being identical model to AD classification Direct. Brainage64D-finetune-AD took Brainage64D-finetune as a pretrained model to extract 64D features as input for training and evaluating a logistic regression model on MCI progression prediction. Direct-AD took Direct as a pretrained model to extract 64D features as input for training and evaluating a logistic regression model on MCI progression prediction. For MCI progression Direct, replication will only be run for the development size of 448. For Brainage64D-finetune-AD and Direct-AD, replication will be run for all development sizes.


* Replication is divided into the following steps:
    * Step 0. Preparing data
    * Step 1. Age & gender balance classes
    * Step 2. Train-validation-test split
    * Step 3A. Run AD classification models
    * Step 3B. Run MCI progression prediction models
    * Step 4. Check reference results
* Step 3A. Run AD classification models (50 random train-val-test splits per model)
    * Development size of only 997 samples
        * BAG
        * BAG-finetune
    * All development sizes of 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, and 997
        * Brainage64D
        * Brainage64D-finetune
        * Direct
* Step 3B. Run MCI progression prediction models (50 random train-val-test splits per model)
    * Development size of only 448 samples
        * Direct
    * All development sizes of 50, 100, 150, 200, 250, 300, 350, 400, and 448
        * Brainage64D-finetune-AD
        * Direct-AD



---

# Data
We used clinical and baseline visit T1 MRI data from 3 datasets: AIBL, ADNI, and MACC Harmonization cohort. Preprocessed baseline visit T1 MRI data for respective datasets are stored in `T1_DIRS` dictionary variable under `${TANNGUYEN2025_BA_DIR}/CBIG_BA_config.py`. These T1 MRI underwent preprocessing consistent with Leonardsen et al., 2022. Clinical data is stored in `REPLICATION_DATA` variable under `${TANNGUYEN2025_BA_DIR}/CBIG_BA_config.py`. Please note that all filenames and directories in this replication **work for CBIG lab only**.

---

# Environment setup
Please follow all the steps under "Setup" section from main README `${TANNGUYEN2025_BA_DIR}/README.md`.

---

# Run

## Option 1: Overall wrapper to run entire replication
```
ssh headnode
bash ${TANNGUYEN2025_BA_DIR}/replication/CBIG_BA_replication_wrapper.sh
```
* This wrapper will automatically check if your outputs are identical to the reference outputs.
* After running this wrapper, your replication is successful and complete if you output:
    * `${TANNGUYEN2025_BA_DIR}/replication/output/4_check_reference_results_ad_classification`
    * `${TANNGUYEN2025_BA_DIR}/replication/output/4_check_reference_results_mci_progression`
* For more details on each step, please refer to "Option 2" section below.

## Option 2: Run individual steps (must run in order)

### Step 0. Preparing data
```
source CBIG_init_conda; conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR}; python -m replication.CBIG_BA_0_prepare_data;
```
Command will:
* Harmonize clinical data from AIBL, ADNI, and MACC datasets.
* Save hyperparameters for Brainage64D-finetune, Direct (AD classification), and Direct (MCI progression) models.

Command output(s):
* `${TANNGUYEN2025_BA_DIR}/data/raw`
* `${TANNGUYEN2025_BA_DIR}/data/params`

### Step 1. Age & gender balance classes
```
source CBIG_init_conda; conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR}; python -m replication.CBIG_BA_1_balance_dx;
```
Command will:
* Perform age and sex matching between binary classes for both AD classification and MCI progression prediction tasks.

Command output(s):
* `${TANNGUYEN2025_BA_DIR}/data/matched`

### Step 2. Train-validation-test split
```
source CBIG_init_conda; conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR}; python -m replication.CBIG_BA_2_data_split;
```
Command will:
* Divide data into 50 random train-validation-test splits for all development set sizes, and for both prediction tasks.

Command output(s):
* `${TANNGUYEN2025_BA_DIR}/data/data_split`

### Step 3A. Run AD classification models
```
ssh headnode; bash ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3a_ad_classification.sh -d "YYMMDD_HHMM";
```
Command will:
* Run all 5 models for AD classification task (i.e., BAG, BAG-finetune, Brainage64D, Brainage64D-finetune, Direct).
    * BAG and BAG-finetune will be run only on the largest development size (i.e., 997)
    * Brainage64D, Brainage64D-finetune, and Direct will be run on all development sizes (i.e., 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 997).

Command output(s):
* `${TANNGUYEN2025_BA_DIR}/replication/output/ad_classification`

### Step 3B. Run MCI progression prediction models
```
ssh headnode; bash ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3b_mci_progression.sh -d "YYMMDD_HHMM";
```
Command will:
* Run all 3 models for MCI progression prediction task (i.e., Brainage64D-finetune, Direct-AD, Direct).
    * Direct will be run only on the largest development size (i.e., 448)
    * Brainage64D-finetune and Direct-AD will be run on all development sizes (i.e., 50, 100, 150, 200, 250, 300, 350, 400, 448).

Command output(s):
* `${TANNGUYEN2025_BA_DIR}/replication/output/mci_progression`

### Step 4. Check reference results
```
source CBIG_init_conda; conda activate CBIG_BA; cd ${TANNGUYEN2025_BA_DIR}; python -m replication.scripts.CBIG_BA_4_check_reference_results;
```
Command will:
* Check if your output results for all 8 models are identical to the reference results.

If you successfully generated these output(s), your outputs are identical to the reference, and your replication is successful:
* `${TANNGUYEN2025_BA_DIR}/replication/output/4_check_reference_results_ad_classification`
* `${TANNGUYEN2025_BA_DIR}/replication/output/4_check_reference_results_mci_progression`

---

# Pretrained Brain Age Weights

`${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_0_prepare_data.py` downloads the following two pretrained weights files for the Pyment SFCN brain age model (regression variant only):
- `regression_sfcn_brain_age_weights.hdf5`
- `regression_sfcn_brain_age_weights_no_top.hdf5`

**No modifications** were made to these original weights.

Original source is [https://github.com/estenhl/pyment-public](https://github.com/estenhl/pyment-public). These weights were from Git commit hash 7eb604d2aa88a76a59fd9a1a18b148450d998d67.

## Attribution

These weights were developed by **Esten H. Leonardsen** and colleagues and released as part of the study:

> Leonardsen, E. H., Peng, H., Kaufmann, T., Agartz, I., Andreassen, O. A., Celius, E. G., Espeseth, T., Harbo, H. F., Hogestol, E. A., Lange, A. M., Marquand, A. F., Vidal-Pineiro, D., Roe, J. M., Selbaek, G., Sorensen, O., Smith, S. M., Westlye, L. T., Wolfers, T., & Wang, Y. (2022). Deep neural networks learn general and clinically relevant representations of the ageing brain. *NeuroImage, 256*, 119210. https://doi.org/10.1016/j.neuroimage.2022.119210

© 2022 Esten H. Leonardsen

## License

These weights were made available under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- License URL: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)