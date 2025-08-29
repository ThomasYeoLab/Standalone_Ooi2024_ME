# Replication of Cross-dataset Evaluation of Dementia Longitudinal Progression Prediction Models

## References
+ Zhang, C., An, L., Wulan, N., Nguyen, K. N., Orban, C., Chen, P., Chen, C., Zhou, J. H., Liu, K., Yeo, B. T. T., 2024.
 [Cross-dataset Evaluation of Dementia Longitudinal Progression Prediction Models](https://doi.org/10.1101/2024.11.18.24317513), medRxiv
  
----

## Data
If you do not have access to [ADNI](http://adni.loni.usc.edu/), [AIBL](https://aibl.org.au/), [MACC](http://www.macc.sg/), and [OASIS](https://sites.wustl.edu/oasisbrains/) data, you need to apply for these datasets to replicate the results in [Zhang et al., 2022](https://doi.org/10.1101/2024.11.18.24317513). 

We run FreeSurfer v6.0 `recon-all` to process T1 MRI data for these four datasets.

----

## Replication

### 1. Setup

Please setup the conda environments before running replication code. There are two environments:
- `CBIG_Zhang2025_py38`: The main environment (Python 3.8) for most of the scripts
- `CBIG_Zhang2025_ADMap`: The environment for AD-Map experiments alone (Python 3.9 to satisfy Leaspy package requirement)

```
cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN

conda env create -f replication/config/CBIG_L2CFNN_python_env.yml

bash replication/scripts/CBIG_L2CFNN_install_env.sh
```

In CBIG, we run experiments via PBS (a job scheduler) on HPC; if you want to run the code on local workstations, you need to run *every shell script* under `replciations/scripts` folder instead of running the wrapper script `CBIG_L2CFNN_replication.sh`; if you want to run the code via other job schedulers, you may need to modify the *submission code* in `CBIG_L2CFNN_replication.sh` accordingly.

### 2. Run replication experiments

We provied code for replicating main comparisons in [Zhang et al., 2022](https://doi.org/10.1101/2024.11.18.24317513), namely `within-cohort prediction` (experiments results in section 3.1) and `cross-cohort prediction` (experiments results in section 3.2). In the replication script, we first train each model on ADNI, and evaluate their prediction performance on ADNI test partitions, AIBL, MACC, and OASIS respectively.


The wrapper script `replciations/CBIG_L2CFNN_replication.sh` contains all steps to replicate the above-mentioned results. For CBIG users, you just need to run the following commands.

```
ssh headnode

cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN

bash replication/CBIG_L2CFNN_replication.sh
```

All the prediction and evaluation metrics can be found in `predictions` folder.

### 3. Compare with reference results

The last step of the wrapper script automatically compares the models' performance with reference results, you can find the comparison results in `replication/job_logs/utils/check.out`. You are expected to see something like below:

```
Starting comparison...
Loading results for Model: L2C_FNN, Site: ADNI...
  PASS [L2C_FNN/ADNI/mAUC] (Mean: 0.945276, Std: 0.016684)
  PASS [L2C_FNN/ADNI/mmseMAE] (Mean: 1.821597, Std: 0.200518)
  PASS [L2C_FNN/ADNI/ventsMAE] (Mean: 0.001823, Std: 0.000253)
Loading results for Model: L2C_FNN, Site: AIBL...
  PASS [L2C_FNN/AIBL/mAUC] (Mean: 0.912099, Std: 0.003350)
  PASS [L2C_FNN/AIBL/mmseMAE] (Mean: 1.311502, Std: 0.025695)
  PASS [L2C_FNN/AIBL/ventsMAE] (Mean: 0.002086, Std: 0.000091)
  ......

============================== Summary ==============================
Processed 20 Model/Site combinations.
Performed 60 comparisons.
All comparisons passed successfully!
======================================================================
```

----
### Clean up

You may want to run the following commands to clean up all generated files.

```
rm -rf $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN/replication/checkpoints
rm -rf $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN/replication/data
rm -rf $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN/replication/job_logs
rm -rf $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN/replication/predictions
rm -rf $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN/replication/raw_data
```

----

## Bugs and Questions
Please contact Chen Zhang at chenzhangsutd@gmail.com, Naren Wulan at wulannarenzhao@gmail.com, and 
Lijun An at anlijuncn@gmail.com
