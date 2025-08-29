# Examples of Cross-dataset Evaluation of Dementia Longitudinal Progression Prediction Models


## References

- Zhang, C., An, L., Wulan, N., Nguyen, K. N., Orban, C., Chen, P., Chen, C., Zhou, J. H., Liu, K., Yeo, B. T. T., 2024. [Cross-dataset Evaluation of Dementia Longitudinal Progression Prediction Models](https://doi.org/10.1101/2024.11.18.24317513), medRxiv

----

## Usage

This example contains 9 steps: \
    0. Generate toy data
    1. Pre-process the toy data to desired format for models \
    2. Train and evaluate L2C-FNN model \
    3. Train and evaluate L2C-XGBw model \
    4. Train and evaluate L2C-XGBnw model \
    5. Train and evaluate MinimalRNN model \
    6. Train and evaluate AD-Map model \
    7. Compare results against reference \
    8. Cleanup generated temporary files

Run the following command in terminal 
```
cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN

bash examples/CBIG_L2CFNN_example.sh
```

You are expected to get the same results as following:

```
Starting comparison...
Loading results for Model: L2C_FNN, Site: ADNI...
  [L2C_FNN/ADNI/mAUC] (Mean: 0.423431)
  [L2C_FNN/ADNI/mmseMAE] (Mean: 1.366667)
  [L2C_FNN/ADNI/ventsMAE] (Mean: 0.013975)
Loading results for Model: L2C_XGBw, Site: ADNI...
  [L2C_XGBw/ADNI/mAUC] (Mean: 0.792478)
  [L2C_XGBw/ADNI/mmseMAE] (Mean: 2.910372)
  [L2C_XGBw/ADNI/ventsMAE] (Mean: 0.049152)
Loading results for Model: L2C_XGBnw, Site: ADNI...
  [L2C_XGBnw/ADNI/mAUC] (Mean: 0.735931)
  [L2C_XGBnw/ADNI/mmseMAE] (Mean: 2.805356)
  [L2C_XGBnw/ADNI/ventsMAE] (Mean: 0.049987)
Loading results for Model: AD_Map, Site: ADNI...
  [AD_Map/ADNI/mAUC] (Mean: 0.487013)
  [AD_Map/ADNI/mmseMAE] (Mean: 1.076469)
  [AD_Map/ADNI/ventsMAE] (Mean: 0.001564)
Loading results for Model: MinimalRNN, Site: ADNI...
  [MinimalRNN/ADNI/mAUC] (Mean: 0.470779)
  [MinimalRNN/ADNI/mmseMAE] (Mean: 1.927430)
  [MinimalRNN/ADNI/ventsMAE] (Mean: 0.011746)

============================== Summary ==============================
Processed 5 Model/Site combinations.
Performed 15 comparisons.
All comparisons passed successfully!
```
Please note, this is just a generated data, please refer to the 
replication folder and the paper for the performance on real dataset.

----

### Clean up

The script will automatically delete all temporary files generated during the example run.

## Additional notes

For more detailed example of AD_Map, please refer to the original paper [Forecasting individual progression trajectories in Alzheimer’sdisease](https://doi.org/10.1038/s41467-022-35712-5) and the [code implementation](https://gitlab.com/icm-institute/aramislab/leaspy)

----

For more detailed example of MinimalRNN, please refer to the original paper [Predicting Alzheimer’s disease progression using deep recurrent neural networks](https://doi.org/10.1016/j.neuroimage.2020.117203) and the [code implementation](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/predict_phenotypes/Nguyen2020_RNNAD)


## Bugs and Questions
Please contact Chen Zhang at chenzhangsutd@gmail.com, Naren Wulan at wulannarenzhao@gmail.com, and Lijun An at anlijuncn@gmail.com