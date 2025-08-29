# Unit test of Cross-dataset Evaluation of Dementia Longitudinal Progression Prediction Models


## Reference

+ Zhang, C., An, L., Wulan, N., Nguyen, K. N., Orban, C., Chen, P., Chen, C., Zhou, J. H., Liu, K., Yeo, B. T. T., 2024. [Cross-dataset Evaluation of Dementia Longitudinal Progression Prediction Models](https://doi.org/10.1101/2024.11.18.24317513), medRxiv

----

## Run

The unit tests is for **CBIG lab only**.

You could run the following commands to run unit tests

```
cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Zhang2025_L2CFNN

python unit_tests/test_CBIG_L2CFNN_unit_test.py
```

The unittest will return the test results in the same terminal session where you run the script.

If the test failed, you would see FAILED message and the failed sub-tests similar to below:

```
F
======================================================================
FAIL: test_L2C (__main__.TestL2C)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "unit_tests/test_CBIG_L2CFNN_unit_test.py", line 149, in test_L2C
    self.assertTrue(test_passed_flag, test_message)
AssertionError: False is not true : Config using root path: /mnt/isilon/CSC1/Yeolab/Users/czhang/CBIG/CBIG_private/stable_projects/predict_phenotypes/Zhang2025_L2CFNN/unit_tests (Set by CBIG_L2C_ROOT_PATH env var or default)
Loaded reference data from: /mnt/isilon/CSC1/Yeolab/Users/czhang/CBIG/CBIG_private/stable_projects/predict_phenotypes/Zhang2025_L2CFNN/unit_tests/ref_results/ref_results.csv

Starting comparison...
Loading results for Model: L2C_FNN, Site: ADNI...
  [L2C_FNN/ADNI/mmseMAE] (Mean: 1.366667)
Loading results for Model: L2C_XGBw, Site: ADNI...
  [L2C_XGBw/ADNI/ventsMAE] (Mean: 0.049152)
Loading results for Model: L2C_XGBnw, Site: ADNI...
  [L2C_XGBnw/ADNI/mAUC] (Mean: 0.735931)
  [L2C_XGBnw/ADNI/ventsMAE] (Mean: 0.049987)

============================== Summary ==============================
Processed 3 Model/Site combinations.
Performed 9 comparisons.

5 COMPARISON(S) FAILED:
- FAIL [L2C_FNN/ADNI/mAUC]: Mean mismatch (Calc: 0.423431, Ref: 0.413690)
- FAIL [L2C_FNN/ADNI/ventsMAE]: Mean mismatch (Calc: 0.013975, Ref: 0.013908)
- FAIL [L2C_XGBw/ADNI/mAUC]: Mean mismatch (Calc: 0.792478, Ref: 0.750812)
- FAIL [L2C_XGBw/ADNI/mmseMAE]: Mean mismatch (Calc: 2.910372, Ref: 2.888956)
- FAIL [L2C_XGBnw/ADNI/mmseMAE]: Mean mismatch (Calc: 2.805356, Ref: 2.877486)
======================================================================


----------------------------------------------------------------------
Ran 1 test in 175.532s

FAILED (failures=1)
You have new mail in /var/spool/mail/czhang
```

If the test passed, you are expected to get the same result as below:

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

============================== Summary ==============================
Processed 3 Model/Site combinations.
Performed 9 comparisons.
All comparisons passed successfully!
======================================================================
.
----------------------------------------------------------------------
Ran 1 test in 149.991s

OK
```
----

### Clean up

The script will automatically delete all temporary files generated during the unit test.

## Bugs and Questions
Please contact Chen Zhang at chenzhangsutd@gmail.com, Naren Wulan at wulannarenzhao@gmail.com, and Lijun An at anlijuncn@gmail.com