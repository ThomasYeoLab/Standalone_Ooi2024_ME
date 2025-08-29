# Examples for brain age comparison

## Background
These examples check the training and testing of the logistic regression model.

## Data
Input data is stored under the `input`.
* For all files, each row corresponds to a single participant. Across all files, the same N-th row corresponds to the same participant.
* There is a single file for each train/validation/test split. 
    * `<train/val/test>_64d.npy` contains 64-dimensional features. Dimension = [# participants, 64].
    * `<train/val/test>_dx.txt` contains the diagnosis. Dimension = [# participants, 1]. `0` = cognitively normal, `1` = Alzheimer's disease.
    * `<train/val/test>_ids.txt` contains the participant ID. Dimension = [# participants, 1].

## Run
* Please follow all the steps under "Setup" section from main README `${TANNGUYEN2025_BA_DIR}/README.md`.
* Please run on the terminal:
    ```
    bash ${TANNGUYEN2025_BA_DIR}/examples/CBIG_BA_examples_wrapper.sh
    ```
