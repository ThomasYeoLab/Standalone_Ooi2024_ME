#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

## general setup
node_name=$(hostname)
if [ $node_name != 'headnode' ]; then
    echo "Error: All replication jobs should be submitted via headnode!"
    exit 1
fi
ROOTDIR=$CBIG_CODE_DIR"/stable_projects/predict_phenotypes/Zhang2025_L2CFNN"

# --- Set the Environment Variable ---
# Ensure the variable name matches what Python config code expects (e.g., CBIG_L2C_ROOT_PATH)
export CBIG_L2C_ROOT_PATH=$ROOTDIR"/replication"

echo "Running script using environment variable CBIG_L2C_ROOT_PATH=$CBIG_L2C_ROOT_PATH"
# Optional: Create the directory if it doesn't exist
mkdir -p "$CBIG_L2C_ROOT_PATH"

job_prefix="L2C"
cd $ROOTDIR


# replication starts
echo ">>> Begin to replicate Zhang2025_L2CFNN... Starting with data generation"
log_dir=$CBIG_L2C_ROOT_PATH'/job_logs/data_gen'
mkdir -p $log_dir

################################### step 0 ####################################
echo ">>> Step0: Copy raw data for replication..."
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_data_gen_step0.sh"
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" $job_prefix 5s
################################### step 0 ####################################

################################### step 1 ####################################
echo ">>> Step1: Data generation step 1: Split generation..."
s1_name="${job_prefix}_Step1"
s1_cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_data_gen_step1.sh"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$s1_cmd" \
    -name $s1_name \
    -walltime 00:05:00 \
    -mem 2G \
    -ncpus 1 \
    -jobout $log_dir"/$s1_name.out" \
    -joberr $log_dir"/$s1_name.err"
sleep 5s
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" $s1_name 10s
################################### step 1 ####################################

################################### step 2 ####################################
echo ">>> Step2: Data generation step 2: Baseline value extraction..."
s2_name="${job_prefix}_Step2"
s2_cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_data_gen_step2.sh"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$s2_cmd" \
    -name $s2_name \
    -walltime 00:05:00 \
    -mem 2G \
    -ncpus 1 \
    -jobout $log_dir"/$s2_name.out" \
    -joberr $log_dir"/$s2_name.err"
sleep 5s
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" $s2_name 10s
################################### step 2 ####################################

################################### step 3 ####################################
echo ">>> Step3: Data generation step 3: Fold generation..."
s3_name="${job_prefix}_Step3"
s3_cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_data_gen_step3.sh"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$s3_cmd" \
    -name $s3_name \
    -walltime 00:10:00 \
    -mem 4G \
    -ncpus 2 \
    -jobout $log_dir"/$s3_name.out" \
    -joberr $log_dir"/$s3_name.err"
sleep 5s
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" $s3_name 10s
################################### step 3 ####################################

################################### step 4-1 ####################################
echo ">>> Step4: Data generation step 4-1: L2C transformation for training ..."
s4_1_name="${job_prefix}_Step4_1"
for fold in $(seq 0 19); do
    s4_1_cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_data_gen_step4.sh train $fold"
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$s4_1_cmd" \
        -name $s4_1_name \
        -walltime 02:00:00 \
        -mem 4G \
        -ncpus 1 \
        -jobout $log_dir"/$s4_1_name.out" \
        -joberr $log_dir"/$s4_1_name.err"
    sleep 2s
done
################################### step 4-1 ####################################

################################### step 4-2 ####################################
echo ">>> Step4: Data generation step 4-2: L2C transformation for test ..."
s4_2_name="${job_prefix}_Step4_2"
s4_2_cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_data_gen_step4.sh test"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$s4_2_cmd" \
    -name $s4_2_name \
    -walltime 02:00:00 \
    -mem 4G \
    -ncpus 2 \
    -jobout $log_dir"/$s4_2_name.out" \
    -joberr $log_dir"/$s4_2_name.err"
sleep 5s
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" "${job_prefix}_Step4" 10m
################################### step 4-2 ####################################

################################### step 5 ####################################
echo ">>> Step5: Data generation step 5: Pre-process for L2CFNN..."
s5_name="${job_prefix}_Step5"
for fold in $(seq 0 19); do
    s5_cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_data_gen_step5.sh $fold"
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$s5_cmd" \
        -name $s5_name \
        -walltime 00:05:00 \
        -mem 4G \
        -ncpus 4 \
        -jobout $log_dir"/$s5_name.out" \
        -joberr $log_dir"/$s5_name.err"
    sleep 2s
done
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" $s5_name 1m
################################### step 5 ####################################

################################### step 6 ####################################
echo ">>> Step6: Data generation step 6: Ground truth filtering..."
s6_name="${job_prefix}_Step6"
s6_cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_data_gen_step6.sh"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$s6_cmd" \
    -name $s6_name \
    -walltime 00:05:00 \
    -mem 4G \
    -ncpus 2 \
    -jobout $log_dir"/$s6_name.out" \
    -joberr $log_dir"/$s6_name.err"
sleep 2s
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" $s6_name 5s
################################### step 6 ####################################

################################### step 7 ####################################
echo ">>> Step7: Data generation step 7: Pre-process for MinimalRNN"
s7_name="${job_prefix}_Step7"
s7_cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_data_gen_step7.sh"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$s7_cmd" \
    -name $s7_name \
    -walltime 00:30:00 \
    -mem 4G \
    -ncpus 2 \
    -jobout $log_dir"/$s7_name.out" \
    -joberr $log_dir"/$s7_name.err"
sleep 2s
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" $s7_name 5s
################################### step 7 ####################################

echo ">>> End of data generation part"

echo ">>> Replicate Zhang2025_L2CFNN... Starting with model training and evaluation"

################################### step 8 ####################################
echo ">>> Step8: L2C-FNN train & evaluation..."
log_dir="${CBIG_L2C_ROOT_PATH}/job_logs/models/L2C_FNN"
mkdir -p $log_dir
starting_folds=(0 4 8 12 16)
walltime=12:00:00
ncpus=4
memory=32G
ngpus=1
# choose from RTX3090 gpuservers
node=1

for id in {0..4}; do
    fold=${starting_folds[id]}
    job_name="${job_prefix}_FNN_${fold}"
    cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_train_eval_fnn.sh $fold"
    job_log=$log_dir"/Fold_$fold.out"
    job_err=$log_dir"/Fold_$fold.err"
    echo "$cmd" | qsub -V -q gpuQ -l walltime=${walltime} -l \
        select=1:ncpus=${ncpus}:mem=${memory}:ngpus=${ngpus}:host=gpuserver${node} \
        -m ae -N ${job_name} -e ${job_err} -o ${job_log}
    sleep 2s
done
################################### step 8 ####################################

################################### step 9 ####################################
echo ">>> Step9: L2C-XGBnw train & evaluation..."
log_dir="${CBIG_L2C_ROOT_PATH}/job_logs/models/L2C_XGBnw"
mkdir -p $log_dir
job_name="${job_prefix}_XGBnw"

for fold in $(seq 0 19); do
    cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_train_eval_xgbnw.sh $fold"
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" \
        -name $job_name \
        -walltime 12:00:00 \
        -mem 16G \
        -ncpus 4 \
        -joberr $log_dir"/$job_name.err" \
        -jobout $log_dir"/$job_name.out"
    sleep 2s
done
################################### step 9 ####################################

################################### step 10 ####################################
echo ">>> Step10: L2C-XGBw train & evaluation..."
log_dir="${CBIG_L2C_ROOT_PATH}/job_logs/models/L2C_XGBw"
mkdir -p $log_dir
job_name="${job_prefix}_XGBw"

for fold in $(seq 0 19); do
    cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_train_eval_xgbw.sh $fold"
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" \
        -name $job_name \
        -walltime 12:00:00 \
        -mem 32G \
        -ncpus 4 \
        -joberr $log_dir"/$job_name.err" \
        -jobout $log_dir"/$job_name.out"
    sleep 2s
done
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" "${job_prefix}" 10m
################################### step 10 ####################################

################################### step 11 ####################################
echo ">>> Step11: AD-Map train & evaluation..."
log_dir="${CBIG_L2C_ROOT_PATH}/job_logs/models/AD_Map"
mkdir -p $log_dir
job_name="${job_prefix}_Map"

for fold in $(seq 0 19); do
    cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_train_eval_admap.sh $fold"
    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" \
        -name $job_name \
        -walltime 06:00:00 \
        -mem 4G \
        -ncpus 4 \
        -joberr $log_dir"/$job_name.err" \
        -jobout $log_dir"/$job_name.out"
    sleep 2s
done
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" "${job_name}" 30m
################################### step 11 ####################################

################################### step 12 ####################################
echo ">>> Step12: MinimalRNN train & evaluation..."
log_dir="${CBIG_L2C_ROOT_PATH}/job_logs/models/MinimalRNN"
mkdir -p $log_dir
starting_folds=(0 4 8 12 16)
walltime=12:00:00
ncpus=4
memory=16G
ngpus=1
# choose from RTX3090 gpuservers
node=1

for id in {0..4}; do
    fold=${starting_folds[id]}
    job_name="${job_prefix}_RNN_${fold}"
    cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_train_eval_minrnn.sh $fold"
    job_log=$log_dir"/Fold_$fold.out"
    job_err=$log_dir"/Fold_$fold.err"
    echo "$cmd" | qsub -V -q gpuQ -l walltime=${walltime} -l \
        select=1:ncpus=${ncpus}:mem=${memory}:ngpus=${ngpus}:host=gpuserver${node} \
        -m ae -N ${job_name} -e ${job_err} -o ${job_log}
    sleep 2s
done
bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" "${job_prefix}_RNN" 30m
################################### step 12 ####################################

################################### step 13 ####################################
echo ">>> Step13: Check results against reference..."
log_dir="${CBIG_L2C_ROOT_PATH}/job_logs/utils"
mkdir -p $log_dir
job_name="check"

cmd="bash $ROOTDIR/replication/scripts/CBIG_L2CFNN_check_results.sh"
$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" \
    -name $job_name \
    -walltime 00:05:00 \
    -mem 4G \
    -ncpus 1 \
    -joberr $log_dir"/$job_name.err" \
    -jobout $log_dir"/$job_name.out"
sleep 2s

bash $ROOTDIR"/replication/scripts/CBIG_L2CFNN_wait4jobs2finish.sh" "${job_name}" 10s
################################### step 13 ####################################