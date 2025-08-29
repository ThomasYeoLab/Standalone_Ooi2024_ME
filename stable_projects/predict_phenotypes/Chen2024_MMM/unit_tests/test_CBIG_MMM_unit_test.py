#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Pansheng Chen and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import sys
import os
import time
import shutil
import unittest
import subprocess
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../cbig/Chen2024'))
from config import config
from CBIG_misc import compare_dicts, get_phe_num

ut_dir = config.UT_DIR
in_dir = config.IN_DIR_UT
out_dir = config.OUT_DIR_UT
inter_dir = config.INTER_DIR_UT
model_dir = config.MODEL_DIR_UT
ref_dir = config.REF_OUT_DIR_UT


def check_job_status():
    cmd = ('ssh headnode " qstat -f | grep -C 1 `whoami` | grep Chen2024_MMM_unit_test | wc -l"')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())


class TestMMM(unittest.TestCase):
    # @classmethod
    # def tearDownClass(cls):
    #     shutil.rmtree(config.MODEL_DIR_UT)
    #     shutil.rmtree(config.INTER_DIR_UT)
    #     shutil.rmtree(config.OUT_DIR_UT)

    def test_DNN(self):
        cmd = (
            "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM;"
            "source CBIG_init_conda;"
            "conda activate CBIG_Chen2024;"
            "export MKL_SERVICE_FORCE_INTEL=1;"
            "python ./cbig/Chen2024/CBIG_dnn_xlarge_train.py --exp-dataset --unit-test --seed 1 --epochs 100 \
               --metric cod --weight_decay 1e-7 --lr 0.01 --dropout 0.2 --n_l1 64 --n_l2 64 --n_l3 64 \
               --n_hidden_layer 2 --batch_size 64 --patience 25")
        submit_cmd = (
            'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
            f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -ngpus 1 -name Chen2024_MMM_unit_test"'
        )
        subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)

        ref = np.load(os.path.join(ref_dir, 'output_intermediate', 'dnn_base.npz'), allow_pickle=True)
        res = np.load(os.path.join(inter_dir, 'dnn_base.npz'), allow_pickle=True)
        self.assertTrue(
            compare_dicts(ref, res),
            'DNN base training metric ' + str(res) + ' vs. ref ' + str(ref))

        cmd = (
            "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
            "source CBIG_init_conda;"
            "conda activate CBIG_Chen2024;"
            "export MKL_SERVICE_FORCE_INTEL=1;"
            "python ./cbig/Chen2024/CBIG_dnn_xlarge_predict.py --exp-dataset --unit-test"
        )
        submit_cmd = (
            'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
            f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -ngpus 1 -name Chen2024_MMM_unit_test"'
        )
        subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)

        ref = np.load(
            os.path.join(ref_dir, 'output_intermediate', 'dnn_prediction.npz'))
        res = np.load(os.path.join(inter_dir, 'dnn_prediction.npz'))
        self.assertTrue(compare_dicts(ref, res), 'DNN prediction are different')

        shutil.copyfile(
            os.path.join(in_dir, 'exp_test_split_ind_3.npz'),
            os.path.join(inter_dir, 'exp_test_split_ind_3.npz'))
        cmd = (
            "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
            "source CBIG_init_conda;"
            "conda activate CBIG_Chen2024;"
            "export MKL_SERVICE_FORCE_INTEL=1;"
            "python ./cbig/Chen2024/CBIG_dnn_transfer_learning.py --exp-dataset --unit-test"
        )
        submit_cmd = (
            'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
            f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -ngpus 1 -name Chen2024_MMM_unit_test"'
        )
        subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)
        ref = np.load(
            os.path.join(ref_dir, 'transfer_learning_2exp_test_result.npz'),
            allow_pickle=True)
        res = np.load(
            os.path.join(out_dir, 'transfer_learning_2exp_test_result.npz'),
            allow_pickle=True)
        self.assertTrue(compare_dicts(ref, res), 'Transfer learning results are different')

    def test_stacking(self):
        os.makedirs(inter_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(ref_dir, 'output_intermediate', 'dnn_prediction.npz'),
            os.path.join(inter_dir, 'dnn_prediction.npz'))
        shutil.copyfile(
            os.path.join(ref_dir, 'output_intermediate', 'rr_prediction.npz'),
            os.path.join(inter_dir, 'rr_prediction.npz'))
        shutil.copyfile(
            os.path.join(in_dir, 'exp_test_split_ind_3.npz'),
            os.path.join(inter_dir, 'exp_test_split_ind_3.npz'))

        n_phe_L = get_phe_num(in_dir, 'exp_train_L')
        n_phe_M = get_phe_num(in_dir, 'exp_train_M')

        for phe in range(n_phe_L):
            cmd = (
                "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
                "source CBIG_init_conda;"
                "conda activate CBIG_Chen2024;"
                "export MKL_SERVICE_FORCE_INTEL=1;"
                f"python ./cbig/Chen2024/CBIG_rr_large.py --exp-dataset --unit-test --phe_idx {phe}"
            )
            submit_cmd = (
                'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
                f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -name Chen2024_MMM_unit_test"'
            )
            subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)

        for phe in range(n_phe_M):
            cmd = (
                "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
                "source CBIG_init_conda;"
                "conda activate CBIG_Chen2024;"
                "export MKL_SERVICE_FORCE_INTEL=1;"
                f"python ./cbig/Chen2024/CBIG_rr_medium.py --exp-dataset --unit-test --phe_idx {phe}"
            )
            submit_cmd = (
                'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
                f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -name Chen2024_MMM_unit_test"'
            )
            subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)

        for phe in range(n_phe_L):
            cmd = (
                "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
                "source CBIG_init_conda;"
                "conda activate CBIG_Chen2024;"
                "export MKL_SERVICE_FORCE_INTEL=1;"
                f"python ./cbig/Chen2024/CBIG_rr_large.py --exp-dataset --unit-test --phe_idx {phe} --2layer"
            )
            submit_cmd = (
                'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
                f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -name Chen2024_MMM_unit_test"'
            )
            subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)

        for phe in range(n_phe_M):
            cmd = (
                "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
                "source CBIG_init_conda;"
                "conda activate CBIG_Chen2024;"
                "export MKL_SERVICE_FORCE_INTEL=1;"
                f"python ./cbig/Chen2024/CBIG_rr_medium.py --exp-dataset --unit-test --phe_idx {phe} --2layer"
            )
            submit_cmd = (
                'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
                f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -name Chen2024_MMM_unit_test"'
            )
            subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)

        cmd = (
            "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
            "source CBIG_init_conda;"
            "conda activate CBIG_Chen2024;"
            "export MKL_SERVICE_FORCE_INTEL=1;"
            "python ./cbig/Chen2024/CBIG_mm_stacking.py --exp-dataset --unit-test --log_stem MM_stacking"
        )
        submit_cmd = (
            'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
            f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -name Chen2024_MMM_unit_test"'
        )
        subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)
        ref = np.load(os.path.join(ref_dir, 'MM_stacking_2exp_test_result.npz'), allow_pickle=True)
        res = np.load(os.path.join(out_dir, 'MM_stacking_2exp_test_result.npz'), allow_pickle=True)
        self.assertTrue(compare_dicts(ref, res), 'Meta-matching with stacking results are different')

        cmd = (
            "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
            "source CBIG_init_conda;"
            "conda activate CBIG_Chen2024;"
            "export MKL_SERVICE_FORCE_INTEL=1;"
            "python ./cbig/Chen2024/CBIG_mm_stacking.py --exp-dataset --unit-test --log_stem dataset_stacking"
        )
        submit_cmd = (
            'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
            f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -name Chen2024_MMM_unit_test"'
        )
        subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)
        ref = np.load(os.path.join(ref_dir, 'dataset_stacking_2exp_test_result.npz'), allow_pickle=True)
        res = np.load(os.path.join(out_dir, 'dataset_stacking_2exp_test_result.npz'), allow_pickle=True)
        self.assertTrue(compare_dicts(ref, res), 'Meta-matching with dataset stacking results are different')

        cmd = (
            "cd $CBIG_CODE_DIR/stable_projects/predict_phenotypes/Chen2024_MMM/;"
            "source CBIG_init_conda;"
            "conda activate CBIG_Chen2024;"
            "export MKL_SERVICE_FORCE_INTEL=1;"
            "python ./cbig/Chen2024/CBIG_mm_stacking.py --exp-dataset --unit-test --log_stem multilayer_stacking"
        )
        submit_cmd = (
            'ssh headnode " $CBIG_CODE_DIR/setup/CBIG_pbsubmit '
            f'-cmd \'{cmd}\' -walltime 1:00:00 -mem 32G -ncpus 4 -name Chen2024_MMM_unit_test"'
        )
        subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
        while check_job_status():
            time.sleep(60)
        ref = np.load(os.path.join(ref_dir, 'multilayer_stacking_2exp_test_result.npz'), allow_pickle=True)
        res = np.load(os.path.join(out_dir, 'multilayer_stacking_2exp_test_result.npz'), allow_pickle=True)
        self.assertTrue(compare_dicts(ref, res), 'Multilayer meta-matching results are different')

    def tearDown(self):
        shutil.rmtree(config.MODEL_DIR_UT)
        shutil.rmtree(config.INTER_DIR_UT)
        shutil.rmtree(config.OUT_DIR_UT)


if __name__ == '__main__':
    unittest.main()
