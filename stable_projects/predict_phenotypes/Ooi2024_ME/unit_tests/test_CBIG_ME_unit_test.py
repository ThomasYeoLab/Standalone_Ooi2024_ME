"""
Written by Shaoshi Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import unittest
import os
from unittest.mock import patch
import runpy
import shutil
import pickle
import numpy as np

EXPECTED_ENV_NAME = "Ooi2024_ME"


class TestME(unittest.TestCase):
    def test_conda_env_name(self):
        env_name = os.environ.get("CONDA_DEFAULT_ENV")
        self.assertEqual(
            env_name,
            EXPECTED_ENV_NAME,
            f"Expected conda environment '{EXPECTED_ENV_NAME}', but found '{env_name}'.",
        )

    script_dir = os.path.join(
        os.environ.get("CBIG_CODE_DIR"),
        "stable_projects",
        "predict_phenotypes",
        "Ooi2024_ME",
        "curve_fitting",
    )
    input_dir = os.path.join(
        os.environ.get("CBIG_CODE_DIR"),
        "stable_projects",
        "predict_phenotypes",
        "Ooi2024_ME",
        "examples",
        "example_data",
    )
    output_dir = os.path.join(
        os.environ.get("CBIG_CODE_DIR"),
        "stable_projects",
        "predict_phenotypes",
        "Ooi2024_ME",
        "examples",
        "example_data",
        "HCP",
        "output",
        "full",
        "curve_fit",
    )
    os.makedirs(output_dir, exist_ok=True)

    @patch(
        "sys.argv", [os.path.join(script_dir, "CBIG_ME_fit_all.py"), "HCP", "59", "full", "predacc", input_dir]
    )
    def test_curve_fitting(self):
        runpy.run_path(os.path.join(self.script_dir, "CBIG_ME_fit_all.py"), run_name="__main__")

        reference = os.path.join(
            os.environ.get("CBIG_CODE_DIR"),
            "stable_projects",
            "predict_phenotypes",
            "Ooi2024_ME",
            "examples",
            "ref_results",
            "predacc_behav59_results.sav",
        )
        example = os.path.join(
            os.environ.get("CBIG_CODE_DIR"),
            "stable_projects",
            "predict_phenotypes",
            "Ooi2024_ME",
            "examples",
            "example_data",
            "HCP",
            "output",
            "full",
            "curve_fit",
            "predacc_behav59_results.sav",
        )

        # load reference files
        pickle_path = reference
        pickle_f = open(pickle_path, "rb")
        ref_file = pickle.load(pickle_f)
        pickle_f.close()

        # load example files
        pickle_path = example
        pickle_f = open(pickle_path, "rb")
        example_file = pickle.load(pickle_f)
        pickle_f.close()
        shutil.rmtree(
            os.path.join(
                os.environ.get("CBIG_CODE_DIR"),
                "stable_projects",
                "predict_phenotypes",
                "Ooi2024_ME",
                "examples",
                "example_data",
                "HCP",
                "output",
                "full",
                "curve_fit",
            )
        )

        # test conditions
        r_diff = np.sum(
            np.array(ref_file["w_r_sav"]) - np.array(example_file["w_r_sav"])
        )
        assert r_diff < 1e-6, "Reliability model weights differ!"
        pa_diff = np.sum(
            np.array(ref_file["w_pa_sav"]) - np.array(example_file["w_pa_sav"])
        )
        assert pa_diff < 1e-6, "Accuracy model weights differ!"
        log_diff = np.sum(
            np.array(ref_file["zk_sav"]) - np.array(example_file["zk_sav"])
        )
        assert log_diff < 1e-6, "Log model weights differ!"
        r_cod = np.sum(
            np.array(ref_file["loss_n_r"]) - np.array(example_file["loss_n_r"])
        )
        assert r_cod < 1e-6, "Reliability model COD values differ!"
        pa_cod = np.sum(
            np.array(ref_file["loss_n_pa"]) - np.array(example_file["loss_n_pa"])
        )
        assert pa_cod < 1e-6, "Accuracy model COD values differ!"
        log_cod = np.sum(
            np.array(ref_file["loss_log"]) - np.array(example_file["loss_log"])
        )
        assert log_cod < 1e-6, "Log model COD values differ!"


if __name__ == "__main__":
    unittest.main()
