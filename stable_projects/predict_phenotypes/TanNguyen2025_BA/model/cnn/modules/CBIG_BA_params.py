"""
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script defines a utility class `ModelParameters` for managing the configuration 
of hyperparameters used in training SFCN/SFCN_FC models. The class allows you to 
store, update, print, and save hyperparameter configurations, along with logging 
the best model performance across epochs.
"""

import json


class ModelParameters():
    """
    Class to store, update, and manage the hyperparameters of an SFCN-based model.

    Attributes:
        params (dict): Dictionary of all hyperparameters and metadata (e.g. timestamp).
    """

    def __init__(self, time_stamp, params_dict):
        """
        Initialize the parameter container with timestamp and dictionary of hyperparameters.

        Args:
            time_stamp (str): Start time of training, typically used for logging or file naming.
            params_dict (dict): Dictionary of training hyperparameters such as batch size, learning rate, etc.

        Raises:
            AssertionError: If any of the required keys are missing from params_dict.
        """
        # Store start time and hyperparameters in a dictionary
        self.params = {'start_time': time_stamp}
        self.params.update(params_dict)

        # Setting attribute to access later
        for k, v in self.params.items():
            setattr(self, k, v)

        # Make sure important attributes are present
        assert all(
            key in self.params for key in [
                "batch_size", "n_epoch", "dropout", "optim", "weight_decay",
                "init_lr", "lr_decay", "lr_step"
            ]
        ), "Missing values, the following values batch_size, n_epoch, dropout, optim, weight_decay, \
            init_lr, lr_decay, lr_step must be passed in."

    def update_params(self, params_dict):
        """
        Update the current hyperparameters with new values.

        Args:
            params_dict (dict): Dictionary of new parameters to merge into the existing configuration.
        """
        for k, v in params_dict.items():
            self.params[k] = v
            setattr(self, k, v)

    def print_params(self):
        """
        Return a human-readable JSON string of all stored parameters.

        Returns:
            str: Formatted JSON string of current parameters.
        """
        return json.dumps(self.params, indent=2, default=str)

    def set_best_performance(self, best_auc, best_epoch):
        """
        Add the best AUC score and its corresponding epoch to the stored parameters.

        Args:
            best_auc (float): Best validation AUC achieved so far.
            best_epoch (int): Epoch number when the best AUC was recorded.
        """
        self.params['best_auc'] = best_auc
        self.params['best_epoch'] = best_epoch

    def save_params(self, json_file):
        """
        Save all stored parameters to a JSON file.

        Args:
            json_file (str): Path where the JSON file will be saved.
        """
        with open(json_file, 'w') as fp:
            json.dump(self.params, fp, indent=4)
