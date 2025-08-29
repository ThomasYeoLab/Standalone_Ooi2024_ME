#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


class OneHot_Config(object):
    """
    Store the hyperparameter configurations for L2C-FNN model
    """

    def __init__(
        self,
        layer_num=3,
        input_dim=101,  # num_dim - 1 for RID
        nb_classes=3,
        nb_measures=2,
        hid_dim_list=[128, 128, 256],
        p_drop=0.5,
        relu_slope=0.01,
    ):
        self.layer_num = layer_num
        self.input_dim = input_dim
        self.nb_classes = nb_classes
        self.nb_measures = nb_measures
        self.hid_dim_list = hid_dim_list
        self.p_drop = p_drop
        self.relu_slope = relu_slope
