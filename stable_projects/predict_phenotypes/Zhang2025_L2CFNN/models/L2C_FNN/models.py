#!/usr/bin/env python
# Written by Chen Zhang and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    A multi-task learning FNN as the core prediction model
    """

    # add batchnorm
    # remove bias since batchnorm don't need bias?
    # dropout after activation instead of before
    # leakyReLU instead of ReLU
    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.ModuleList()
        current_dim = config.input_dim
        for hdim in config.hid_dim_list:
            self.linear_relu_stack.append(
                nn.Sequential(
                    nn.Linear(current_dim, hdim),
                    nn.LeakyReLU(config.relu_slope),
                    nn.Dropout(p=config.p_drop),
                )
            )
            current_dim = hdim
        self.features = nn.Sequential(*self.linear_relu_stack)
        self.hid2category = nn.Linear(current_dim, config.nb_classes)
        self.hid2measures = nn.Linear(current_dim, config.nb_measures)

    def forward(self, x):
        x = self.features(x)
        # here we shouldn't add softmax as we'll use crossentropy loss
        cat_out = self.hid2category(x)
        val_out = self.hid2measures(x)
        return cat_out, val_out
