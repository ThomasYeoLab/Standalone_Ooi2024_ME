"""
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

CNN model definitions for brain age prediction and classification using SFCN-based architectures.

This module contains:
- `SFCN`: A CNN architecture designed for age prediction with constrained output range (outputs brain age).
          This architecture is used by BAG and BAG-finetune models.
- `SFCN_FC`: A classification variant of SFCN that includes a final fully connected output layer
             (outputs binary class label). This architecture is used for Direct (both AD classification
             & MCI progression prediction) and Brainage64D-finetune.
- `SFCN_FC_Intermediate`: A variant of SFCN_FC that returns intermediate 64D features instead of predictions.
             This architecture is used for Direct-AD and Brainage64D-finetune-AD.

All models use 3D convolutions and batch normalization with an epsilon value set to 1e-3,
matching TensorFlow's default behavior.
"""

import torch.nn as nn
import torch


class SFCN(nn.Module):
    """
    Implementation of the original SFCN model for brain age prediction.

    The network includes:
    - A series of 3D convolutional blocks with batch normalization and ReLU activation.
    - Global average pooling and dropout before the final regression layer.
    - Output prediction constrained to a realistic age range using clamping and ReLU.

    Args:
        channel_number (list): List of channel sizes for each convolutional block.
        output_dim (int): Output dimensionality. Typically 1 for age regression.
        dropout (float): Dropout rate used before the final linear layer.
    """

    def __init__(self,
                 channel_number=[32, 64, 128, 256, 256, 64],
                 output_dim=1,
                 dropout=0.5):

        super(SFCN, self).__init__()
        n_layer = len(channel_number)

        # Feature extractor includes 6 blocks of conv layers
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer - 1):
            in_channel = channel_number[i - 1] if i > 0 else 1
            out_channel = channel_number[i]
            self.feature_extractor.add_module(
                'conv_%d' % i,
                self.conv_layer(in_channel,
                                out_channel,
                                kernel_size=3,
                                padding=1,
                                maxpool=2,
                                maxpool_stride=2))
        self.feature_extractor.add_module(
            'conv_%d' % (n_layer - 1),
            self.conv_layer(channel_number[-2],
                            channel_number[-1],
                            kernel_size=1,
                            padding=0))

        # Classifier: keep this name to make the code work with current trained model
        self.classifier = nn.Sequential()
        self.classifier.add_module('average_pool', nn.AvgPool3d([5, 6, 5]))
        self.classifier.add_module('dropout', nn.Dropout(dropout))

        # Age prediction layer
        self.fc = nn.Sequential()
        self.fc.add_module('fc_0', nn.Linear(channel_number[-1], output_dim))

    @staticmethod
    def conv_layer(in_channel,
                   out_channel,
                   kernel_size=3,
                   padding=0,
                   maxpool=None,
                   maxpool_stride=None):
        """
        Constructs a convolutional block with BatchNorm3d, ReLU, and optional MaxPool3d.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            kernel_size (int): Kernel size of the Conv3D layer.
            padding (int): Padding size.
            maxpool (int or tuple, optional): Size of maxpool kernel (if pooling is used).
            maxpool_stride (int or tuple, optional): Stride of maxpooling.

        Returns:
            nn.Sequential: Sequential container of layers.
        """
        layers = [
            nn.Conv3d(in_channel,
                      out_channel,
                      padding=padding,
                      kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel, eps=1e-3),
            nn.ReLU()
        ]
        if maxpool:
            assert maxpool_stride is not None
            layers.append(nn.MaxPool3d(maxpool, stride=maxpool_stride))

        return nn.Sequential(*layers)

    def forward(self, x, lower_age=3, upper_age=95):
        """
        Forward pass for the SFCN model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, D, H, W).
            lower_age (int): Minimum age to clamp the output.
            upper_age (int): Maximum age to clamp the output.

        Returns:
            torch.Tensor: Age predictions in the range [lower_age, upper_age].
        """
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        # Restrict predicted age like in pyment-public's restrict_range.py
        x = torch.relu(x)
        x = torch.clamp(x, max=(upper_age - lower_age))
        x = torch.add(x, lower_age)

        return x


class SFCN_FC(nn.Module):
    """
    Modified SFCN model for classification tasks.

    Similar to the original SFCN, but the final layer produces multi-class outputs instead of a single scalar.
    This version is based on SFCN with a final linear layer for classification, used in e.g., diagnostic prediction.

    Args:
        channel_number (list): List of channels for each convolutional layer.
        output_dim (int): Number of output classes.
        dropout (float): Dropout rate used before the classifier.
    """

    def __init__(self,
                 channel_number=[32, 64, 128, 256, 256, 64],
                 output_dim=2,
                 dropout=0.5):
        super(SFCN_FC, self).__init__()
        n_layer = len(channel_number)

        # Feature extractor includes 6 blocks of conv layers
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer - 1):
            in_channel = channel_number[i - 1] if i > 0 else 1
            out_channel = channel_number[i]
            self.feature_extractor.add_module(
                'conv_%d' % i,
                self.conv_layer(in_channel,
                                out_channel,
                                kernel_size=3,
                                padding=1,
                                maxpool=2,
                                maxpool_stride=2))
        self.feature_extractor.add_module(
            'conv_%d' % (n_layer - 1),
            self.conv_layer(channel_number[-2],
                            channel_number[-1],
                            kernel_size=1,
                            padding=0))

        # Classifier: keep this name to make the code work with current trained model
        self.classifier = nn.Sequential()
        self.classifier.add_module('average_pool', nn.AvgPool3d([5, 6, 5]))
        self.classifier.add_module('dropout', nn.Dropout(dropout))

        # Fully connected layer
        self.fc = nn.Sequential()
        self.fc.add_module('fc_0', nn.Linear(channel_number[-1], output_dim))

    @staticmethod
    def conv_layer(in_channel,
                   out_channel,
                   kernel_size=3,
                   padding=0,
                   maxpool=None,
                   maxpool_stride=None):
        """
        Convolutional layer with batch normalization and ReLU activation.
        If maxpool is not None, then a maxpooling layer is added after the convolutional layer.
        """
        layers = [
            nn.Conv3d(in_channel,
                      out_channel,
                      padding=padding,
                      kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel, eps=1e-3),
            nn.ReLU()
        ]
        if maxpool:
            assert maxpool_stride is not None
            layers.append(nn.MaxPool3d(maxpool, stride=maxpool_stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for SFCN_FC classification model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, D, H, W).

        Returns:
            torch.Tensor: Raw classification logits of shape (N, output_dim).
        """
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class SFCN_FC_Intermediate(SFCN_FC):
    """
    SFCN_FC variant that returns intermediate 64-dimensional feature vectors instead of final predictions.

    Useful for transfer learning, feature visualization, or further downstream tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        Forward pass to extract intermediate features.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, D, H, W).

        Returns:
            torch.Tensor: Flattened feature vectors of shape (N, 64).
        """
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = x.reshape(x.size(0), -1)

        return x
