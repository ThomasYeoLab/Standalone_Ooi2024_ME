'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script provides helper functions for managing CNN model initialization, 
weight loading, fine-tuning settings, checkpoint saving, and plotting training curves.

It supports model initialization from scratch or from pretrained weights, selective
layer updating, and visualization of performance metrics across training epochs.
'''

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from matplotlib import pyplot as plt


def initialize_model(model,
                     device,
                     pretrained_file,
                     update_layers,
                     logger=None):
    """
    Initialize the CNN model for training or fine-tuning.

    If `pretrained_file` is provided, load pretrained weights and freeze or unfreeze layers
    depending on `update_layers`. Otherwise, initialize weights from scratch.

    Args:
        model (nn.Module): PyTorch model instance.
        device (torch.device): Device to load the model onto (e.g. 'cuda' or 'cpu').
        pretrained_file (str): Path to pretrained model weights (.pt file).
        update_layers (str): Either 'last' or 'all' — defines which layers to update.
        logger (logging.Logger, optional): Logger instance for verbose output.

    Returns:
        nn.Module: Initialized model on the specified device.
    """
    if pretrained_file:
        assert update_layers is not None, 'Missing input argument. finetune_layers \
            must not None when pretrain_file is not None.'

        if logger:
            logger.info('Load pretrained model from {}.\n'.format(
                os.path.realpath(pretrained_file)))
        model.load_state_dict(torch.load(pretrained_file, map_location=device))
        set_parameter_requires_grad(model, update_layers)
    else:
        if logger: logger.info('Initialize weights...\n')
        model.apply(init_weights_bias)

    return model.to(device)


def init_weights_bias(m):
    """
    Initialize weights and biases of the model layers using Xavier uniform initialization.

    Applies to Conv3D, BatchNorm3D, and Linear layers.

    Args:
        m (nn.Module): A PyTorch layer passed during `model.apply()`.
    """
    if isinstance(m, nn.Conv3d):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)


def set_parameter_requires_grad(model, update_layers):
    """
    Set the `requires_grad` flag for model parameters based on finetuning strategy.

    If 'last', freezes all parameters except the final fully connected layer.
    If 'all', unfreezes the entire model.

    Args:
        model (nn.Module): Model whose parameters will be updated.
        update_layers (str): Either 'last' or 'all'.

    Raises:
        ValueError: If `update_layers` is not one of the supported options.
    """

    if update_layers == 'last':  # finetune last layer
        # freeze the whole network
        for param in model.parameters():
            param.requires_grad = False

        # unfreeze last block
        for param in model.fc.parameters():
            param.requires_grad = True

    elif update_layers == 'all':  # all: don't do anything -> all the parameters will be update
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError("update_layers can only be last/all.")


def get_model_params(model):
    """
    Extract parameters that will be updated during training.

    Returns a list of parameters and their corresponding names for which `requires_grad` is True.

    Args:
        model (nn.Module): The model to extract trainable parameters from.

    Returns:
        tuple:
            - List[torch.nn.Parameter]: Parameters to update.
            - List[str]: Corresponding parameter names.
    """
    params_to_update, params_names = [], []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            params_names.append(name)

    return params_to_update, params_names


def save_checkpoint(epoch, model, optimizer, scheduler, save_path):
    """
    Save a model training checkpoint containing model, optimizer, and scheduler states.

    Args:
        epoch (int): Current training epoch.
        model (nn.Module): Trained model.
        optimizer (torch.optim.Optimizer): Optimizer in use.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        save_path (str): File path to save the checkpoint.
    """
    checkpoint = {
        'current_epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)


def plot_curve(train, val, plot_type, plot_fig):
    """
    Plot and save the training vs validation metric curve over epochs.

    Args:
        train (List[float]): Training metric values (e.g. loss or AUC).
        val (List[float]): Validation metric values.
        plot_type (str): Metric type used for Y-axis label ('loss' or 'AUC').
        plot_fig (str): Path to save the generated plot (e.g. 'curve.png').

    Raises:
        AssertionError: If the lengths of train and val lists do not match.
    """

    assert len(train) == len(val)

    n_epoch = len(train)

    plt.plot(np.arange(n_epoch).tolist(), train, color='blue', label='train')
    plt.plot(np.arange(n_epoch).tolist(), val, color='orange', label='val')
    plt.xlabel('epoch')
    plt.ylabel(plot_type)
    plt.legend(loc='best')
    plt.title('Train-val {}'.format(plot_type.upper()))
    plt.savefig(plot_fig)
    plt.clf()
    plt.close()
