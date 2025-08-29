'''
Written by Kim-Ngan Nguyen, Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script trains and evaluates SFCN or SFCN_FC models on a validation set using
MAE (for brain age prediction) or AUC (for classification) as performance metrics.

This script supports two main tasks:
1. Finetuning for brain age prediction using the SFCN model, evaluated with MAE.
2. AD classification or MCI progression prediction using the SFCN_FC model, evaluated with AUC.

Expected outputs:
- Best model weights: `best_[score].pt`
- Training/validation performance logs: `train_loss.txt`, `val_auc.txt`, etc.
- Hyperparameter configuration: `params.json`

Example:
    cd $TANNGUYEN2025_BA_DIR; conda activate CBIG_BA;
    python -m model.cnn.CBIG_BA_train --data_split /not/real/path \
        --data_split_csv /not/real/path \
            --out_dir /not/real/path --train_csv /not/real/path \
                 --val_csv /not/real/path --task ad_classification;
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import copy
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from shutil import copy2

from model.utils import CBIG_BA_io
from model.cnn.modules.CBIG_BA_dataset import ADDataset
from model.cnn.modules.CBIG_BA_params import ModelParameters
from model.cnn.modules.CBIG_BA_sfcn import SFCN, SFCN_FC
from model.cnn.modules.CBIG_BA_helper import plot_curve, initialize_model, save_checkpoint, get_model_params
import torch.nn.functional as F

from CBIG_BA_config import global_config
from utils.CBIG_BA_complete_step import generate_complete_flag


def set_random_seeds(seed):
    """
    Set all relevant random seeds for reproducibility.

    Args:
        seed (int): Seed value to ensure deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_step(args,
               model,
               loss_fn,
               train_loader,
               train_size,
               optimizer,
               device,
               history,
               logger=None):
    """
    Perform one training epoch.

    Args:
        args (argparse.Namespace): Configuration parameters.
        model (nn.Module): Model to train.
        loss_fn (nn.Module): Loss function.
        train_loader (DataLoader): Training data loader.
        train_size (int): Number of training samples.
        optimizer (Optimizer): Optimizer for training.
        device (torch.device): Computation device (CPU/GPU).
        history (dict): Training history dictionary to store loss and score.
        logger (logging.Logger, optional): Logger to log training progress.
    """
    # Initialize variables
    running_loss = 0.0
    pred_score_epoch = []
    label_epoch = []
    age_epoch = []

    # Iterate through training set
    for _, inputs, ages, labels in train_loader:
        inputs = inputs.to(device)
        if args.model_type == 'SFCN':
            ages = ages.to(device)
        elif args.model_type == 'SFCN_FC':
            labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Make prediction
        outputs = model(inputs)

        # Append ground truth labels &
        # predicted labels
        if args.model_type == 'SFCN':
            pred_scores = outputs
            ages = ages.unsqueeze(1)
            loss = loss_fn(outputs, ages)
            age_epoch.extend(ages.tolist())
        elif args.model_type == 'SFCN_FC':
            outputs_sm = F.softmax(outputs, dim=1)
            pred_scores = outputs_sm[:, 1]
            loss = loss_fn(outputs, labels)
            label_epoch.extend(labels.tolist())

        # Compute loss and do gradient descend
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Compute loss
        running_loss += loss.detach() * inputs.size(0)
        pred_score_epoch.extend(pred_scores.tolist())

    epoch_loss = running_loss / train_size
    if args.score == 'mae':
        epoch_score = mean_absolute_error(age_epoch, pred_score_epoch)
    elif args.score == 'auc':
        epoch_score = roc_auc_score(label_epoch, pred_score_epoch)

    # Save history
    history['loss'].append(epoch_loss)
    history[args.score].append(epoch_score)
    if logger:
        logger.info('--- TRAIN. Loss: {:.4f}, {}: {:.4f}'.format(
            epoch_loss, args.score, epoch_score))


def val_step(args,
             model,
             loss_fn,
             val_loader,
             val_size,
             device,
             history,
             logger=None):
    """
    Evaluate the model on the validation set.

    Args:
        args (argparse.Namespace): Configuration parameters.
        model (nn.Module): Model to evaluate.
        loss_fn (nn.Module): Loss function.
        val_loader (DataLoader): Validation data loader.
        val_size (int): Number of validation samples.
        device (torch.device): Computation device (CPU/GPU).
        history (dict): Validation history dictionary to store loss and score.
        logger (logging.Logger, optional): Logger to log validation results.

    Returns:
        float: Validation performance score (MAE or AUC).
    """
    # Initialize variables
    running_loss = 0.0
    pred_score_epoch = []
    label_epoch = []
    age_epoch = []

    # Iterate through validation set
    for _, inputs, ages, labels in val_loader:
        inputs = inputs.to(device)
        if args.model_type == 'SFCN':
            ages = ages.to(device)
        elif args.model_type == 'SFCN_FC':
            labels = labels.to(device)

        # Make prediction
        outputs = model(inputs)

        # Append ground truth labels &
        # predicted labels
        if args.model_type == 'SFCN':
            pred_scores = outputs
            ages = ages.unsqueeze(1)
            loss = loss_fn(outputs, ages)
            age_epoch.extend(ages.tolist())
        elif args.model_type == 'SFCN_FC':
            outputs_sm = F.softmax(outputs, dim=1)
            pred_scores = outputs_sm[:, 1]
            loss = loss_fn(outputs, labels)
            label_epoch.extend(labels.tolist())

        # Compute loss
        running_loss += loss.detach() * inputs.size(0)
        pred_score_epoch.extend(pred_scores.tolist())

    epoch_loss = running_loss / val_size
    if args.score == 'mae':
        epoch_score = mean_absolute_error(age_epoch, pred_score_epoch)
    elif args.score == 'auc':
        epoch_score = roc_auc_score(label_epoch, pred_score_epoch)

    # Save history
    history['loss'].append(epoch_loss)
    history[args.score].append(epoch_score)
    if logger:
        logger.info('--- VAL Loss: {:.4f}, {}: {:.4f}\n'.format(
            epoch_loss, args.score, epoch_score))

    return epoch_score


def main_1_split(args):
    """
    Main function to train and validate a model for one specific data split.

    This function initializes the model, loads data, runs training and validation
    for multiple epochs, and saves the best-performing model based on validation metrics.

    Args:
        args (argparse.Namespace): Input arguments containing paths, hyperparameters, etc.

    Returns:
        float: Best validation score (lowest MAE or highest AUC).
    """

    # Set random seeds
    set_random_seeds(args.seed)

    # Set output and plot name
    start_time, output_dir = CBIG_BA_io.create_output_folder(args.out_dir)
    plot_loss_png = os.path.join(output_dir, 'training_loss.png')
    plot_score_png = os.path.join(output_dir, f'training_{args.score}.png')

    # Set logging
    logger = logging.getLogger()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        os.path.join(output_dir, '{}.log'.format(args.model_type)))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info('Training with input argument: {} \n'.format(str(args)))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    if args.model_type == 'SFCN':
        model = SFCN(dropout=args.dropout)
    elif args.model_type == 'SFCN_FC':
        model = SFCN_FC(dropout=args.dropout)
    model = initialize_model(model,
                             device=device,
                             pretrained_file=args.pretrain_file,
                             update_layers=args.finetune_layers,
                             logger=logger)
    logger.info(model)

    # Send the model to GPU if available
    model = model.to(device)
    logger.info(f"Model is at GPU.") if next(
        model.parameters()).is_cuda else logger.info(f"Model is at CPU.")

    # Get model parameters
    params_to_update, params_name = get_model_params(model)
    logger.info('Params to learn:')
    logger.info('\t' + '\n\t'.join(params_name))

    # Hyperparameters
    hp = ModelParameters(time_stamp=start_time,
                         params_dict={
                             'batch_size': args.batch_size,
                             'n_epoch': args.epochs,
                             'dropout': args.dropout,
                             'optim': args.optim_type,
                             'weight_decay': args.weight_decay,
                             'init_lr': args.init_lr,
                             'lr_decay': args.lr_decay,
                             'lr_step': args.lr_step
                         })
    logger.info('\nHYPERPARAMETERS\n{}'.format(hp.print_params()))

    # Set optimizer and learning rate scheduler
    optimizer = optim.SGD(params_to_update,
                          lr=hp.init_lr,
                          weight_decay=hp.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=hp.lr_step,
                                             gamma=hp.lr_decay)

    # Loss function
    if args.score == 'mae':
        loss_fn = nn.L1Loss()
    elif args.score == 'auc':
        loss_fn = nn.CrossEntropyLoss()

    # Set up dataloader for train and validation: train_csv, val_csv
    t1_dirs = global_config.T1_DIRS

    training_data = ADDataset(t1_dirs,
                              label_csv=args.train_csv,
                              task=args.task)
    train_dataloader = DataLoader(training_data,
                                  batch_size=hp.batch_size,
                                  shuffle=True,
                                  num_workers=4)
    validation_data = ADDataset(t1_dirs,
                                label_csv=args.val_csv,
                                task=args.task)
    val_dataloader = DataLoader(validation_data,
                                batch_size=hp.batch_size,
                                shuffle=True,
                                num_workers=4)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(training_data), 'val': len(validation_data)}

    # Output log information
    logger.info('\nPATHS')
    logger.info('out_dir {}'.format(args.out_dir))
    logger.info('train_csv {}'.format(args.train_csv))
    logger.info('val_csv {}'.format(args.val_csv))
    logger.info('T1_dirs {}'.format(str(t1_dirs)))
    logger.info('\nInitializing Datasets and Dataloaders...')
    logger.info('# training samples {}, # validation samples {}'.format(
        dataset_sizes['train'], dataset_sizes['val']))

    # Train and evaluate
    start = time.time()

    # Monitor loss and score
    history = {
        'train': {
            'loss': [],
            args.score: []
        },
        'val': {
            'loss': [],
            args.score: []
        },
        f'best_{args.score}': None,
        'best_epoch': None
    }

    # Initialize best model
    best_model_wts = copy.deepcopy(model.state_dict())
    if args.score == 'mae':
        best_score = float('inf')
    elif args.score == 'auc':
        best_score = 0
    best_epoch = -1

    # Main iteration: iterate through each epoch and update weights
    logger.info('\nStart training...')
    for epoch in range(hp.n_epoch):
        logger.info('Epoch {}/{}; lr: {}'.format(
            epoch, hp.n_epoch - 1, optimizer.param_groups[0]['lr']))

        # TRAINING
        # Train 1 epoch
        model.train()
        train_step(args, model, loss_fn, dataloaders['train'],
                   dataset_sizes['train'], optimizer, device, history['train'],
                   logger)

        # VALIDATION
        # Predict on validation set and update the best model
        with torch.set_grad_enabled(False):
            model.eval()
            epoch_val_score = val_step(args, model, loss_fn,
                                       dataloaders['val'],
                                       dataset_sizes['val'], device,
                                       history['val'], logger)

            # Save the current best model
            if args.score == 'mae':
                if epoch_val_score < best_score:
                    best_epoch = epoch
                    best_score = epoch_val_score
                    best_model_wts = copy.deepcopy(model.state_dict())
            elif args.score == 'auc':
                if epoch_val_score > best_score:
                    best_epoch = epoch
                    best_score = epoch_val_score
                    best_model_wts = copy.deepcopy(model.state_dict())
        # Update learning rate for next epoch
        lr_scheduler.step()

        # Update plot of loss and score
        if args.plot_during_training and ((epoch + 1) %
                                          args.plot_during_training == 0):
            plot_curve([i.item() for i in history['train']['loss']],
                       [i.item() for i in history['val']['loss']], 'loss',
                       plot_loss_png)
            plot_curve(history['train'][args.score],
                       history['val'][args.score], args.score, plot_score_png)

        # Save current model every k epochs or at the last epoch
        if ((epoch + 1) % 50 == 0) or ((epoch + 1) == hp.n_epoch):
            save_checkpoint(epoch,
                            model,
                            optimizer,
                            lr_scheduler,
                            save_path=os.path.join(output_dir,
                                                   'checkpoint.pt'))

    # Output log information
    time_elapsed = time.time() - start
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val {}: {:.4f} at epoch {}'.format(
        args.score, best_score, best_epoch))

    # Save loss, score history to files
    for phase in ['train', 'val']:
        for metric in ['loss', args.score]:
            history_txt = os.path.join(output_dir,
                                       '{}_{}.txt'.format(phase, metric))
            logger.info(f'Saved {phase}_{metric}_history to {history_txt}')
            CBIG_BA_io.list2txt(history[phase][metric], history_txt)

    # Save hyperparameters
    hp.set_best_performance(best_score, best_epoch)
    hp.save_params(json_file=os.path.join(output_dir, 'params.json'))

    # Save best model performance and weights
    history[f'best_{args.score}'] = best_score
    history['best_epoch'] = best_epoch
    model.load_state_dict(best_model_wts)
    model_pt = os.path.join(output_dir, f'best_{args.score}.pt')
    torch.save(model.state_dict(), model_pt)

    # Clear memory
    del model
    torch.cuda.empty_cache()

    # Remove the current logger handle
    logger.removeHandler(fh)

    return best_score


def get_args():
    """
    Get arguments from command line and return args.
    """
    parser = argparse.ArgumentParser()

    # general parameters - folders and files
    parser.add_argument('--data_split',
                        nargs="+",
                        type=int,
                        help='List of datasplit ID')
    parser.add_argument(
        '--data_split_csv',
        type=str,
        help=
        'Hyperparameters file with each row is for 1 data split. First column must be split.',
        default=None)
    parser.add_argument('--out_dir', type=str, help='Output directory.')
    parser.add_argument('--train_csv', type=str, help='Train csv file.')
    parser.add_argument('--val_csv', type=str, help='Val csv file.')
    parser.add_argument('--pretrain_file',
                        type=str,
                        help='Pretrained model from task 1.',
                        default=None)
    parser.add_argument(
        '--task',
        type=str,
        choices=['ad_classification', 'mci_progression'],
        help='Task name. Used to determine dataset labels and output folder')
    parser.add_argument('--score',
                        type=str,
                        choices=['mae', 'auc'],
                        default='auc')

    # general parameters for network
    parser.add_argument('--model_type',
                        type=str,
                        choices=['SFCN', 'SFCN_FC'],
                        help='Model architecture',
                        default='SFCN_FC')
    parser.add_argument('--optim_type',
                        type=str,
                        choices=['sgd'],
                        help='Optimizer type: sgd',
                        default='sgd')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument(
        '--finetune_layers',
        type=str,
        choices=['all', 'last'],
        help='Layers to finetune if pretrain_file is not None.',
        default=None)
    parser.add_argument(
        '--plot_during_training',
        type=int,
        help='If not None, plot loss and performance score every n epoch.',
        default=50)

    # hyperparameter
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_step', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)

    return parser.parse_args()


def main(args):
    """
    Read command line arguments, change input data split folder and pretrained file based on the current seed_id.
    """
    # Read command line arguments
    out_dir_template = args.out_dir
    train_csv_template = args.train_csv
    val_csv_template = args.val_csv
    pretrain_pt = args.pretrain_file

    # Remove data split list out of input argument list
    data_split_list = args.data_split
    del args.data_split

    # Use hyperparameters from csv file if available
    if args.data_split_csv:
        print(
            "WARNING: Using hyperparameters from CSV file, not input arguments"
        )
        params = pd.read_csv(args.data_split_csv, index_col=0)
        assert all(
            i in params.index.tolist() for i in data_split_list
        ), "All data splits must have hyperparameters in the csv file."
        params_name = params.columns.tolist()

        params = params.to_dict('index')

    # Loop through each train-val-test split
    for seed_id in data_split_list:
        # Replace placeholder values with actual seed ID
        args.out_dir = out_dir_template.replace('SEEDID', str(seed_id))
        args.train_csv = train_csv_template.replace('SEEDID', str(seed_id))
        args.val_csv = val_csv_template.replace('SEEDID', str(seed_id))
        # print('Input arguments:', str(args))

        # Hyperparameters of the current split
        for p in params_name:
            setattr(args, p, params[seed_id][p])

        # Replace placeholder values with actual seed ID
        if pretrain_pt is not None:
            args.pretrain_file = pretrain_pt.replace('SEEDID', str(seed_id))
            print('Pretrained file:', os.path.realpath(args.pretrain_file))
        main_1_split(args)

        # Generate complete flags for next step to commence
        generate_complete_flag(args.out_dir, append='train')


if __name__ == "__main__":
    main(get_args())
