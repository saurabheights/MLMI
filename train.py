from __future__ import print_function

import argparse
import logging
import os
import random
from pprint import pformat
from typing import List

import numpy as np
import torch
import torch.utils.data

from callbacks.callback_utils import run_callbacks
from callbacks.callbacks import CallbackMode, Callbacks
from dataset.BaseDataset import BaseDataset
from dataset.factory import create_dataset, SupportedDataset, MAP_DATASET_TO_ENUM
from loss.utils import create_loss
from models.utils import get_model
from optimizer.utils import create_optimizer
from utils import logger
from utils.RunningAverage import RunningAverage
from utils.fileutils import make_results_dir
from utils.progress_bar import ProgressBar
from utils.tensorboard_writer import initialize_tensorboard, close_tensorboard

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=50,
                    help='Number of epochs for training. Default 50.')
parser.add_argument('--dataset', type=str, default='MNIST', choices=['CIFAR10', 'MNIST'],
                    help='Required - The dataset to choose')

# Data Inflation Study, allows training on smaller subset of selected Dataset
parser.add_argument('--training_subset_percentage', type=float, default=1.0,
                    help='Optional - Subset of data to use for training')

# Pretrained weights
parser.add_argument('--model_weights', type=str, required=False, default=None,
                    help='Optional - Pretrained weights')
parser.add_argument('--output_dir', type=str, required=False, default='./logs/',
                    help='Optional - Output directory path')

opt = parser.parse_args()


def generate_callbacks(arguments: dict,
                       dataset: BaseDataset,
                       device,
                       outdir) -> List[Callbacks]:
    callbacks = []
    """
    Create and add all callbacks instances to `callbacks` list 
    """

    return callbacks


def objective(arguments):
    """
    Main Pipeline for training and cross-validation. ToDo - Testing will be done separately in test.py.
    """

    """ Setup result directory and enable logging to file in it """
    outdir = make_results_dir(arguments)
    logger.init(outdir, logging.INFO)
    logger.info('Arguments:\n{}'.format(pformat(arguments)))

    """ Initialize Tensorboard """
    tensorboard_writer = initialize_tensorboard(outdir)

    """ Set random seed throughout python, pytorch and numpy """
    logger.info('Using Random Seed value as: %d' % arguments['random_seed'])
    torch.manual_seed(arguments['random_seed'])  # Set for pytorch, used for cuda as well.
    random.seed(arguments['random_seed'])  # Set for python
    np.random.seed(arguments['random_seed'])  # Set for numpy

    """ Set device - cpu or gpu """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device - {device}')

    """ Load Model with weights(if available) """
    model: torch.nn.Module = get_model(arguments.get('model_args')).to(device)

    """ Create loss function """
    criterion = create_loss(arguments['loss_args'])

    """ Create optimizer """
    optimizer = create_optimizer(model.parameters(), arguments['optimizer_args'])

    """ Load parameters for the Dataset """
    dataset: BaseDataset = create_dataset(arguments['dataset_args'],
                                          arguments['train_data_args'],
                                          arguments['val_data_args'])

    """ Generate all callbacks """
    callbacks: List[Callbacks] = generate_callbacks(arguments, dataset, device, outdir)

    """ Debug the inputs to model and save graph to tensorboard """
    dataset.debug()
    dummy_input = (torch.rand(1,
                              arguments['dataset_args']['name'].value['channels'],
                              *arguments['dataset_args']['name'].value['image_size'],
                              )).to(device)
    tensorboard_writer.save_graph(model, dummy_input)

    """ Pipeline - loop over the dataset multiple times """
    max_validation_accuracy = 0
    itr = 0

    run_callbacks(callbacks, model=model, mode=CallbackMode.ON_TRAIN_BEGIN)
    for epoch in range(arguments['nb_epochs']):
        """ Train the model """
        train_data_args = arguments['train_data_args']
        if train_data_args['to_train']:
            train_dataloader = dataset.train_dataloader
            progress_bar = ProgressBar(target=len(train_dataloader),
                                       clear=True,
                                       description=f"Training {epoch + 1}/{arguments['nb_epochs']}: ")
            loss_running_average = RunningAverage()

            run_callbacks(callbacks, model=model, mode=CallbackMode.ON_EPOCH_BEGIN, epoch=epoch)
            model.train()
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward Pass
                outputs = model(inputs)

                classification_loss = criterion(outputs, labels)
                tensorboard_writer.save_scalar('Classification_Loss', classification_loss.item(), itr)
                classification_loss.backward()
                optimizer.step()

                # Compute running loss. Not exact but efficient.
                running_loss = loss_running_average.add_new_sample(classification_loss.item())
                progress_bar.update(i + 1,
                                    [('current loss', classification_loss.item()),
                                     ('running loss', running_loss), ])
                tensorboard_writer.save_scalar('Training_Loss', classification_loss, itr)
                itr += 1

            # Callbacks ON_EPOCH_END should be run only when training is enabled. Thus call here.
            run_callbacks(callbacks, model=model, mode=CallbackMode.ON_EPOCH_END, epoch=epoch)

        """ Validate the model """
        val_data_args = arguments['val_data_args']
        if val_data_args['validate_step_size'] > 0 and \
                epoch % val_data_args['validate_step_size'] == 0:
            correct, total = 0, 0
            validation_dataloader = dataset.validation_dataloader
            progress_bar = ProgressBar(target=len(validation_dataloader),
                                       clear=True,
                                       description=f"Validating {epoch + 1}/{arguments['nb_epochs']}: ")
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(validation_dataloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    progress_bar.update(i + 1,
                                        [('Batch Accuracy', 100 * correct / total), ])

            val_accuracy = 100 * correct / total
            tensorboard_writer.save_scalar('Validation_Accuracy', val_accuracy, itr)
            logger.info(
                f'Accuracy of the network on the {dataset.get_val_dataset_size} validation images: {val_accuracy} %%')

            """ Save Model """
            if val_accuracy > max_validation_accuracy:
                torch.save(model.state_dict(),
                           os.path.join(outdir, f'epoch_{epoch:04}-model-val_accuracy_{val_accuracy}.pth'))
                max_validation_accuracy = val_accuracy

        tensorboard_writer.flush()

        # Exit loop if training not needed
        if not train_data_args['to_train']:
            break

    run_callbacks(callbacks, model=model, mode=CallbackMode.ON_TRAIN_END)

    logger.info('Finished Training')
    close_tensorboard()
    logger.info(f'Max Validation accuracy is {max_validation_accuracy}')
    return max_validation_accuracy  # Return in case later u wanna add hyperopt.


def main():
    dataset_args = dict(
        name=MAP_DATASET_TO_ENUM[opt.dataset],
        training_subset_percentage=opt.training_subset_percentage,
        # For Data Inflation Study - Set to None to use full dataset
    )

    train_data_args = dict(
        batch_size=64,
        shuffle=True,
        to_train=True,
    )

    val_data_args = dict(
        batch_size=train_data_args['batch_size'] * 4,
        shuffle=False,
        validate_step_size=1,
    )

    model_args = dict(
        # Use Enums here
        model_arch_name='classification.ConvNetSimple.ConvNetSimple',
        model_weights_path=opt.model_weights,
        model_constructor_args=dict(
            input_size=dataset_args['name'].value['image_size'],
            number_of_input_channels=dataset_args['name'].value['channels'],
            number_of_classes=dataset_args['name'].value['labels_count'],
        )
    )

    loss_args = dict(
        name='torch.nn.CrossEntropyLoss'
    )

    optimizer_args = dict(
        name='torch.optim.Adam',
        lr=1e-3
    )

    arguments = dict(
        dataset_args=dataset_args,
        train_data_args=train_data_args,
        val_data_args=val_data_args,
        model_args=model_args,
        loss_args=loss_args,
        optimizer_args=optimizer_args,
        outdir=opt.output_dir,
        nb_epochs=opt.num_epoch,
        random_seed=42
    )

    try:
        objective(arguments)
    except Exception as e:
        logger.exception("Exception caught from objective function")


if __name__ == '__main__':
    main()
