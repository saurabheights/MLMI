import argparse
import logging
import random
from pprint import pformat
from typing import List

import numpy as np
import torch.utils.data

from callbacks.callback_utils import generate_callbacks, run_callbacks
from callbacks.callbacks import Callbacks, CallbackMode
from dataset.BaseDataset import BaseDataset
from dataset.factory import create_dataset, MAP_DATASET_TO_ENUM
from models.utils import get_model
from utils import logger
from utils.fileutils import make_results_dir
from utils.tensorboard_writer import initialize_tensorboard

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='dcgan', choices=['dcgan', 'wgan-wp'],
                    help='Optional - To train dcgan or wgan. Default value dcgan.')

parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CELEBA'],
                    help='Optional - The dataset to choose')

# Data Inflation Study, allows training on smaller subset of selected Dataset
parser.add_argument('--training_subset_percentage', type=float, default=1.0,
                    help='Optional - Subset of data to use for training. Default use whole dataset')

parser.add_argument('--generator_model_path', type=str, required=True,
                    help='Path for generator pretrained weights.')

parser.add_argument('--output_dir', type=str, required=False, default='./logs/',
                    help='Optional - Where to create output directory path. Default ./logs.')

parser.add_argument('--gpu', type=int, default=0,
                    help='Optional - If gpu available, which one to use. Default value 0.')

opt = parser.parse_args()


def infinite_train_gen(dataloader):
    def f():
        while True:
            for images, targets in dataloader:
                yield images, targets

    return f()


def eval_gan(arguments):
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
    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device - {device}')

    """ Load Model with weights(if available) """
    G: torch.nn.Module = get_model(arguments.get('generator_model_args')).to(device)
    D: torch.nn.Module = get_model(arguments.get('discriminator_model_args')).to(device)
    logger.info(G)
    logger.info(D)

    """ Load parameters for the Dataset """
    dataset: BaseDataset = create_dataset(arguments['dataset_args'],
                                          arguments['train_data_args'],
                                          arguments['val_data_args'])

    """ Generate all callbacks """
    callbacks: List[Callbacks] = generate_callbacks(arguments, dataset, device, outdir)

    global_step = 20000
    run_callbacks(callbacks,
                  model=(G, D),
                  optimizer=None,  # To Save optimizer dict for retraining.
                  mode=CallbackMode.ON_TRAIN_END,
                  iteration=global_step)


def main():
    dataset_specific_configs = dict(
        MNIST=dict(
            training_batch_size=64,
            z_dim=100,
            inception_metric=dict(
                evaluation_arch_name='models.classification.ConvNetSimple.ConvNetSimple',
                evaluation_classifier_weights='logs/2019-12-27T13:09:07.398172_mode_classification_model_ConvNetSimple_dataset_MNIST_subset_1.0_bs_64_name_Adam_lr_0.001_weight_decay_0.005/epoch_0034-model-val_accuracy_98.06859806859806.pth',
                classifier_model_layer=4,
                evaluation_size=10000,
                evaluation_classifier_std=(0.5,),
                evaluation_classifier_mean=(0.5,)
            ),
            fid_metric=dict(
                evaluation_arch_name='models.evaluation.inception.InceptionV3',
                evaluation_classifier_weights=None,
                classifier_model_layer=None,
                evaluation_size=10000,
                evaluation_classifier_std=(0.5, 0.5, 0.5),
                evaluation_classifier_mean=(0.5, 0.5, 0.5)
            )
        )
    )

    assert opt.dataset in dataset_specific_configs.keys()
    dataset_specific_config = dataset_specific_configs[opt.dataset]

    dataset_args = dict(
        name=MAP_DATASET_TO_ENUM[opt.dataset],
        training_subset_percentage=opt.training_subset_percentage,
        mean=(0.5,),
        std=(0.5,)
        # For Data Inflation Study - Set to None to use full dataset
    )

    train_data_args = dict(
        batch_size=dataset_specific_config['training_batch_size'],
        shuffle=True,
        to_train=True,
    )

    val_data_args = dict(
        batch_size=train_data_args['batch_size'] * 4,
        shuffle=False,
        validate_step_size=1,
    )

    generator_model_args = dict(
        # Use Enums here
        model_arch_name='models.gan.DCGAN.Generator',
        model_weights_path=opt.generator_model_path,
        model_constructor_args=dict(
            z_dim=dataset_specific_config['z_dim'],
            channels=dataset_args['name'].value['channels'],
        )
    )

    discriminator_model_args = dict(
        # Use Enums here
        model_arch_name='models.gan.DCGAN.Discriminator',
        model_weights_path=opt.generator_model_path.replace('G-', 'D-'),
        model_constructor_args=dict(
            channels=dataset_args['name'].value['channels']
        )
    )

    # dataset_inception_metric = dataset_specific_config['inception_metric']
    # dataset_fid_metric = dataset_specific_config['fid_metric']
    callbacks_args = [
        # ToDo - Following callback wont work, method ON_Train_End not implemented
        # dict(InceptionMetric=dict(sample_size=train_data_args['batch_size'],
        #                           total_samples=dataset_inception_metric['evaluation_size'],
        #                           classifier_model_args=dict(
        #                               # Use Enums here
        #                               model_arch_name=dataset_inception_metric['evaluation_arch_name'],
        #                               model_weights_path=dataset_inception_metric['evaluation_classifier_weights'],
        #                               model_constructor_args=dict(
        #                                   input_size=dataset_args['name'].value['image_size'],
        #                                   number_of_input_channels=dataset_args['name'].value['channels'],
        #                                   number_of_classes=dataset_args['name'].value['labels_count'],
        #                               )
        #                           ),
        #                           transform=dict(mean=dataset_inception_metric['evaluation_classifier_mean'],
        #                                          std=dataset_inception_metric['evaluation_classifier_std']),
        #                           mode='gan')
        #      ),
        # dict(FrechetMetric=dict(sample_size=train_data_args['batch_size'],
        #                         total_samples=dataset_fid_metric['evaluation_size'],
        #                         classifier_model_args=dict(
        #                             # Use Enums here
        #                             model_arch_name=dataset_fid_metric['evaluation_arch_name'],
        #                             model_weights_path=dataset_fid_metric['evaluation_classifier_weights'],
        #                             model_constructor_args=dict()
        #                         ),
        #                         classifier_model_layer=dataset_fid_metric['classifier_model_layer'],
        #                         transform=dict(mean=dataset_fid_metric['evaluation_classifier_mean'],
        #                                        std=dataset_fid_metric['evaluation_classifier_std']), )
        #      ),
        # dict(GanSampler=dict(write_to_tensorboard=True,
        #                      write_to_disk=True,
        #                      num_samples=16, )
        #      ),
        dict(GanEmbeddingSampler=dict(write_to_tensorboard=True,
                                      run_on_nth_iteration=False,
                                      run_on_train_end=True,
                                      batch_size=val_data_args['batch_size'],
                                      num_samples=2500,  # 400 MB size data for 10000 images and tensorboard too slow.
                                      step_size=5
                                      )
             )
    ]

    arguments = dict(
        mode=opt.mode,
        dataset_args=dataset_args,
        train_data_args=train_data_args,
        val_data_args=val_data_args,
        generator_model_args=generator_model_args,
        discriminator_model_args=discriminator_model_args,
        callbacks_args=callbacks_args,
        outdir=opt.output_dir,
        random_seed=dataset_specific_config.get('random_seed', random.randint(0, 100))
    )

    try:
        eval_gan(arguments)
    except Exception as e:
        logger.exception("Exception caught from objective function")
        raise


if __name__ == '__main__':
    main()
