import argparse
import logging
import random
from pprint import pformat
from typing import List

import numpy as np
import torch.utils.data
from tqdm import trange

from callbacks.callback_utils import generate_callbacks, run_callbacks
from callbacks.callbacks import Callbacks, CallbackMode
from dataset.BaseDataset import BaseDataset
from dataset.factory import create_dataset, MAP_DATASET_TO_ENUM
from models.utils import get_model
from optimizer.utils import create_optimizer
from utils import logger
from utils.fileutils import make_results_dir
from utils.sysutils import is_debug_mode
from utils.tensorboard_writer import initialize_tensorboard

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='dcgan', choices=['dcgan', 'wgan-wp'],
                    help='Optional - To train dcgan or wgan. Default value dcgan.')

parser.add_argument('--num_iterations', type=int, default=10000,
                    help='Optional - Number of iterations for training gan. Default value 10000.')

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


def train_gan(arguments):
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

    """ Create optimizer """
    G_optimizer = create_optimizer(G.parameters(), arguments['generator_optimizer_args'])
    D_optimizer = create_optimizer(D.parameters(), arguments['discriminator_optimizer_args'])

    """ Create Loss """
    loss = torch.nn.BCELoss().to(device=device)  # GAN

    """ Load parameters for the Dataset """
    dataset: BaseDataset = create_dataset(arguments['dataset_args'],
                                          arguments['train_data_args'],
                                          arguments['val_data_args'])

    """ Generate all callbacks """
    callbacks: List[Callbacks] = generate_callbacks(arguments, dataset, device, outdir)

    # """ Create loss function """
    # criterion = create_loss(arguments['loss_args'])

    """ Debug the inputs to model and save graph to tensorboard """
    dataset.debug()

    # Only One model is allowed
    # G_dummy_input = torch.rand(size=(1, arguments['generator_model_args']['model_constructor_args']['latent_dim']))
    # D_dummy_input = (torch.rand(1,
    #                           arguments['dataset_args']['name'].value['channels'],
    #                           32, 32  # *arguments['dataset_args']['name'].value['image_size']  # ToDo Fix this
    #                           ))
    # tensorboard_writer.save_graph('Generator', G, G_dummy_input.to(device))
    # tensorboard_writer.save_graph('Discriminator', D, D_dummy_input.to(device))
    logger.info(G)
    logger.info(D)

    def reset_grad():
        G.zero_grad()
        D.zero_grad()

    batch_size = arguments['train_data_args']['batch_size']
    z_dim = arguments['generator_model_args']['model_constructor_args']['z_dim']

    generator = infinite_train_gen(dataset.train_dataloader)
    interval_length = 10 if is_debug_mode() else 400
    num_intervals = 1 if is_debug_mode() else int(arguments['num_iterations'] / interval_length)

    global_step = 0
    for it in range(num_intervals):

        logger.info(f'Interval {it + 1}/{num_intervals}')

        # Set model in train mode
        G.train()
        D.train()

        t = trange(interval_length)
        for _ in t:
            if arguments['mode'] == 'dcgan':
                D_loss, G_loss = train_gan_iter(D, D_optimizer, G, G_optimizer,
                                                loss, device, generator, batch_size, reset_grad, z_dim)
            elif arguments['mode'] == 'wgan-wp':
                D_loss, G_loss = train_wgan_iter(D, D_optimizer, G, G_optimizer, device,
                                                 generator, batch_size, reset_grad, z_dim)

            # Log D_Loss and G_Loss in progress_bar
            t.set_postfix(D_Loss=D_loss.data.cpu().item(),
                          G_Loss=G_loss.data.cpu().item())

            # Save Loss In Tensorboard
            tensorboard_writer.save_scalars(f'{arguments["mode"].upper()}_Loss',
                                            {
                                                'Discriminator' if arguments['mode'] == 'dcgan' else
                                                'Critic': D_loss.data.cpu().item(),
                                                'Generator': G_loss.data.cpu().item()},
                                            global_step)
            global_step += 1

        print(f'Discriminator Loss: {D_loss.data.cpu().item()}, Generator Loss: {G_loss.data.cpu().item()}')

        run_callbacks(callbacks,
                      model=(G, D),
                      optimizer=(G_optimizer, D_optimizer),  # To Save optimizer dict for retraining.
                      mode=CallbackMode.ON_NTH_ITERATION, iteration=global_step)
        reset_grad()


def train_gan_iter(D, D_optimizer, G, G_optimizer,
                   loss, device, generator, batch_size, reset_grad, z_dim):
    real_labels = torch.ones(batch_size).to(device)
    fake_labels = torch.zeros(batch_size).to(device)

    # Train discriminator
    # Compute BCE_Loss using real images
    images, _ = next(generator)
    images = images.to(device)
    outputs = D(images)
    D_loss_real = loss(outputs.squeeze(), real_labels)

    # Compute BCE Loss using fake images
    z = torch.rand((batch_size, z_dim, 1, 1), device=device)
    fake_images = G(z)
    outputs = D(fake_images)
    D_loss_fake = loss(outputs.squeeze(), fake_labels)

    # Optimize discriminator
    D_loss = D_loss_real + D_loss_fake
    D.zero_grad()
    # if D_loss > 0.01:  # Dont make discriminator too strong.
    D_loss.backward()
    D_optimizer.step()
    reset_grad()

    # Train generator
    # Compute loss with fake images
    z = torch.rand((batch_size, z_dim, 1, 1), device=device)
    fake_images = G(z)
    outputs = D(fake_images)
    G_loss = loss(outputs.squeeze(), real_labels)

    # Optimize generator
    G_loss.backward()
    G_optimizer.step()
    reset_grad()
    return D_loss, G_loss


def train_wgan_iter(D, D_optimizer,
                    G, G_optimizer,
                    device, generator, mb_size, reset_grad, z_dim, num_critic_iter=5):
    D_loss = 0
    assert num_critic_iter >= 5
    for i in range(num_critic_iter):
        # Sample data
        z = torch.randn(mb_size, z_dim, 1, 1, device=device)
        x, _ = next(generator)
        x = x.to(device)

        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        D_real = D(x)
        D_fake = D(G_sample)

        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

        D_loss.backward()
        D_optimizer.step()

        # Weight clipping
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

        reset_grad()

    # Generator forward-loss-backward-update
    z = torch.randn(mb_size, z_dim, 1, 1).to(device)
    G_sample = G(z)
    D_fake = D(G_sample)
    G_loss = -torch.mean(D_fake)
    G_loss.backward()
    G_optimizer.step()
    reset_grad()
    return D_loss, G_loss


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

    loss_args = dict(
        name='default'
    )

    if opt.mode == 'dcgan':
        generator_optimizer_args = dict(
            name='torch.optim.Adam',
            lr=5e-5,
            betas=(0.5, 0.999)
        )

        discriminator_optimizer_args = dict(
            name='torch.optim.SGD',
            lr=5e-5
        )
    elif opt.mode == 'wgan-wp':
        generator_optimizer_args = dict(
            name='torch.optim.RMSprop',
            lr=0.00005
        )

        discriminator_optimizer_args = dict(
            name='torch.optim.RMSprop',
            lr=0.00005
        )

    dataset_inception_metric = dataset_specific_config['inception_metric']
    dataset_fid_metric = dataset_specific_config['fid_metric']
    callbacks_args = [
        dict(InceptionMetric=dict(sample_size=train_data_args['batch_size'],
                                  total_samples=dataset_inception_metric['evaluation_size'],
                                  classifier_model_args=dict(
                                      # Use Enums here
                                      model_arch_name=dataset_inception_metric['evaluation_arch_name'],
                                      model_weights_path=dataset_inception_metric['evaluation_classifier_weights'],
                                      model_constructor_args=dict(
                                          input_size=dataset_args['name'].value['image_size'],
                                          number_of_input_channels=dataset_args['name'].value['channels'],
                                          number_of_classes=dataset_args['name'].value['labels_count'],
                                      )
                                  ),
                                  transform=dict(mean=dataset_inception_metric['evaluation_classifier_mean'],
                                                 std=dataset_inception_metric['evaluation_classifier_std']),
                                  mode='gan')
             ),
        dict(FrechetMetric=dict(sample_size=train_data_args['batch_size'],
                                total_samples=dataset_fid_metric['evaluation_size'],
                                classifier_model_args=dict(
                                    # Use Enums here
                                    model_arch_name=dataset_fid_metric['evaluation_arch_name'],
                                    model_weights_path=dataset_fid_metric['evaluation_classifier_weights'],
                                    model_constructor_args=dict()
                                ),
                                classifier_model_layer=dataset_fid_metric['classifier_model_layer'],
                                transform=dict(mean=dataset_fid_metric['evaluation_classifier_mean'],
                                               std=dataset_fid_metric['evaluation_classifier_std']), )
             ),
        dict(GanSampler=dict(
            write_to_tensorboard=True,
            write_to_disk=True,
            num_samples=16, )
        )
    ]

    arguments = dict(
        mode=opt.mode,
        dataset_args=dataset_args,
        train_data_args=train_data_args,
        val_data_args=val_data_args,
        generator_model_args=generator_model_args,
        discriminator_model_args=discriminator_model_args,
        loss_args=loss_args,
        generator_optimizer_args=generator_optimizer_args,
        discriminator_optimizer_args=discriminator_optimizer_args,
        callbacks_args=callbacks_args,
        outdir=opt.output_dir,
        num_iterations=opt.num_iterations,
        random_seed=dataset_specific_config.get('random_seed', random.randint(0, 100))
    )

    try:
        train_gan(arguments)
    except Exception as e:
        logger.exception("Exception caught from objective function")


if __name__ == '__main__':
    main()
