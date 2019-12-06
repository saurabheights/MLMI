import argparse
import logging
import os
import random
from pprint import pformat
from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from tqdm import tqdm

from callbacks.callback_utils import generate_callbacks, run_callbacks
from callbacks.callbacks import Callbacks, CallbackMode
from dataset.BaseDataset import BaseDataset
from dataset.factory import create_dataset, MAP_DATASET_TO_ENUM
from models.utils import get_model
from optimizer.utils import create_optimizer
from utils import logger
from utils.fileutils import make_results_dir
from utils.tensorboard_writer import initialize_tensorboard

parser = argparse.ArgumentParser()

parser.add_argument('--num_iterations', type=int, default=10000,
                    help='Optional - Number of iterations for training gan. Default value 10000.')

parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CELEBA'],
                    help='Optional - The dataset to choose')

# Data Inflation Study, allows training on smaller subset of selected Dataset
parser.add_argument('--training_subset_percentage', type=float, default=1.0,
                    help='Optional - Subset of data to use for training. Default use whole dataset')

parser.add_argument('--model_weights_directory', type=str, required=False, default=None,
                    help='Optional - Directory containing Pretrained weights for generator and discriminator. '
                         'Requires testing')

parser.add_argument('--output_dir', type=str, required=False, default='./logs/',
                    help='Optional - Where to create output directory path. Default ./logs.')

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device - {device}')

    """ Load Model with weights(if available) """
    G: torch.nn.Module = get_model(arguments.get('generator_model_args')).to(device)
    D: torch.nn.Module = get_model(arguments.get('discriminator_model_args')).to(device)

    """ Create optimizer """
    G_optimizer = create_optimizer(G.parameters(), arguments['generator_optimizer_args'])
    D_optimizer = create_optimizer(D.parameters(), arguments['discriminator_optimizer_args'])

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

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    cnt = 0

    mb_size = arguments['train_data_args']['batch_size']
    z_dim = arguments['generator_model_args']['model_constructor_args']['z_dim']

    generator = infinite_train_gen(dataset.train_dataloader)
    interval_length = 1000
    num_intervals = int(arguments['num_iterations'] / interval_length)

    for it in range(num_intervals):

        logger.info(f'Interval {it+1}/{num_intervals}')

        for _ in tqdm(range(interval_length)):
            D_loss, G_loss, z = train_iter(D, D_optimizer, G, G_optimizer, device,
                                           generator, mb_size, reset_grad, z_dim)
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())


        run_callbacks(callbacks, model=(G, D), mode=CallbackMode.ON_NTH_ITERATION, iteration=it)
        reset_grad()
        print('Iter-{}; D_loss: {}; G_loss: {}'
              .format(it,
                      D_loss.data.cpu().item(),
                      G_loss.data.cpu().item()))

        samples = G(z).data.cpu().numpy()[:64]

        fig = plt.figure(figsize=(8, 8))  #ToDo Use Pytorch grid method
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')  # ToDo: Hardcoded Image Size

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.title("Fake Images")
        plt.show()
        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        real_batch = next(generator)

        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                                (1, 2, 0)))

        # ToDo - Save Model at every nth iteration


def train_iter(D, D_optimizer, G, G_optimizer, device, generator, mb_size, reset_grad, z_dim):
    for i in range(5):
        # Sample data
        z = torch.randn(mb_size, z_dim).to(device)
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

        # Housekeeping - reset gradient
        reset_grad()

    # Generator forward-loss-backward-update
    z = torch.randn(mb_size, z_dim).to(device)
    G_sample = G(z)
    D_fake = D(G_sample)
    G_loss = -torch.mean(D_fake)
    G_loss.backward()
    G_optimizer.step()
    # Housekeeping - reset gradient
    reset_grad()
    return D_loss, G_loss, z


def main():
    dataset_specific_configs = dict(
        MNIST=dict(
            training_batch_size=32,
            z_dim=10,
            latent_dim=128,
            evaluation_classifier_weights='logs/2019-12-03T22:52:33.058070_model_ConvNetSimple_dataset_MNIST_subset_1.0_bs_64_'
                                          'name_Adam_lr_0.001/epoch_0038-model-val_accuracy_99.23409923409923.pth',
            evaluation_size=10000,
            evaluation_classifier_std=(0.5, ),
            evaluation_classifier_mean=(0.5, )
        )
    )

    assert opt.dataset in dataset_specific_configs.keys()
    dataset_specific_config = dataset_specific_configs[opt.dataset]

    dataset_args = dict(
        name=MAP_DATASET_TO_ENUM[opt.dataset],
        training_subset_percentage=opt.training_subset_percentage,
        mean=(0,),
        std=(1,)
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
        model_weights_path=os.path.join(opt.model_weights_directory, 'G.pth') if opt.model_weights_directory else None,
        model_constructor_args=dict(
            latent_dim=dataset_specific_config['latent_dim'],
            z_dim=dataset_specific_config['z_dim'],
            image_size=dataset_args['name'].value['image_size'] + (dataset_args['name'].value['channels'],)
        )
    )

    discriminator_model_args = dict(
        # Use Enums here
        model_arch_name='models.gan.DCGAN.Discriminator',
        model_weights_path=os.path.join(opt.model_weights_directory, 'D.pth') if opt.model_weights_directory else None,
        model_constructor_args=dict(
            image_size=dataset_args['name'].value['image_size'] + (dataset_args['name'].value['channels'],),
            latent_dim=dataset_specific_config['latent_dim']
        )
    )

    loss_args = dict(
        name='default'
    )

    generator_optimizer_args = dict(
        name='torch.optim.RMSprop',
        lr=1e-4
    )

    discriminator_optimizer_args = dict(
        name='torch.optim.RMSprop',
        lr=1e-4
    )

    callbacks_args = [
        # dict(SampleSaver=dict(num_samples=8)),
        dict(InceptionMetric=dict(sample_size=train_data_args['batch_size'],
                                  total_samples=dataset_specific_config['evaluation_size'],
                                  classifier_model_args=dict(
                                      # Use Enums here
                                      model_arch_name='models.classification.ConvNetSimple.ConvNetSimple',
                                      model_weights_path=dataset_specific_config['evaluation_classifier_weights'],
                                      model_constructor_args=dict(
                                          input_size=dataset_args['name'].value['image_size'],
                                          number_of_input_channels=dataset_args['name'].value['channels'],
                                          number_of_classes=dataset_args['name'].value['labels_count'],
                                      )
                                  ),
                                  transform=dict(mean=dataset_specific_config['evaluation_classifier_mean'],
                                                 std=dataset_specific_config['evaluation_classifier_std']),
                                  mode='gan')
             )
    ]

    arguments = dict(
        mode='gan',
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
        random_seed=dataset_specific_config.get('random_seed', 42)
    )

    try:
        train_gan(arguments)
    except Exception as e:
        logger.exception("Exception caught from objective function")


if __name__ == '__main__':
    main()
