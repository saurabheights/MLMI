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

parser.add_argument('--mode', type=str, default='dcgan', choices=['dcgan', 'wgan-wp', 'wgan-noise-adversarial'],
                    help='Optional - To train dcgan or wgan or wgan with noise removal. Default value dcgan.')

parser.add_argument('--num_iterations', type=int, default=10000,
                    help='Optional - Number of iterations for training gan. Default value 10000.')

parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CELEBA'],
                    help='Optional - The dataset to choose')

parser.add_argument('--generator_model_path', type=str, required=False, default=None,
                    help='Optional - Path for generator pretrained weights.')

parser.add_argument('--output_dir', type=str, required=False, default='./logs/',
                    help='Optional - Where to create output directory path. Default ./logs.')

parser.add_argument('--gpu', type=int, default=0,
                    help='Optional - If gpu available, which one to use. Default value 0.')

#Parse Contamination arguments
parser.add_argument('--enable_contamination', action='store_true', help='Optional - ToDo')
parser.add_argument('--contamination_percentage', type=float, default=0.1, help='Optional - ToDo')
parser.add_argument('--contamination_std', type=float, default=0.25, help='Optional - ToDo')

parser.add_argument('--contamination_loss_weight', type=float, default=1.0, help='Optional - ToDo')

opt = parser.parse_args()

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def infinite_train_gen(dataloader):
    def f():
        while True:
            for samples in dataloader:
                yield samples

    return f()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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

    if arguments['mode'] == 'dcgan':
        G.apply(weights_init)
        D.apply(weights_init)

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
    z_dim = arguments['generator_model_args']['model_constructor_args']['nz']

    generator = infinite_train_gen(dataset.train_dataloader)
    interval_length = 10 if is_debug_mode() else 400
    num_intervals = 1 if is_debug_mode() else int(arguments['num_iterations'] / interval_length)

    global_step = 0

    # TO allocate memory required for the GPU during training and validation
    run_callbacks(callbacks,
                  model=(G, D),
                  optimizer=(G_optimizer, D_optimizer),  # To Save optimizer dict for retraining.
                  mode=CallbackMode.ON_NTH_ITERATION, iteration=global_step)
    reset_grad()

    for it in range(num_intervals):

        logger.info(f'Interval {it + 1}/{num_intervals}')

        # Set model in train mode
        G.train()
        D.train()

        t = trange(interval_length)
        for _ in t:
            if arguments['mode'] == 'dcgan':
                D_loss, G_loss = train_gan_iter(D, D_optimizer, G, G_optimizer,
                                                loss, device, generator, batch_size, reset_grad, z_dim,
                                                tensorboard_writer, global_step)
            elif arguments['mode'] == 'wgan-wp':
                D_loss, G_loss = train_wgan_iter(D, D_optimizer, G, G_optimizer, device,
                                                 generator, batch_size, reset_grad, z_dim,
                                                 tensorboard_writer, global_step)
            elif arguments['mode'] == 'wgan-noise-adversarial':
                D_loss, G_loss = train_noisy_wgan_iter(D, D_optimizer, G, G_optimizer, device,
                                                       generator, batch_size, reset_grad, z_dim,
                                                       tensorboard_writer, global_step,
                                                       contamination_loss_weight=arguments['contamination_loss_weight'])


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
                   loss, device, generator, batch_size, reset_grad, z_dim, tensorboard_writer, global_step):
    real_label = 1
    fake_label = 0

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    D.zero_grad()
    samples = next(generator)
    images = samples[0].to(device)
    batch_size = images.size(0)
    label = torch.full((batch_size,), real_label, device=device)

    output = D(images)
    errD_real = loss(output, label)
    errD_real.backward()
    D_loss_real = output.mean().item()

    # train with fake
    noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
    fake = G(noise)
    label.fill_(fake_label)
    output = D(fake.detach())
    errD_fake = loss(output, label)
    errD_fake.backward()
    D_loss = errD_real + errD_fake
    D_optimizer.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    G.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    output = D(fake)
    G_loss = loss(output, label)
    G_loss.backward()
    G_optimizer.step()

    reset_grad()

    print(f'{D_loss} / {G_loss}')
    return D_loss, G_loss


def train_wgan_iter(D, D_optimizer,
                    G, G_optimizer,
                    device, generator, mb_size, reset_grad, z_dim, tensorboard_writer, global_step, num_critic_iter=10):
    D_loss = 0
    assert num_critic_iter >= 5
    for i in range(num_critic_iter):
        # Sample data
        z = torch.randn(mb_size, z_dim, 1, 1, device=device)
        samples = next(generator)
        x = samples[0]
        x = x.to(device)

        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        D_real, _ = D(x)
        D_fake, _ = D(G_sample)

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
    D_fake, _ = D(G_sample)
    G_loss = -torch.mean(D_fake)
    G_loss.backward()
    G_optimizer.step()
    reset_grad()
    return D_loss, G_loss


def train_noisy_wgan_iter(D, D_optimizer,
                          G, G_optimizer,
                          device, generator, mb_size, reset_grad, z_dim, tensorboard_writer, global_step,
                          num_critic_iter=10, contamination_loss_weight=1):
    D_loss = 0
    assert num_critic_iter >= 5
    for i in range(num_critic_iter):
        # Sample data
        z = torch.randn(mb_size, z_dim, 1, 1, device=device)
        x, label, noisy = next(generator)
        x = x.to(device)

        clean = ~noisy

        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        D_real, D_real_clean_and_noisy = D(x)
        D_fake, D_fake_noisy = D(G_sample)

        D_real_vs_fake_loss = -(torch.mean(D_real) - torch.mean(D_fake))

        D_real_clean_mean, D_real_noisy_mean = 0, 0
        if clean.any():
            D_real_clean_mean = torch.mean(D_real_clean_and_noisy[clean])
        if noisy.any():
            D_real_noisy_mean = torch.mean(D_real_clean_and_noisy[noisy])

        # We assume all fake samples are noisy. Isn't this counterintuitive to Discriminator
        # D_noisy_vs_clean_loss = -contamination_loss_weight * (D_real_clean_mean -
        #                                                       D_real_noisy_mean -
        #                                                       torch.mean(D_fake_noisy))

        # Train only discriminator on real data
        D_noisy_vs_clean_loss = -contamination_loss_weight * (D_real_clean_mean - D_real_noisy_mean)

        D_loss = D_real_vs_fake_loss + D_noisy_vs_clean_loss
        D_loss.backward()
        D_optimizer.step()

        # Weight clipping
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

        reset_grad()

    tensorboard_writer.save_scalars(f'W_GAN_NOISE_ADVERSARIAL_Critic_Loss',
                                    {
                                        'noisy_vs_clean_loss': D_noisy_vs_clean_loss.data.cpu().item(),
                                        'real_vs_fake_loss': D_real_vs_fake_loss.data.cpu().item()},
                                    global_step)

    # Generator forward-loss-backward-update
    z = torch.randn(mb_size, z_dim, 1, 1).to(device)
    G_sample = G(z)
    D_fake, D_fake_noisy = D(G_sample)
    G_loss = -torch.mean(D_fake) - contamination_loss_weight * torch.mean(D_fake_noisy)
    G_loss.backward()
    G_optimizer.step()
    reset_grad()
    return D_real_vs_fake_loss, G_loss


def main():
    dataset_specific_configs = dict(
        MNIST=dict(
            training_batch_size=64,
            z_dim=100,
            inception_metric=dict(
                evaluation_arch_name='models.classification.ConvNetSimple.ConvNetSimple',
                evaluation_classifier_weights='./logs/2020-02-03T14:55:00.666781_train_mode_classification_model_ConvNetSimple_dataset_MNIST_bs_64_name_Adam_lr_0.001/epoch_0037-model-val_accuracy_98.45154845154845.pth',
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
        mean=(0.5,),
        std=(0.5,),
        contamination_args=dict(noise_mean = 0.0,
                                noise_std = opt.contamination_std,
                                contamination_percentage = opt.contamination_percentage) if opt.enable_contamination else None
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

    loss_args = dict(
        name='default'
    )

    if opt.mode == 'dcgan':
        generator_model_args = dict(
            # Use Enums here
            model_arch_name='models.gan.DCGAN.DcGanGenerator',
            model_weights_path=opt.generator_model_path,
            model_constructor_args=dict(
                nz=dataset_specific_config['z_dim'],
                channels=dataset_args['name'].value['channels'],
            )
        )

        discriminator_model_args = dict(
            # Use Enums here
            model_arch_name='models.gan.DCGAN.DcGanDiscriminator',
            model_weights_path=opt.generator_model_path.replace('G-', 'D-') if opt.generator_model_path else None,
            model_constructor_args=dict(
                channels=dataset_args['name'].value['channels']
            )
        )
        generator_optimizer_args = dict(
            name='torch.optim.Adam',
            lr=0.0002,
            betas=(0.5, 0.999)
        )

        discriminator_optimizer_args = dict(
            name='torch.optim.Adam',
            lr=0.0002,
            betas=(0.5, 0.999)
        )
    elif opt.mode in ['wgan-wp', 'wgan-noise-adversarial']:
        generator_model_args = dict(
            # Use Enums here
            model_arch_name='models.gan.DCGAN.Generator',
            model_weights_path=opt.generator_model_path,
            model_constructor_args=dict(
                nz=dataset_specific_config['z_dim'],
                channels=dataset_args['name'].value['channels'],
            )
        )

        discriminator_model_args = dict(
            # Use Enums here
            model_arch_name='models.gan.DCGAN.Discriminator',
            model_weights_path=opt.generator_model_path.replace('G-', 'D-') if opt.generator_model_path else None,
            model_constructor_args=dict(
                channels=dataset_args['name'].value['channels']
            )
        )

        generator_optimizer_args = dict(
            name='torch.optim.RMSprop',
            lr=0.0002
        )

        discriminator_optimizer_args = dict(
            name='torch.optim.RMSprop',
            lr=0.0002
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
        random_seed=dataset_specific_config.get('random_seed', random.randint(0, 1000)),
        contamination_loss_weight=opt.contamination_loss_weight
    )

    try:
        train_gan(arguments)
    except Exception as e:
        logger.exception("Exception caught from objective function")


if __name__ == '__main__':
    main()
