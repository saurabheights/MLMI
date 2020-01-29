"""
Creator: Saurabh Khanduja
This file computes FID.
To evaluate generated dataset, it accepts real dataset and generator model.
"""
import os
import sys
import time

import torch
import numpy as np
from tqdm import tqdm, trange

from callbacks.callbacks import Callbacks
from dataset.BaseDataset import BaseDataset
from dataset.factory import create_dataset, SupportedDataset
from models.classification.ConvNetSimple import ConvNetSimple
from utils import logger
from scipy import linalg
from utils.tensorboard_writer import initialize_tensorboard


class FrechetInceptionScoreCallback(Callbacks):
    def __init__(self,
                 outdir,
                 device,
                 batch_size,
                 classifier,
                 total_samples=None,
                 transform=None,
                 dataset: BaseDataset = None):
        """

        :param classifier:  Modified model, only to the layer which features gives the feature values.
        :param outdir:
        :param device:
        :param batch_size:
        :param total_samples:
        :param transform:
        :param dataset:
        """
        super().__init__()
        self.outdir = outdir
        self.device = device
        self.classifier = classifier.to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.transform = transform if transform else dataset.get_normalize_transform()

        os.makedirs(self.outdir, exist_ok=True)
        self.tensorboard_writer = initialize_tensorboard(self.outdir)
        self.best_frechet_score = sys.float_info.max
        self.mu1, self.sigma1 = None, None

    def on_nth_iteration(self, iteration):
        """
        :param iteration:
        :return: None
        """
        with torch.no_grad():
            frechet_score = self.compute_frechet_score()
        logger.info(f'Frechet score at iteration {iteration} is {frechet_score}')
        if frechet_score < self.best_frechet_score:
            torch.save(self.model[0].state_dict(),
                       os.path.join(self.outdir, f'./G-iter_{iteration}_frechet_score_{frechet_score}.pth'))
            torch.save(self.model[1].state_dict(),
                       os.path.join(self.outdir, f'./D-iter_{iteration}_frechet_score_{frechet_score}.pth'))
            self.best_frechet_score = frechet_score

        self.tensorboard_writer.save_scalar('FID', frechet_score, iteration)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, 'Covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            logger.info(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * np.trace(covmean))

    def compute_frechet_score(self):
        if self.mu1 is None:
            # Compute Activations of specific layer of model using samples from self.dataset.test_dataloader
            self.classifier.normalize_input=False
            self.classifier.eval()
            real_activations = []
            for images, labels in tqdm(self.dataset.test_dataloader):
                images = images.repeat(1, 3, 1, 1)
                activation = self.classifier(images.to(self.device))
                activation = activation[0].squeeze().detach().cpu().numpy()
                real_activations.append(activation)
            real_activations = np.concatenate(real_activations, axis=0)
            self.mu1 = np.mean(real_activations, axis=0)
            self.sigma1 = np.cov(real_activations, rowvar=False)

        # Compute Activations of specific layer of model using generated samples from generated samples
        self.classifier.normalize_input = False
        generator = self.model[0]
        generator.eval()
        fake_activations = []
        num_batch = int(self.total_samples / self.batch_size)
        for _ in trange(num_batch):
            noise = torch.randn(self.batch_size, generator.z_dim, 1, 1, device=self.device)
            images = generator(noise).detach()
            images = images.repeat(1, 3, 1, 1)
            activation = self.classifier(self.transform(images).to(self.device))
            activation = activation[0].squeeze().detach().cpu().numpy()
            fake_activations.append(activation)
        fake_activations = np.concatenate(fake_activations, axis=0)
        mu2 = np.mean(fake_activations, axis=0)
        sigma2 = np.cov(fake_activations, rowvar=False)
        return self.calculate_frechet_distance(self.mu1, self.sigma1, mu2, sigma2)


if __name__ == '__main__':
    # Model trained on MNIST
    eval_model = ConvNetSimple(input_size=SupportedDataset.MNIST_Enum.value['image_size'],
                               number_of_input_channels=1,
                               number_of_classes=10)

    # Load MNIST dataset
    dataset_args = dict(
        name=SupportedDataset.MNIST_Enum,
        training_subset_percentage=1.0,
        mean=(0.5,),
        std=(0.5,),
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

    dataset: BaseDataset = create_dataset(dataset_args,
                                          train_data_args,
                                          val_data_args)

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    eval_model.load_state_dict(torch.load(
        './logs/2019-12-22T02:24:08.329024_mode_classification_model_ConvNetSimple_dataset_MNIST_subset_1.0_bs_64_name_Adam_lr_0.001/epoch_0032-model-val_accuracy_99.11754911754912.pth'))
    eval_model = torch.nn.Sequential(*list(eval_model.children())[:4])

    start = time.time()
    outdir = './logs/fretchet_score'
    transform = None

    callback = FrechetInceptionScoreCallback(outdir='./logs/frechet_score',
                                             device=device,
                                             classifier=eval_model,
                                             dataset=dataset,
                                             total_samples=10000,
                                             batch_size=64
                                             )
    print('\nInception Score of real dataset is ', callback.compute_frechet_score())
    end = time.time()
    print(f'Time taken = {end - start}')
