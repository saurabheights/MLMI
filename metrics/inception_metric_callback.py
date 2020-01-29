"""
Creator: Saurabh Khanduja
Inception Score is evaluated either on real dataset or generated dataset.
This file provides two ways to do either.

To evaluate generated dataset, it accepts dataset and generator model.
To evaluate real dataset, pass generator model as None in set_model. See callbacks.callbacks.Callbacks.set_model

Path to trained model on training dataset are hardcoded for now.
"""
import logging
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from callbacks.callbacks import Callbacks
from dataset.BaseDataset import BaseDataset
from dataset.factory import create_dataset, SupportedDataset
from models.classification.ConvNetSimple import ConvNetSimple
from utils import logger
from utils.tensorboard_writer import initialize_tensorboard


class InceptionScoreCallback(Callbacks):
    def __init__(self,
                 classifier,
                 outdir,
                 mode='gan',
                 device=None,
                 batch_size=None,
                 total_samples=None,
                 transform=None,
                 dataset: BaseDataset = None):
        super().__init__()
        self.outdir = outdir
        self.device = device
        self.classifier = classifier.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.mode = mode
        self.transform = transform if transform else dataset.get_normalize_transform()
        assert mode in ['classifier', 'gan']

        os.makedirs(self.outdir, exist_ok=True)
        self.tensorboard_writer = initialize_tensorboard(self.outdir)
        self.best_inception_score=1

    def on_nth_iteration(self, iteration):
        """
        :param iteration:
        :return: None
        """
        inception_score = self.compute_inception_score()
        logger.info(f'Inception score at iteration {iteration} is {inception_score}')
        if self.mode != 'classifier' and inception_score > self.best_inception_score:
            torch.save(self.model[0].state_dict(), os.path.join(self.outdir, f'./G-iter_{iteration}_inception_score_{inception_score}.pth'))
            torch.save(self.model[1].state_dict(), os.path.join(self.outdir, f'./D-iter_{iteration}_inception_score_{inception_score}.pth'))
            self.best_inception_score = inception_score

        self.tensorboard_writer.save_scalar('InceptionScore', inception_score, iteration)

    def metric_ops(self):
        """
        :param generator: The generator which needs to be evaluated.
        :return: The Classifier Score (scalar quantity)
        """
        generator = self.model[0]
        generator.eval()
        noise = torch.randn(self.batch_size, generator.z_dim, 1, 1, device=self.device)
        img = generator(noise).detach()
        score = self.calculate_score(self.classifier(self.transform(img).to(self.device)))
        return score

    def calculate_score(self, x):
        """
        Computes the Inception Score for the Input Logit values.
        :param x: (torch.Tensor) : Image in tensor format
        :return: The Inception Score.
        """

        p = F.softmax(x, dim=1)
        q = torch.mean(p, dim=0)
        kl = torch.sum(p * (F.log_softmax(x, dim=1) - torch.log(q)), dim=1)
        return torch.exp(torch.mean(kl)).data

    def compute_inception_score(self):
        score = 0
        self.classifier.eval()
        with torch.no_grad():
            if self.mode == 'classifier':
                dataloader = self.dataset.test_dataloader
                num_batch = len(dataloader)
                for images, labels in tqdm(self.dataset.test_dataloader):
                    batch_inception_score = self.calculate_score(self.classifier(images.to(self.device)))
                    score += batch_inception_score
            else:
                num_batch = int(self.total_samples / self.batch_size)
                for _ in range(num_batch):
                    score += self.metric_ops()
        return score / num_batch  # ToDo Size of test dataset may not be divisible by BatchSize.


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

    eval_model.load_state_dict(torch.load('./logs/2019-12-22T02:24:08.329024_mode_classification_model_ConvNetSimple_dataset_MNIST_subset_1.0_bs_64_name_Adam_lr_0.001/epoch_0032-model-val_accuracy_99.11754911754912.pth'))
    outdir = './logs/2019-12-22T02:24:08.329024_mode_classification_model_ConvNetSimple_dataset_MNIST_subset_1.0_bs_64_name_Adam_lr_0.001/'
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    logger.init(outdir, logging.INFO)
    start = time.time()
    callback = InceptionScoreCallback(eval_model, device=device,
                                      dataset=dataset, mode='classifier', outdir=outdir)
    logger.info(f'Inception Score of real dataset is {callback.compute_inception_score()}')
    end = time.time()
    logger.info(f'Time taken = {end - start}')
