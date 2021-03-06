import math
import random
from pprint import pprint

import numpy
import torch
import torchvision
from torch.utils.data import DataLoader

from dataset.BaseDataset import BaseDataset
from utils import logger
import skimage


class GaussianContaminationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, contamination_args):
        self.dataset = dataset
        self.contamination_args = contamination_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        noisy = False
        if random.uniform(0, 1) < self.contamination_args['contamination_percentage']:
            x = skimage.util.random_noise(x,  # Adds noise, not rerturns noise.
                                          mode='gaussian',
                                          var=self.contamination_args['noise_std']**2)
            x = torch.from_numpy(x).type(torch.FloatTensor)
            noisy = True
        return x, y, noisy



class MNIST(BaseDataset):
    """
    Loads the train/validation/test set.
    Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
    for A-J respectively.
    Split of Train is done with 9:1 to generate training and validation dataset
    """

    def __init__(self,
                 dataset_args,
                 train_data_args,
                 val_data_args):
        super(MNIST, self).__init__(train_data_args, val_data_args)

        dataset_dir = './data/' + self.__class__.__name__

        self.mean = dataset_args.get('mean', (0.5,))
        self.std = dataset_args.get('std', (0.5,))

        logger.info(f'Mean and Std used: {self.mean}, {self.std}')

        self.demean = [-m / s for m, s in zip(self.mean, self.std)]
        self.destd = [1 / s for s in self.std]

        self.normalize_transform = self.get_normalize_transform()

        # Normalization transform does (x - mean) / std
        # To denormalize use mean* = (-mean/std) and std* = (1/std)
        self.denormalization_transform = torchvision.transforms.Normalize(self.demean, self.destd)

        self.original_training_set = torchvision.datasets.MNIST(root=dataset_dir,
                                                                train=True,
                                                                download=True,
                                                                transform=self.normalize_transform)

        # Split train data into training and cross validation dataset using 9:1 split ration
        split_ratio = 0.9
        self.trainset, self.validationset = self._uniform_train_val_split(split_ratio)
        self.full_training_set = self.trainset

        if dataset_args.get('training_subset_percentage'):
            training_subset_percentage = dataset_args['training_subset_percentage']
            self.trainset = self.get_uniform_subset_by_percentage(training_subset_percentage)

        self.testset = torchvision.datasets.MNIST(root=dataset_dir,
                                                  train=False,
                                                  download=True,
                                                  transform=self.normalize_transform)

        if dataset_args.get('contamination_args'):
            self.trainset = GaussianContaminationDataset(self.trainset, dataset_args['contamination_args'])

    def debug(self):
        # get some random training images
        data_iter = iter(self.train_dataloader)
        samples = data_iter.next()

        if len(samples) == 2:
            images, labels = samples
            is_noisy = None
        else:
            images, labels, is_noisy = samples

        # show images
        super(MNIST, self).imshow(torchvision.utils.make_grid(images))

        # print labels
        pprint(' '.join('%s' % self.classes[labels[j]] for j in range(len(images))))
        if is_noisy is not None:
            pprint(' '.join('%s' % is_noisy[j].item() for j in range(len(images))))

    def get_normalize_transform(self):
        normalize_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Scale(64),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(self.mean, self.std)])
        return normalize_transform

    def denormalize(self, x):
        return self.denormalization_transform(x)

    @property
    def classes(self):
        return [str(i) for i in range(10)]

    def get_uniform_subset(self, samples_per_label):
        targets = self.original_training_set.targets
        if type(targets) == list:
            targets = numpy.array(targets)
            labels = targets
        elif type(targets) == torch.tensor or type(targets) == torch.Tensor:
            labels = targets.numpy()

        indices = []
        for i in range(len(self.classes)):
            label_indices = numpy.argwhere(labels == i)
            label_indices = label_indices[:samples_per_label]
            indices.extend(label_indices.squeeze().tolist())

        uniform_subset = torch.utils.data.Subset(self.original_training_set, indices)
        return uniform_subset

    def get_uniform_subset_by_percentage(self, training_subset_percentage):
        labels = self.get_subset_targets(self.full_training_set)
        indices = []
        for label_value in range(len(self.classes)):
            label_indices = numpy.argwhere(labels == label_value)
            number_of_labels = int(math.floor(training_subset_percentage * len(label_indices) + 0.5))
            label_indices = label_indices[:number_of_labels]
            indices.extend(label_indices.squeeze().tolist())

        uniform_subset = torch.utils.data.Subset(self.full_training_set, indices)
        return uniform_subset
