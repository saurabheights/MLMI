import math
from pprint import pprint

import numpy
import torch
import torchvision

from dataset.BaseDataset import BaseDataset


class CIFAR10(BaseDataset):

    def __init__(self,
                 dataset_args,
                 train_data_args,
                 val_data_args):
        super(CIFAR10, self).__init__(train_data_args, val_data_args)

        dataset_dir = './data/' + self.__class__.__name__

        # ToDo - Fix mean and std.
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        self.__normalize_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

        # Normalization transform does (x - mean) / std
        # To denormalize use mean* = (-mean/std) and std* = (1/std)
        self.denormalization_transform = torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2))

        self.original_training_set = torchvision.datasets.CIFAR10(root=dataset_dir,
                                                                  train=True,
                                                                  download=True,
                                                                  transform=self.__normalize_transform)

        # Split train data into training and cross validation dataset using 9:1 split ration
        split_ratio = 0.9
        self.trainset, self.validationset = self._uniform_train_val_split(split_ratio)
        self.full_training_set = self.trainset

        if dataset_args.get('training_subset_percentage'):
            training_subset_percentage = dataset_args['training_subset_percentage']
            self.trainset = self.get_uniform_subset_by_percentage(training_subset_percentage)

        self.testset = torchvision.datasets.CIFAR10(root=dataset_dir,
                                                    train=False,
                                                    download=True,
                                                    transform=self.__normalize_transform)

    def debug(self):
        # get some random training images
        data_iter = iter(self.train_dataloader)
        images, labels = data_iter.next()

        # show images
        super(CIFAR10, self).imshow(torchvision.utils.make_grid(images))

        # print labels
        pprint(' '.join('%s' % self.classes[labels[j]] for j in range(len(images))))

    def denormalize(self, x):
        return self.denormalization_transform(x)

    @property
    def classes(self):
        return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
