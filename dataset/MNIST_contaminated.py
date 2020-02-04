import math
from pprint import pprint

import numpy
import torch
import torchvision
from torch.utils.data import DataLoader

from dataset.BaseDataset import BaseDataset
from torch.utils.data import Dataset

from utils import logger
import numpy as np


class MNISTContaminated(BaseDataset):
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
        super(MNISTContaminated, self).__init__(train_data_args, val_data_args)

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

        self.original_training_set = NoiseAdderDataset(self.original_training_set, interval=10)

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

    def debug(self):
        # get some random training images
        import pdb;pdb.set_trace()
        data_iter = iter(self.train_dataloader)
        images, labels, flag = data_iter.next()

        # show images
        super(MNISTContaminated, self).imshow(torchvision.utils.make_grid(images))

        # print labels
        pprint(' '.join('%s' % self.classes[labels[j]] for j in range(len(images))))

    def get_normalize_transform(self):
        normalize_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Scale(32),
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


class NoiseAdderDataset(Dataset):
    """
    Transform a dataset by registering a transform for every n input. Skip transformation by setting the transform to None.
    Take
        dataset: the `Dataset` to transform (which must be a `SegData`).
        interval: interval at which to apply the transform
        mean: the mean of the noise
        std: the standard deviation for the noise
    """

    def __init__(self, dataset,interval, mean=0,std=1):
        #super().__init__(dataset)
        self.ds = dataset
        self.interval=interval
        self.mean=mean
        self.std=std
        self.targets=dataset.targets


    def set_noise(self, img,id):
        if id%self.interval==0:
            noise=np.random.normal(loc=self.mean, scale=self.std, size=img.shape)
            img+=torch.tensor(img).float()
            flag=True
        else:
            flag=False
        return img,flag


    def __getitem__(self, idx):
        # extract data from inner dataset
        image, label = self.ds[idx]

        noisy_image,flag=self.set_noise(image,idx)

        return noisy_image,label,flag

    def __len__(self):
        return len(self.ds)