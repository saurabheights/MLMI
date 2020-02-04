import math
from pprint import pprint
from random import random

import numpy
import skimage
import torch
import torchvision
import csv
from os import listdir
import os
from dataset.BaseDataset import BaseDataset
import shutil

from torch.utils.data import Dataset
import random
from utils import logger
import numpy as np


class ISIC(BaseDataset):

    def __init__(self,
                 dataset_args,
                 train_data_args,
                 val_data_args):
        super(ISIC, self).__init__(train_data_args, val_data_args)
        dataset_dir = '/home/saosurvivor/Projects/MLMI/MLMI/data/' + self.__class__.__name__
        #ToDo - This is CIFAR code copy pasted. Fix it.
        # ToDo - Fix mean and std.
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        self.__normalize_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

        # Normalization transform does (x - mean) / std
        # To denormalize use mean* = (-mean/std) and std* = (1/std)
        self.denormalization_transform = torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2))

        self.original_training_set = torchvision.datasets.ImageFolder(root=dataset_dir,
                                                                  transform=self.__normalize_transform)
        # data_iter = iter(self.original_training_set)
        # images, labels = data_iter.next()


        self.original_training_set = NoiseAdderDataset(self.original_training_set, interval=10)

        print(type(self.original_training_set))
        #import pdb;pdb.set_trace()
        # Split train data into training and cross validation dataset using 9:1 split ration
        split_ratio = 0.9
        self.trainset, self.validationset = self._uniform_train_val_split(split_ratio)
        self.full_training_set = self.trainset

        if dataset_args.get('training_subset_percentage'):
            training_subset_percentage = dataset_args['training_subset_percentage']
            self.trainset = self.get_uniform_subset_by_percentage(training_subset_percentage)

        # self.testset = torchvision.datasets.CIFAR10(root=dataset_dir,
        #                                             train=False,
        #                                             download=True,
        #                                             transform=self.__normalize_transform)

    def debug(self):
        # get some random training images
        data_iter = iter(self.train_dataloader)
        print(type(data_iter))
        images, labels,flag = data_iter.next()

        # show images
        super(ISIC, self).imshow(torchvision.utils.make_grid(images))

        # print labels
        pprint(' '.join('%s' % self.classes[labels[j]] for j in range(len(images))))

    def denormalize(self, x):
        return self.denormalization_transform(x)

    @property
    def classes(self):
        return ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

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

    def csv_loader(self):
        label_path = '/home/saosurvivor/Projects/MLMI/MLMI/data/' + self.__class__.__name__ +\
                     "/labels/ISIC2018_Task3_Training_GroundTruth.csv"
        dataset_dir = '/home/saosurvivor/Projects/MLMI/MLMI/data/' + self.__class__.__name__
        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            line_count = 0
            for row in reader:
                if line_count==0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    print(f'row names are {", ".join(row)}')
                    fileName = dataset_dir + "/images/" + row[0] + ".jpg"
                    path_label = ""
                    if os.path.exists(fileName):
                        for i in range(1,8):
                            if (row[i] == '1.0'):
                                path_label = self.classes[i-1]
                                break
                        path_label = dataset_dir+ "/images/" + path_label
                        if os.path.exists(path_label) == False:
                            os.mkdir(path_label)
                        shutil.move(fileName, path_label)
                    line_count += 1
                    print(fileName)
                    print("copied to")
                    print(path_label)


class NoiseAdderDataset(Dataset):
    """
    Transform a dataset by registering a transform for every n input. Skip transformation by setting the transform to None.
    Take
        dataset: the `Dataset` to transform (which must be a `SegData`).
        interval: interval at which to apply the transform
        mean: the mean of the noise
        std: the standard deviation for the noise
    """

    def __init__(self, dataset,interval, mean=0,std=0.1):
        #super().__init__(dataset)
        self.ds = dataset
        self.interval=interval
        self.mean=mean
        self.std=std
        self.targets=dataset.targets


    def set_noise(self, img,id):
        if random.uniform(0, 1) < 0.2:
            img = skimage.util.random_noise(img,  # Adds noise, not rerturns noise.
                                          mode='gaussian',
                                          var=0.25**2)
            img = torch.from_numpy(img).type(torch.FloatTensor)
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



def main():

    dataset_args = dict(
        name=ISIC,
        training_subset_percentage=80,
        # For Data Inflation Study - Set to None to use full dataset
    )
    train_data_args = dict(
        batch_size=64,
        shuffle=True,
        to_train=True,
    )
    val_data_args = dict(
        batch_size=256,
        shuffle=False,
        validate_step_size=1,
    )
    dataset = ISIC(dataset_args, train_data_args, val_data_args)
    dataset.debug()

if __name__== "__main__":
  main()

