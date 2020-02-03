import math
from pprint import pprint

import numpy
import torch
import torchvision
import csv
from os import listdir
import os
from dataset.BaseDataset import BaseDataset
import shutil


class ISIC(BaseDataset):

    def __init__(self,
                 dataset_args,
                 train_data_args,
                 val_data_args):
        super(ISIC, self).__init__(train_data_args, val_data_args)
        dataset_dir = 'data/' + self.__class__.__name__
        #ToDo - This is CIFAR code copy pasted. Fix it.
        # ToDo - Fix mean and std.
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        ##mean = [0.49139968, 0.48215827, 0.44653124]
        # mean = [0.4914, 0.4822, 0.4465]
        # # dsStd = [0.24703233, 0.24348505, 0.26158768]
        # dsStd = [0.2023, 0.1994, 0.2010]

        # std = (0.2023, 0.1994, 0.2010)
        self.__normalize_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((64,64)),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

        # Normalization transform does (x - mean) / std
        # To denormalize use mean* = (-mean/std) and std* = (1/std)
        self.denormalization_transform = torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2))

        self.original_training_set = torchvision.datasets.ImageFolder(root=dataset_dir,
                                                                  transform=self.__normalize_transform)

        # Split train data into training and cross validation dataset using 9:1 split ration
        split_ratio = 0.8
        self.trainset, self.validationset, self.testset = self._uniform_train_val_test_split(split_ratio, 0.1)
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
        images, labels = data_iter.next()
        # import pdb;pdb.set_trace()

        # show images
        # super(ISIC, self).imshow(torchvision.utils.make_grid(images))

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
        label_path = 'home/student/sumant/MLMI/data/' + self.__class__.__name__ +\
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
                    fileName = dataset_dir + row[0] + ".jpg"
                    path_label = ""
                    if os.path.exists(fileName):
                        for i in range(1,8):
                            if (row[i] == '1.0'):
                                path_label = self.classes[i-1]
                                break
                        path_label = dataset_dir + path_label
                        if os.path.exists(path_label) == False:
                            os.mkdir(path_label)
                        shutil.move(fileName, path_label)
                    line_count += 1
                    print(fileName)
                    print("copied to")
                    print(path_label)


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
    # dataset.csv_loader()

if __name__== "__main__":
  main()
