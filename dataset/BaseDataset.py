import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.sysutils import get_cores_count


class BaseDataset:

    def __init__(self, train_data_args, val_data_args):
        self.cpu_count = get_cores_count()
        self.train_data_args = train_data_args
        self.val_data_args = val_data_args

    def imshow(self, img):
        img = self.denormalize(img)  # denormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def debug(self):
        """ This method should display some sample of Dataset """
        raise NotImplementedError

    def denormalize(self, x):
        """ This method should denormalize any preprocessing done to inputs x when reading through DataLoader """
        raise NotImplementedError

    @property
    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.trainset,
                                           batch_size=self.train_data_args['batch_size'],
                                           shuffle=self.train_data_args['shuffle'],
                                           pin_memory=True,
                                           num_workers=self.cpu_count)

    @property
    def validation_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.validationset,
                                           batch_size=self.train_data_args['batch_size'],
                                           shuffle=self.train_data_args['shuffle'],
                                           pin_memory=True,
                                           num_workers=self.cpu_count)

    @property
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset,
                                           batch_size=self.val_data_args['batch_size'],
                                           shuffle=self.val_data_args['shuffle'],
                                           pin_memory=True,
                                           num_workers=self.cpu_count)

    @property
    def get_train_dataset_size(self):
        return len(self.trainset)

    @property
    def get_val_dataset_size(self):
        return len(self.validationset)

    @property
    def get_test_dataset_size(self):
        return len(self.testset)

    @property
    def classes(self):
        raise NotImplementedError

    def get_uniform_subset(self, samples_per_label):
        """
        Creates a training subset with fixed number of samples per class.
        :param samples_per_label: For each label, number of samples to choose.
        :return: The dataloader.
        """
        raise NotImplementedError

    def get_uniform_subset_dataloader(self, samples_per_label, no_grad):
        """
        Creates a dataloader of subset with fixed number of samples per class.
        :param samples_per_label: For each label, number of samples to choose.
        :param no_grad: If no_grad is true, higher batch size is used using validation_data_args
        :return: The dataloader.
        """
        uniform_subset = self.get_uniform_subset(samples_per_label)
        if no_grad:
            data_args = self.val_data_args
        else:
            data_args = self.train_data_args

        uniform_dataloader = torch.utils.data.DataLoader(uniform_subset,
                                                         batch_size=data_args['batch_size'],
                                                         shuffle=False,  # No need to shuffle
                                                         pin_memory=True,
                                                         num_workers=0)

        return uniform_dataloader

    @staticmethod
    def get_subset_targets(subset):
        targets = subset.dataset.targets
        if type(targets) == list:
            targets = numpy.array(targets)
        elif type(targets) == torch.Tensor:
            targets = targets.numpy()

        labels = targets[numpy.asarray(subset.indices)]
        return labels

    def _uniform_train_val_split(self, split_ratio):
        targets = self.original_training_set.targets
        if type(targets) == list:
            targets = numpy.array(targets)
            labels = targets
        elif type(targets) == torch.tensor or type(targets) == torch.Tensor:
            labels = targets.numpy()
        training_indices = []
        validation_indices = []
        for i in range(len(self.classes)):
            label_indices = numpy.argwhere(labels == i)
            samples_per_label = int(split_ratio * len(label_indices))
            training_label_indices = label_indices[:samples_per_label]
            validation_label_indices = label_indices[samples_per_label:]
            training_indices.extend(training_label_indices.squeeze().tolist())
            validation_indices.extend(validation_label_indices.squeeze().tolist())
            assert not set(training_label_indices.ravel().tolist()) & set(validation_label_indices.ravel().tolist())

        uniform_training_subset = torch.utils.data.Subset(self.original_training_set, training_indices)
        uniform_validation_subset = torch.utils.data.Subset(self.original_training_set, validation_indices)
        assert not set(training_indices) & set(validation_indices)
        return uniform_training_subset, uniform_validation_subset

    def get_uniform_subset_by_percentage(self, training_subset_percentage):
        """
        Creates a subset with certain %age of total number of samples per class.
        :param training_subset_percentage:
        """
        raise NotImplementedError

    def get_uniform_subset_by_percentage_dataloader(self, training_subset_percentage, no_grad):
        """
        Returns a data loader of a subset with certain %age of total number of samples per class.
        :param training_subset_percentage:
        :param no_grad: If no_grad is true, higher batch size is used using validation_data_args
        :return:
        """
        uniform_subset = self.get_uniform_subset_by_percentage(training_subset_percentage)
        if no_grad:
            data_args = self.val_data_args
        else:
            data_args = self.train_data_args

        uniform_dataloader = torch.utils.data.DataLoader(uniform_subset,
                                                         batch_size=data_args['batch_size'],
                                                         shuffle=data_args['shuffle'],
                                                         pin_memory=True,
                                                         num_workers=self.cpu_count)

        return uniform_dataloader
