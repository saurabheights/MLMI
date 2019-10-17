from enum import Enum

from dataset.CIFAR10 import CIFAR10
from dataset.MNIST import MNIST


class SupportedDataset(Enum):
    CIFAR10_Enum = dict(
        dataloader=CIFAR10,
        image_size=(32, 32),  # Used for model FC layer.
        channels=3,
        training_size=50000,
        labels_count=10
    )

    MNIST_Enum = dict(
        dataloader=MNIST,
        image_size=(28, 28),  # Used for model FC layer.
        channels=1,
        training_size=60000,
        labels_count=10
    )


MAP_DATASET_TO_ENUM = dict(
    CIFAR10=SupportedDataset.CIFAR10_Enum,
    MNIST=SupportedDataset.MNIST_Enum,
)


def create_dataset(dataset_args, train_data_args, val_data_args):
    if dataset_args['name'] not in SupportedDataset:
        raise ValueError('Unsupported Dataset')

    return dataset_args['name'].value['dataloader'](dataset_args, train_data_args, val_data_args)
