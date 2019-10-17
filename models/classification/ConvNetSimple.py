import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.utils import num_flat_features


class ConvNetSimple(nn.Module):
    """ A simple convolution net to test classification performance on CIFAR10. """

    def __init__(self,
                 input_size=(32, 32),
                 number_of_input_channels=3,
                 number_of_classes=10):
        super(ConvNetSimple, self).__init__()
        # 1 or 3 - input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(number_of_input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Compute output size of previous convolution layer for self.fc1
        conv2_output_size = [((size - self.conv1.kernel_size[i] + 1) / 2 - self.conv2.kernel_size[i] + 1) / 2
                             for i, size in enumerate(input_size)]

        if any([not float(size).is_integer() for size in conv2_output_size]):
            raise ValueError('Input Size not supported.')

        self.fc1 = nn.Linear(16 * int(conv2_output_size[0]) * int(conv2_output_size[1]), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, number_of_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # Note: Flatten features before passing to FC layers
        x = x.view(-1, num_flat_features(x))

        # Note: Relu to introduce non-linearity in FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Note: No relu to avoid only positive values.
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    writer = SummaryWriter()
    dummy_input = (torch.rand(1, 1, 28, 28),)
    model = ConvNetSimple(input_size=(28, 28), number_of_input_channels=1)
    writer.add_graph(model, dummy_input)
    writer.close()
