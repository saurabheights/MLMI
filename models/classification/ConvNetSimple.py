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
        # 1 or 3 - input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(number_of_input_channels, 8, 3)

        self.conv2 = nn.Conv2d(8, 16, 3)

        self.conv3 = nn.Conv2d(16, 64, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, number_of_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.avgpool(F.relu(self.conv3(x)))
        x = x.view(-1, num_flat_features(x))
        x = self.fc(x)

        return x


if __name__ == '__main__':
    writer = SummaryWriter()
    dummy_input = (torch.rand(1, 1, 28, 28),)
    model = ConvNetSimple(input_size=(28, 28), number_of_input_channels=1)
    writer.add_graph(model, dummy_input)
    writer.close()
