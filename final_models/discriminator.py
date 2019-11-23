import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_shape, num_filters_init):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], num_filters_init, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_filters_init, 2 * num_filters_init, 4, 2, 1),
            nn.BatchNorm2d(2 * num_filters_init),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * num_filters_init, 4 * num_filters_init, 4, 2, 1),
            nn.BatchNorm2d(4 * num_filters_init),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * num_filters_init, 8 * num_filters_init, 4, 2, 1),
            nn.BatchNorm2d(8 * num_filters_init),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * num_filters_init, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network.forward(x)