import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, filter_size):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, filter_size, stride=1, padding=2),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(input_channels, output_channels, filter_size, stride=1, padding=2),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        return x + self.block(x)

class DeblurrerBN(nn.Module):
    def __init__(self):
        features = 8
        super(DeblurrerBN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, features, 5, stride=1, padding=2),
            nn.BatchNorm2d(features),
            ResBlock(features, features, 5),
            ResBlock(features, features, 5),
            ResBlock(features, features ,5),
            ResBlock(features, features, 5),
            ResBlock(features, features, 5),
            ResBlock(features, features ,5),
            nn.Conv2d(features, 3, 5, stride=1, padding=2),
            nn.BatchNorm2d(3),
        )

    def forward(self, x):
        return self.network(x)