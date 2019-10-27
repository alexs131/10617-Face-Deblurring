from torch import nn
import torch
from torchvision.datasets import FakeData
from torch.autograd import Variable
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

from models.LFWC import LFWC


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, filter_size):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, filter_size, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(input_channels, output_channels, filter_size, stride=1, padding=2)
        )

    def forward(self, x):
        return x + self.block(x)

class Deblurrer(nn.Module):
    def __init__(self):
        super(Deblurrer, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=2),
            ResBlock(64, 64, 5),
            ResBlock(64, 64, 5),
            ResBlock(64, 64 ,5),
            ResBlock(64, 64, 5),
            ResBlock(64, 64, 5),
            ResBlock(64, 64 ,5),
            nn.Conv2d(64, 3, 5, stride=1, padding=2),
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = Deblurrer()
    learning_rate = .001
    num_epochs = 100

    dataset = LFWC("../lfwcrop_color/faces_blurred", "../lfwcrop_color/faces")

    #dataset = FakeData(size=1000, image_size=(3, 128, 128), transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for data in data_loader:
            img = Variable(data['blurred'])
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, data['nonblurred'])
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))


