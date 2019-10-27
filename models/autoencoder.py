import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import FakeData
import os
from collections import OrderedDict

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = FakeData()

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class ResBlock(nn.Module):
    def __init__(self, inputChannels, outputChannels, filterSize):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, filterSize),
            nn.ReLU(),
            nn.Conv2d(inputChannels, outputChannels, filterSize)
        )

    def forward(self, x):
        return x + self.block(x)


class ResAutoencoder(nn.Module):
    def __init__(self):
        super(ResAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2),
        self.res1 = ResBlock(8, 8, 5),
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2),
        self.res2 = ResBlock(16, 16, 5),
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2),
        self.res3 = ResBlock(32, 32, 3)

        self.encoder = nn.Sequential(
                        OrderedDict([('conv1', self.conv1),
                                     (self.res1, self.res1, self.conv2, self.res2, self.conv3, self.res3]))

        self.deconv1 = nn.ConvTranspose2d(8, 3, 4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.decoder = [self.res3, self.deconv3, self.res2, self.res2, self.deconv2, self.res1, self.res1, self.deconv1]


    def forward(self, x):
        for m in en

'''
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''

model = ResAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')