from PIL import Image
from torch import nn
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
import sys
from LFWC import LFWC
import matplotlib.pyplot as plt

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
        features = 8
        super(Deblurrer, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, features, 5, stride=1, padding=2),
            ResBlock(features, features, 5),
            ResBlock(features, features, 5),
            ResBlock(features, features ,5),
            ResBlock(features, features, 5),
            ResBlock(features, features, 5),
            ResBlock(features, features ,5),
            nn.Conv2d(features, 3, 5, stride=1, padding=2),
        )

    def forward(self, x):
        return self.network(x)


def run_model(model_path):
    model = Deblurrer()
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    dataset = LFWC(["../data/train/faces_blurred"], "../data/train/faces")
    #dataset = FakeData(size=1000, image_size=(3, 128, 128), transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for data in data_loader:
        blurred_img = Variable(data['blurred'])
        nonblurred = Variable(data['nonblurred'])
        #im = Image.open(image_path)
        #transform = transforms.ToTensor()
        transformback = transforms.ToPILImage()
        plt.imshow(transformback(blurred_img[0]))
        plt.title('Blurred')
        plt.show()
        plt.imshow(transformback(nonblurred[0]))
        plt.title('Non Blurred')
        plt.show()


        out = model(blurred_img)
        #print(out.shape)
        outIm = transformback(out[0])

        plt.imshow(outIm)
        plt.title('Model out')
        plt.show()

if __name__ == "__main__":
    model = Deblurrer()
    learning_rate = .0001
    num_epochs = 100

    #dataset = LFWC(["../lfwcrop_color/faces_blurred", "../lfwcrop_color/faces_pixelated"], "../lfwcrop_color/faces")
    dataset = LFWC(["../data/train/faces_blurred"], "../data/train/faces")

    #dataset = FakeData(size=1000, image_size=(3, 128, 128), transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5,amsgrad=True)
    criterion = nn.MSELoss()

    while(True):
        try:
            for epoch in range(100):
                for data in data_loader:
                    blurred_img = Variable(data['blurred'])
                    nonblurred_img = Variable(data['nonblurred'])

                    # ===================forward=====================
                    output = model(blurred_img)
                    loss = criterion(output, nonblurred_img)
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # ===================log========================
                print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))
        except KeyboardInterrupt:
            torch.save(model.state_dict(),'semanticmodelinterrupt')
            sys.exit()

        break


    torch.save(model.state_dict(), 'semanticmodel')
