from PIL import Image
from torch import nn
import cv2
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
import sys
from LFWC import LFWC
import matplotlib.pyplot as plt
from metrics import psnr,ssim1
from vgg_face import return_loaded_model

class Clamper(nn.Module):
    def __init__(self, clamp_lower=False):
        super(Clamper, self).__init__()
        self.clamp_lower = clamp_lower

    def forward(self, x):
        if self.clamp_lower:
            return x.clamp(min=0,max=255)
        return x.clamp(max=255)

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, filter_size):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, filter_size, stride=1, padding=2),
	    Clamper(),
            nn.ReLU(True),
            nn.Conv2d(input_channels, output_channels, filter_size, stride=1, padding=2),
	    Clamper()
        )

    def forward(self, x):
        return x + self.block(x)

class Deblurrer(nn.Module):
    def __init__(self):
        features = 8
        super(Deblurrer, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, features, 5, stride=1, padding=2),
            Clamper(),
            ResBlock(features, features, 5),
            ResBlock(features, features, 5),
            ResBlock(features, features ,5),
            ResBlock(features, features, 5),
            ResBlock(features, features, 5),
            ResBlock(features, features ,5),
            nn.Conv2d(features, 3, 5, stride=1, padding=2),
 	    Clamper(True)
        )

    def forward(self, x):
        return self.network(x)
def perceptual_loss(vgg_net,output,nonblurred_img):
    vgg_net.eval()
    f = nn.MSELoss()
    l = 0
    output1 = vgg_net(output)
    output2 = vgg_net(nonblurred_img)
    for (a,b) in zip(output1,output2):
        l += f(a,b)
    return l



def evaluate_metrics(model_path):
    model = Deblurrer()
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    dataset = LFWC(["../data/test/faces_blurred"], "../data/test/faces")
    #dataset = FakeData(size=1000, image_size=(3, 128, 128), transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    count = 0
    avg0 = 0
    avg1 = 0
    avgs = 0
    avgs1 = 0
    for data in data_loader:
        blurred_img = Variable(data['blurred'])
        nonblurred = Variable(data['nonblurred'])
        #im = Image.open(image_path)
        #transform = transforms.ToTensor()
        transformback = transforms.ToPILImage()


        out = model(blurred_img)
        #print(out.shape)
        outIm = transformback(out[0])
        nonblurred = transformback(nonblurred[0])
        blurred = transformback(blurred_img[0])
        ps = psnr(outIm,nonblurred)
        avg0 += ps
        ps1 = psnr(blurred,nonblurred)
        avg1 += ps1
        similarity = ssim1(outIm,nonblurred)
        avgs += similarity
        sim1 = ssim1(blurred,nonblurred)
        avgs1 += sim1
        count += 1
    avg0 /= count
    avg1 /= count
    avgs /= count
    avgs1 /= count
    print(avg0)
    print(avg1)
    print(avgs)
    print(avgs1)


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
    vgg_net = return_loaded_model()
    '''
    im = cv2.imread("../vgg_face_torch/21172.ppm")
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 64, 64).double()
    im2 = cv2.imread("../vgg_face_torch/21172_2.ppm")
    im2 = torch.Tensor(im2).permute(2, 0, 1).view(1, 3, 64, 64).double()
    print(perceptual_loss(vgg_net,im,im2))'''

    #dataset = FakeData(size=1000, image_size=(3, 128, 128), transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5,amsgrad=True)
    criterion = nn.MSELoss()

    while(True):
        try:
            for epoch in range(100):
                for data in data_loader:
                    blurred_img = Variable(data['blurred']).cuda()
                    nonblurred_img = Variable(data['nonblurred']).cuda()

                    # ===================forward=====================
                    output = model(blurred_img)
                    loss = criterion(output, nonblurred_img) + perceptual_loss(vgg_net,output,nonblurred_img)
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