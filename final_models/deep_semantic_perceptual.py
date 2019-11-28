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
from discriminator import Discriminator, weights_init

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

def one_im_discrim(discrim_path, im_path):
    discriminator = Discriminator(3, 64)
    discriminator.load_state_dict(torch.load(discrim_path, map_location=torch.device('cpu')))
    discriminator.eval()

    tensor = transforms.ToTensor()
    im = torchImage.open(im_path)

    result = discriminator(tensor(Image.open(im_path))).view(-1)
    print(result.data.item())

def run_model(model_path, discrim_path):
    model = Deblurrer()
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()

    discriminator = Discriminator(3, 64)
    discriminator.load_state_dict(torch.load(discrim_path, map_location=torch.device('cpu')))
    discriminator.eval()

    dataset = LFWC(["../data/train/faces_blurred"], "../data/train/faces")
    #dataset = FakeData(size=1000, image_size=(3, 128, 128), transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for data in data_loader:
        blurred_img = Variable(data['blurred'])
        nonblurred = Variable(data['nonblurred'])

        # Should be near zero
        discrim_output_blurred = discriminator(blurred_img).view(-1).data.item()
        # Should be naer one
        discrim_output_nonblurred = discriminator(nonblurred).view(-1).data.item()

        #im = Image.open(image_path)
        #transform = transforms.ToTensor()
        transformback = transforms.ToPILImage()
        plt.imshow(transformback(blurred_img[0]))
        plt.title('Blurred, Discrim value: ' + str(discrim_output_blurred))
        plt.show()
        plt.imshow(transformback(nonblurred[0]))
        plt.title('Non Blurred, Discrim value: ' + str(discrim_output_nonblurred))
        plt.show()


        out = model(blurred_img)
        discrim_output_model = discriminator(out).view(-1).data.item()
        #print(out.shape)
        outIm = transformback(out[0])

        plt.imshow(outIm)
        plt.title('Model out, Discrim value: ' + str(discrim_output_model))
        plt.show()

if __name__ == "__main__":
    run_model("semanticmodel_gen_loss_1e3.pth", "discrim_gen_loss_1e3.pth")
    sys.exit(0)


    if torch.cuda.is_available():
        model = Deblurrer().cuda()
    else:
        model = Deblurrer()
    learning_rate = .0001
    learning_rate_discrim = .0002
    beta1 = 0.5
    num_epochs = 100
    batch_size = 8

    # Usually 5e-5 (in paper)
    gen_loss_weight = 5e-3
    mse_loss_weight = 50
    # Usually 1e-5 (in paper)
    perceptual_loss_weight = 1e-3

    num_filters_init = 64
    if torch.cuda.is_available():
        discriminator = Discriminator(3, num_filters_init).cuda()
    else:
        discriminator = Discriminator(3, num_filters_init)
    discriminator.apply(weights_init)

    #dataset = LFWC(["../lfwcrop_color/faces_blurred", "../lfwcrop_color/faces_pixelated"], "../lfwcrop_color/faces")
    dataset = LFWC(["../data/train/faces_blurred"], "../data/train/faces")
    if torch.cuda.is_available():
        vgg_net = return_loaded_model().cuda()
    else:
        vgg_net = return_loaded_model()
    '''
    im = cv2.imread("../vgg_face_torch/21172.ppm")
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 64, 64).double()
    im2 = cv2.imread("../vgg_face_torch/21172_2.ppm")
    im2 = torch.Tensor(im2).permute(2, 0, 1).view(1, 3, 64, 64).double()
    print(perceptual_loss(vgg_net,im,im2))'''

    #dataset = FakeData(size=1000, image_size=(3, 128, 128), transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5,amsgrad=True)
    mse_criterion = nn.MSELoss()

    discrim_criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
    optimizer_discrim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_discrim, betas=(beta1, .999))
    losses_per_epoch = []

    while(True):
        try:
            for epoch in range(num_epochs):
                loss_values = {
                    'deblur_total_average_loss': 0.0,
                    'deblur_average_percep_loss': 0.0,
                    'deblur_average_gen_loss': 0.0,
                    'deblur_average_mse_loss': 0.0,
                    'discrim_average_loss': 0.0,
                    'discrim_average_loss_on_real': 0.0,
                    'discrim_average_loss_on_deblurred': 0.0
                }
                for data in data_loader:
                    if torch.cuda.is_available():
                        blurred_img = Variable(data['blurred']).cuda()
                        nonblurred_img = Variable(data['nonblurred']).cuda()
                    else:
                        blurred_img = Variable(data['blurred'])
                        nonblurred_img = Variable(data['nonblurred'])

                    output = model(blurred_img)

                    # ==================Train Discriminator=================
                    discriminator.zero_grad()
                    # Pass through real inputs
                    output_discrim = discriminator(nonblurred_img).view(-1)
                    # Get loss
                    if torch.cuda.is_available():
                        labels = Variable(torch.full(output_discrim.shape, real_label)).cuda()
                    else:
                        labels = Variable(torch.full(output_discrim.shape, real_label))
                    discrim_error_real = discrim_criterion(output_discrim, labels)
                    del labels

                    # Accumulate grads
                    discrim_error_real.backward(retain_graph=True)

                    # Pass through deblurred inputs
                    output_discrim = discriminator(output).view(-1)
                    # Get loss
                    if torch.cuda.is_available():
                        labels = Variable(torch.full(output_discrim.shape, fake_label)).cuda()
                    else:
                        labels = Variable(torch.full(output_discrim.shape, fake_label))
                    discrim_error_fake = discrim_criterion(output_discrim, labels)
                    del labels

                    # Accumulate grads
                    discrim_error_fake.backward(retain_graph=True)
                    # Sum loss and backprop
                    discrim_total_error = discrim_error_fake + discrim_error_real

                    optimizer_discrim.step()

                    # For record keeping
                    loss_values['discrim_average_loss_on_deblurred'] += discrim_error_fake.data.item()
                    loss_values['discrim_average_loss'] += discrim_total_error.data.item()
                    loss_values['discrim_average_loss_on_real'] += discrim_error_real.data.item()

                    optimizer.zero_grad()
                    # ===================Train Deblurrer (generator)=====================
                    # Get discrim output
                    output_discrim = discriminator(output).view(-1)
                    # Get discrim loss
                    if torch.cuda.is_available():
                        labels = Variable(torch.full(output_discrim.shape, real_label)).cuda()
                    else:
                        labels = Variable(torch.full(output_discrim.shape, real_label))
                    # Discrim loss
                    gen_loss = discrim_criterion(output_discrim, labels)
                    del labels

                    # perceptual_loss
                    loss_perceptual = perceptual_loss(vgg_net,output,nonblurred_img)

                    # MSE loss
                    mse_loss = mse_criterion(output, nonblurred_img)

                    # Total loss
                    total_loss = mse_loss_weight * mse_loss + perceptual_loss_weight*loss_perceptual + gen_loss_weight * gen_loss

                    # ===================backward====================
                    total_loss.backward()
                    optimizer.step()
                    loss_values['deblur_average_gen_loss'] += gen_loss.data.item()
                    loss_values['deblur_average_percep_loss'] += loss_perceptual.data.item()
                    loss_values['deblur_average_mse_loss'] += mse_loss.data.item()
                    loss_values['deblur_total_average_loss'] += total_loss.data.item()

                # ===================log========================
                loss_values = {k: v/num_epochs for k, v in loss_values.items()}
                losses_per_epoch.append(loss_values)

                print('epoch [{}/{}], {}'.format(epoch+1, num_epochs, loss_values))

                #print('epoch [{}/{}], Deblurrer Total Average Loss: {:.4f}, ' +
                #                'Discrim Average Loss: {:.4f}, '
                #.format(epoch + 1, num_epochs, total_loss.data, discrim_total_error.data))
        except KeyboardInterrupt:
            torch.save(model.state_dict(),'semantic_model_interrupt.pth')
            torch.save(discriminator.state_dict(), 'discrim_interrupt.pth')
            f = open("losses.txt", "w")
            f.write(str(losses_per_epoch))
            f.close()
            sys.exit()
        break


    torch.save(model.state_dict(), 'semanticmodel.pth')
    torch.save(discriminator.state_dict(), 'discrim.pth')
    f = open("losses.txt", "w")
    f.write(str(losses_per_epoch))
    f.close()