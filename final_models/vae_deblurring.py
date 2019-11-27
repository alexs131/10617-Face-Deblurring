import torch
from torch import nn
from torch.autograd import Variable
from face_vae import ConvVAE
from LFWC import LFWC
import sys

def loss(output, x, mu, logvar, loss_fn):
    BCE = loss_fn(output, x)

    KLD = torch.sum(mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)).mul_(-0.5)
    return BCE + KLD

if __name__ == '__main__':
    num_init_filters = 128
    num_latent_variables = 500
    num_epochs = 100
    batch_size = 8

    vae = ConvVAE(3, num_init_filters, num_latent_variables)

    recon_loss = nn.BCELoss()
    recon_loss.size_average = False
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    dataset = LFWC(["../data/train/faces_blurred"], "../data/train/faces")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    while (True):
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for data in data_loader:
                    if torch.cuda.is_available():
                        blurred_img = Variable(data['blurred']).cuda()
                        nonblurred_img = Variable(data['nonblurred']).cuda()
                    else:
                        blurred_img = Variable(data['blurred'])
                        nonblurred_img = Variable(data['nonblurred'])

                    optimizer.zero_grad()
                    output, mu, logvar = vae(blurred_img)

                    loss_out = loss(output, nonblurred_img, mu, logvar, recon_loss)
                    loss_out.backward()
                    epoch_loss += loss_out.data.item()

                    optimizer.step()
                print("Epoch [{}/{}], Epoch Ave Loss: {:.4f}".format(epoch, num_epochs, epoch_loss / num_epochs))
        except KeyboardInterrupt:
            torch.save(vae.state_dict(),'vae_model_interrupt.pth')
            sys.exit()
        break
    torch.save(vae.state_dict(), 'vae_model.pth')


