import torch
from torch import nn
from torch.autograd import Variable

# Inspired by https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py
class ConvVAE(nn.Module):
    def __init__(self, input_channels, init_filter_size, num_latent):
        super(ConvVAE, self).__init__()
        self.input_channels = input_channels
        self.init_filter_size = init_filter_size
        self.num_latent = num_latent

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, init_filter_size, 4, 2, 1),
            nn.BatchNorm2d(init_filter_size),
            nn.LeakyReLU(0.2),

            nn.Conv2d(init_filter_size, init_filter_size * 2, 4, 2, 1),
            nn.BatchNorm2d(init_filter_size * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(init_filter_size * 2, init_filter_size * 4, 4, 2, 1),
            nn.BatchNorm2d(init_filter_size * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(init_filter_size * 4, init_filter_size * 8, 4, 2, 1),
            nn.BatchNorm2d(init_filter_size * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(init_filter_size * 8, init_filter_size * 8, 4, 2, 1),
            nn.BatchNorm2d(init_filter_size * 8),
            nn.LeakyReLU(0.2)
        )

        self.fc1 = nn.Linear(init_filter_size* 8 * 4, num_latent)
        self.fc2 = nn.Linear(init_filter_size * 8 * 4, num_latent)

        self.d1 = nn.Linear(num_latent, init_filter_size * 8 * 2 * 4)

        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(init_filter_size * 8 * 2, init_filter_size * 8, 3, 1),
            nn.BatchNorm2d(init_filter_size * 8, 1.e-3),
            nn.LeakyReLU(0.2),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(init_filter_size * 8, init_filter_size * 4, 3, 1),
            nn.BatchNorm2d(init_filter_size * 4, 1.e-3),
            nn.LeakyReLU(0.2),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(init_filter_size * 4, init_filter_size * 2, 3, 1),
            nn.BatchNorm2d(init_filter_size * 2, 1.e-3),
            nn.LeakyReLU(0.2),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(init_filter_size * 2, init_filter_size, 3, 1),
            nn.BatchNorm2d(init_filter_size, 1.e-3),
            nn.LeakyReLU(0.2),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(init_filter_size, input_channels, 3, 1),
            nn.Sigmoid()
        )

        self.relu_decode = nn.ReLU()

    def encode(self, x):
        out = self.encoder.forward(x)
        encoded = out.view(-1, self.init_filter_size * 8 * 4)
        return self.fc1(encoded), self.fc2(encoded)

    def decode(self, z):
        d1_out = self.relu_decode(self.d1(z))
        d1_out = d1_out.view(-1, self.init_filter_size * 8 * 2, 2, 2)
        return self.decoder(d1_out)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def get_latent_variables(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_channels, self.init_filter_size, self.init_filter_size))
        return self.reparametrize(mu, logvar)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_channels, self.init_filter_size, self.init_filter_size))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar