import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    """This is a VAE model for MNIST dataset. If using other datasets, you need to change the input_dim"""
    def __init__(self, input_dim=784,latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.fc21 = nn.Linear(input_dim//2, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetGenerator(nn.Module):
    """Generator architecture from CycleGAN paper"""
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        # Resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator from CycleGAN paper"""
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class CycleGAN(nn.Module):
    """
    CycleGAN model: contains 2 generators and 2 discriminators.
    G: X -> Y, F: Y -> X
    D_Y: Discriminates Y, D_X: Discriminates X
    """
    def __init__(self, input_nc, output_nc, ngf=64, ndf=64, n_blocks=9):
        super(CycleGAN, self).__init__()
        # Generators
        self.G = ResnetGenerator(input_nc, output_nc, ngf, n_blocks)
        self.F = ResnetGenerator(output_nc, input_nc, ngf, n_blocks)
        # Discriminators
        self.D_X = NLayerDiscriminator(input_nc, ndf)
        self.D_Y = NLayerDiscriminator(output_nc, ndf)

    def forward(self, x, y):
        """
        x: images from domain X
        y: images from domain Y
        Returns:
            fake_y: G(x)
            rec_x: F(G(x))
            fake_x: F(y)
            rec_y: G(F(y))
        """
        fake_y = self.G(x)
        rec_x = self.F(fake_y)
        fake_x = self.F(y)
        rec_y = self.G(fake_x)
        return fake_y, rec_x, fake_x, rec_y


#TODO: 
# 1) Code a VAE model
# 2) Code a GAN model
# 3) Code a DDPM model
# 4) Code a Joint-VAE model
# 5) Code a Audio-VAE model
# 6) Code a model which generates an obj kind of thing
# 7) Code a shape model
# 9) Code a resnet kind of thing on ImageNet or MNIST