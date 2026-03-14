# Deep Convolutional GANs
# Implementation of DCGAN (Radford et al., 2015) trained on CIFAR-10.

# Importing the libraries
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Setting some hyperparameters
batchSize = 64  # We set the size of the batch.
imageSize = 64  # We set the size of the generated images (64x64).

# Detect GPU/CPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating the transformations
transform = transforms.Compose([
    transforms.Resize(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Defining the weights_init function that takes as input a neural network m
# and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Defining the generator

class G(nn.Module):
    """Generator network: maps a latent vector (z=100) to a 3x64x64 image."""

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


# Defining the discriminator

class D(nn.Module):
    """Discriminator network: classifies 3x64x64 images as real or fake."""

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


def train():
    """Main training loop for the DCGAN."""

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Loading the dataset
    dataset = dset.CIFAR10(
        root=data_dir, download=True, transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batchSize, shuffle=True, num_workers=2,
    )

    # Creating the generator
    netG = G().to(device)
    netG.apply(weights_init)

    # Creating the discriminator
    netD = D().to(device)
    netD.apply(weights_init)

    # Training the DCGANs
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(25):

        for i, data in enumerate(dataloader, 0):

            # 1st Step: Updating the weights of the discriminator

            netD.zero_grad()

            # Training the discriminator with a real image of the dataset
            real, _ = data
            real = real.to(device)
            target = torch.ones(real.size(0), device=device)
            output = netD(real)
            errD_real = criterion(output, target)

            # Training the discriminator with a fake image from the generator
            noise = torch.randn(real.size(0), 100, 1, 1, device=device)
            fake = netG(noise)
            target = torch.zeros(real.size(0), device=device)
            output = netD(fake.detach())
            errD_fake = criterion(output, target)

            # Backpropagating the total error
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # 2nd Step: Updating the weights of the generator

            netG.zero_grad()
            target = torch.ones(real.size(0), device=device)
            output = netD(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

            # 3rd Step: Printing the losses and saving images every 100 steps

            print(
                '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                % (epoch, 25, i, len(dataloader), errD.item(), errG.item())
            )
            if i % 100 == 0:
                vutils.save_image(
                    real,
                    os.path.join(results_dir, 'real_samples.png'),
                    normalize=True,
                )
                with torch.no_grad():
                    netG.eval()
                    fake_samples = netG(noise)
                    netG.train()
                vutils.save_image(
                    fake_samples.data,
                    os.path.join(results_dir, 'fake_samples_epoch_%03d.png' % epoch),
                    normalize=True,
                )

    print("Training complete.")


if __name__ == "__main__":
    train()
