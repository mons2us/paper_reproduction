import torch, torchvision
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F


'''
DCGAN model
(1) Generator

(2) Discriminator
'''

## DCGAN
class Generator(nn.Module):
    '''
    ConvTranspose2d(input_c, output_c, kernel_size, stride, padding)
    '''
    def __init__(self):

        super(Generator, self).__init__()

        self.g_fclayer = nn.Sequential(
            nn.Linear(100, 512*7*7),
            nn.BatchNorm1d(512*7*7),
            nn.LeakyReLU()
        )
        
        self.g_convlayer = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, 3, 2, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.g_fclayer(z)
        out = out.view(out.size()[0], 512, 7, 7)
        out = self.g_convlayer(out)
        return out



class Discriminator(nn.Module):
    '''
    ConvTranspose2d(input_c, output_c, kernel_size, stride, padding)
    '''
    def __init__(self):

        super(Discriminator, self).__init__()

        self.d_convlayer = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding = 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.d_fclayer = nn.Sequential(
            nn.Linear(64*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.d_convlayer(x)
        out = out.view(out.size()[0], -1)
        med_feature = out
        out = self.d_fclayer(out)
        return out, med_feature