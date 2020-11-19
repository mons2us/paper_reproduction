import torch, torchvision
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
import easydict

CONFIGS = easydict.EasyDict({
    'image_size': 28,
    'z_dim': 100,
    'g_dim': 64,
    'd_dim': 8,
    'bridge_channel': 1
})

## DCGAN
class Generator(nn.Module):
    '''
    (1) Generator
    '''
    def __init__(self,
                 image_size = CONFIGS.image_size,
                 z_dim = CONFIGS.z_dim,
                 g_dim = CONFIGS.g_dim,
                 channel = CONFIGS.bridge_channel):

        super(Generator, self).__init__()

        self.rat = int(image_size / 4)
        self.channel = channel

        self.z_dim = z_dim
        self.g_dim = g_dim

        self.g_fclayer = nn.Sequential(
            nn.Linear(z_dim, g_dim*8*self.rat*self.rat),
            nn.BatchNorm1d(g_dim*8*self.rat*self.rat),
            nn.ReLU()
        )
        
        self.g_convlayer = nn.Sequential(
            nn.ConvTranspose2d(g_dim*8, g_dim*4, 3, 2, 1, 1),
            nn.BatchNorm2d(g_dim*4),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(g_dim*4, g_dim*2, 3, 1, 1),
            nn.BatchNorm2d(g_dim*2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(g_dim*2, g_dim, 3, 1, 1),
            nn.BatchNorm2d(g_dim),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(g_dim, channel, 3, 2, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.g_fclayer(z)
        out = out.view(out.shape[0], self.g_dim*8, self.rat, self.rat)
        out = self.g_convlayer(out)
        return out



class Discriminator(nn.Module):
    '''
    ConvTranspose2d(input_c, output_c, kernel_size, stride, padding)
    '''
    def __init__(self,
                 image_size = CONFIGS.image_size,
                 d_dim = CONFIGS.d_dim,
                 channel = CONFIGS.bridge_channel):

        super(Discriminator, self).__init__()

        self.rat = int(image_size / 4)
        self.d_dim = d_dim

        self.d_convlayer = nn.Sequential(

            nn.Conv2d(channel, d_dim, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(d_dim),
            nn.LeakyReLU(),

            nn.Conv2d(d_dim, d_dim*2, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(d_dim*2),
            nn.LeakyReLU(),

            nn.Conv2d(d_dim*2, d_dim*4, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(d_dim*4),
            nn.LeakyReLU(),

            nn.Conv2d(d_dim*4, d_dim*8, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(d_dim*8),
            nn.LeakyReLU()
        )
        
        self.d_fclayer = nn.Sequential(
            nn.Linear(d_dim*8*self.rat*self.rat, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.d_convlayer(x)
        out = out.view(out.shape[0], -1)
        med_feature = out
        out = self.d_fclayer(out)
        return out, med_feature