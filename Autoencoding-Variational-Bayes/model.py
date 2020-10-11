import torch, torchvision
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F

# Modeling Variational Autoencoder
class VAE(nn.Module):
    '''
    `VAE 모델
    (1) 데이터를 input으로 받고
    (2) encoder를 통해 latent variable로 변환 (샘플)
    (3) latent variable을 다시 decoder로 동일한 크기의 x' 생성
    
    `hyperparameter
    인코더와 디코더가 hidden <--> latent의 구조일 때
    각 layer의 크기 (hidden: 400, latent: 20)
    '''
    def __init__(self, input_dim = 784, hidden_dim = 400, latent_dim = 20):
        super(VAE, self).__init__()
        
        self.fc_encode = nn.Linear(input_dim, hidden_dim)
        self.fc_mean   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.fc_decoder1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_decoder2 = nn.Linear(hidden_dim, input_dim)
    
    def encoder(self, to_encode):
        e1   = F.relu(self.fc_encode(to_encode.view(-1, 784)))
        mu   = self.fc_mean(e1)
        logvar  = self.fc_logvar(e1)
        return mu, logvar
    
    def decoder(self, to_decode):
        d1 = F.relu(self.fc_decoder1(to_decode))
        out = self.fc_decoder2(d1)
        out = F.sigmoid(out)
        return out


    def sample_q(self, mu, logvar):
        '''
        `앞선 neural network 구조에서 생성된 mu, logvar을 이용해 정규분포를 만들고,
        이 정규분포로부터 z를 샘플링 (encode)
        `이 때 reparametrization trick을 위해 N(0, 1)로부터 epsilon을 샘플링
        '''
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std) # sampling epsilon
        Z = mu + std * eps
        return Z
    
    def forward(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.sample_q(z_mu, z_logvar)
        reconstructed = self.decoder(z).reshape((-1, 1, 28, 28))
        return reconstructed, z_mu, z_logvar