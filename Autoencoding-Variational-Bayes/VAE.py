import easydict
import argparse
from tqdm import tnrange
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image



# parser = argparse.ArgumentParser(description = "Variational Autoencoder on MNIST")
# parser.add_argument('--batch_size', type = int, default = 64, metavar = 'N',
#                     help = "Batch size to be used for training (default: 64)")
# parser.add_argument('--epochs', type = int, default = 10, metavar = 'N',
#                     help = "Number of epochs to be used for training (default: 10)")
# parser.add_argument('--use_cuda', action='store_true', default = True,
#                     help = "Whether to use cuda in training. If you don't want to use cuda, set this to False")
# parser.add_argument('--seed', type = int, default = 2020, metavar = 'S',
#                     help = "Random seed (default: 2020)")
# parser.add_argument('--log_interval', type = int, default = 10, metavar = 'N',
#                     help = "Logging interval in training (default: 10)")

# args = parser.parse_args()

# arguments stored
args = easydict.EasyDict({'batch_size': 64,
                          'epochs': 10,
                          'use_cuda': True,
                          'seed': 2020,
                          'log_interval': 10})

# whether to use cuda
args.use_cuda = args.use_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

# set device (gpu/cpu)
device = torch.device("cuda" if args.use_cuda else "cpu")

# train/test loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train = True, download = True, transform = transforms.ToTensor()),
    batch_size = args.batch_size,
    shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train = False, download = True, transform = transforms.ToTensor()),
    batch_size = args.batch_size,
    shuffle = True
)


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
        reconstructed = self.decoder(z)
        return reconstructed, z_mu, z_logvar
    

'''
Latent Dimension을 달리하면서 학습할 수 있다.
물론 dim이 커질 수록 복원이 잘 될 것이라는 추측은 가능하다.
'''
model20 = VAE(latent_dim = 20).to(device) # to cuda if possible, or to cpu
model10 = VAE(latent_dim = 10).to(device) # to cuda if possible, or to cpu
model5 = VAE(latent_dim = 5).to(device) # to cuda if possible, or to cpu
model2 = VAE(latent_dim = 2).to(device) # to cuda if possible, or to cpu

selected_model = model2

def loss_function(x_org, x_recons, mu, logvar):
    '''
    VAE 모델의 목적함수(손실함수)
     (1) Recontruction Error: 원본 데이터 x_org와 복원된 데이터 x_recons 간 차이(정보손실) 측정
     (2) KLD Error: p(z|x)에 근사하려는 q(z|x)가 얼마나 잘 근사되는지 측정
                    이 때 가정하는 p(z|x)가 N(0, 1)이므로, q(z|x)가 N(0, 1)이 될 때 최소화 된다.
    '''
    REC_loss = F.binary_cross_entropy(x_recons, x_org, size_average = False)
    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    tot_loss = (REC_loss + KLD_loss)
    return tot_loss

# 학습 모델 정의
def train(model, trainset, lr = 1e-3, epochs = args.epochs):
    
    model = model
    trainset = trainset
    
    train_losses = []
    train_loss = 0
    
    model.train()
    
    # optimizer는 Adam 사용
    optimizer = optim.Adam(
        model.parameters(),
        lr = lr
    )
    
    
    for epoch in tnrange(epochs, desc = 'Training Process'):
        
        for batch_idx, (images, _) in enumerate(trainset):
            
            images = Variable(images).to(device)
            optimizer.zero_grad()
            
            reconstructed, mu, logvar = model(images)
            loss = loss_function(images, reconstructed, mu, logvar) # loss 계산
            loss.backward() # Backpropagation
            
            train_losses.append(loss.item() / len(images)) # 배치별로 backprop하여 loss를 loss list에 담는다.
            train_loss += loss.item()
            
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.4f}".format(
                    epoch + 1,
                    batch_idx * len(images),
                    len(trainset.dataset),
                    batch_idx * 100. / len(trainset),
                    loss.item() / len(images)
                ))
                
        print("======= Epoch: {}  Average Loss: {:.4f} =======".format(
            epoch + 1,
            train_loss / len(trainset.dataset)
        ))
        
        train_loss = 0
        
    return train_losses



# 학습 수행
train_losses = train(model = selected_model, trainset = train_loader)

# 학습 결과 확인
plt.figure(figsize = (12, 6))
plt.plot(train_losses)
plt.show()



# Inference 모델 정의
def test(model, testset):
    '''
    train과 동일한 방법으로 loss_function을 계산, 즉,
     (1) 학습된 모델로 input image를 reconstruct하고 원본 이미지와 비교
     (2) 추정된 q(z|x)와 N(0, 1) 간 거리를 비교
    하여 test_loss 계산
    '''
    model = model
    testset = testset
    
    test_loss = 0.0
    
    model.eval()
    with torch.no_grad(): # gradients를 freezing하여 inference만 수행
        
        for batch_idx, (images, _) in enumerate(testset):
            
            images = Variable(images).to(device)
            reconstructed, mu, logvar = model(images)
            test_loss += loss_function(images, reconstructed, mu, logvar).item()
            
        return test_loss / len(testset.dataset)
    
    
test_loss = test(selected_model, testset = test_loader)
print("Test loss: {:.4f}".format(test_loss))




# 학습 결과 시각화

# (1) MNIST 데이터 복원
def show_image(img):
    
    img = img.cpu().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0))) # 원래 이미지가 (channel, x, y)로 되어있기 때문에 plt.imshow를 하기 위해 차원을 변경해준다.
    plt.axis('off')
    plt.show()
    
    
def visualize_reconstructed(model1, model2, dataset, to_show = 24):

    images, _ = iter(dataset).next()
    images = images.to(device) # to cuda
    images = images[:to_show, :, :] # 20개 show
    
    original = Variable(images)
    reconstructed = model1(original)[0].cpu().data
    reconstructed = reconstructed.reshape((-1, 1, 28, 28)) # shape 변경

    reconstructed2 = model2(original)[0].cpu().data
    reconstructed2 = reconstructed2.reshape((-1, 1, 28, 28)) # shape 변경
    
    show_image(make_grid(original))
    show_image(make_grid(reconstructed))
    show_image(make_grid(reconstructed2))
    
visualize_reconstructed(model1 = model20, model2 = model5, dataset = train_loader)


# (2) encoder 시각화
def visualize_encoder(model, dataset):
    mu_x, mu_y, label_list = [], [], []
    
    for images, labels in iter(dataset):
        images = images.to(device) # to cuda
        mu_set, _ = model.encoder(images)
        mu_x = np.append(mu_x, mu_set[:, 0].data.cpu().numpy())
        mu_y = np.append(mu_y, mu_set[:, 1].data.cpu().numpy())
        label_list = np.append(label_list, labels.numpy())
        
    plt.figure(figsize = (10, 10))
    plt.scatter(mu_x, mu_y, c = label_list, cmap = 'viridis')
    plt.colorbar()
    
    plt.xlabel("mu_1")
    plt.ylabel("mu_2")
    plt.show()
    
visualize_encoder(selected_model, test_loader)


# (3) decoder 시각화


def visualize_decoder(model, num = 20, range_type = 'g'):
    image_grid = np.zeros([num*28, num*28])
    
    if range_type == 'l':
        range_space = np.linspace(-4, 4, num)
    elif range_type == 'g':
        range_space = norm.ppf(np.linspace(0.01, 0.99, num))
    else:
        range_space = range_type
        
    for i, x in enumerate(range_space):
        for j, y in enumerate(reversed(range_space)):
            z = torch.FloatTensor([[x, y]]).to(device)
            image = model.decoder(z)
            image = image.data.cpu().numpy().reshape(28, 28)
            image_grid[(j*28): ((j+1)*28), (i*28):((i+1)*28)] = image
        
    plt.figure(figsize = (10, 10))
    
    plt.imshow(image_grid, cmap = 'gray')
    plt.show()
    
visualize_decoder(model = model2)

norm.ppf(np.linspace(0.01, 0.99, 20))
list(reversed(norm.ppf(np.linspace(0.01, 0.99, 20))))



z = torch.FloatTensor([[1.5, 0]]).to(device)

image = model2.decoder(z)
image = image.data.cpu().numpy().reshape(28, 28)

plt.figure(figsize = (2, 2))
plt.imshow(image, cmap = 'gray')
plt.show()

