import torch
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import numpy as np
from utils import show_image
from scipy.stats import norm


# 학습 결과 시각화
# (1) MNIST 데이터 복원
def visualize_reconstructed(model, dataset, to_show = 24, cuda = True):

    device = torch.device("cuda" if cuda else "cpu") # to cuda if possible
    
    images, _ = iter(dataset).next()
    images = images.to(device) # to cuda
    images = images[:to_show, :, :] # 20개 show
    
    original = Variable(images)
    reconstructed = model(original)[0].cpu().data
    reconstructed = reconstructed.reshape((-1, 1, 28, 28)) # shape 변경
    
    show_image(make_grid(original))
    show_image(make_grid(reconstructed))


# (2) encoder 시각화
def visualize_encoder(model, dataset, cuda = True):
    
    device = torch.device("cuda" if cuda else "cpu") # to cuda if possible
    
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


# (3) decoder 시각화
def visualize_decoder(model, num = 20, range_type = 'g', cuda = True):
    
    device = torch.device("cuda" if cuda else "cpu") # to cuda if possible
    
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
    
