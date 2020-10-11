import torch
from torch import nn, optim
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np

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

def show_image(img):
    img = img.to('cpu').numpy()
    plt.imshow(np.transpose(img, (1, 2, 0))) # 원래 이미지가 (channel, x, y)로 되어있기 때문에 plt.imshow를 하기 위해 차원을 변경해준다.
    plt.axis('off')
    plt.show()