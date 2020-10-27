import torch
from torch import nn, optim
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np


a = torch.mul(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
print(torch.max(torch.tensor(0), a))



class loss_softmax(nn.Module):
    '''
    Softmax를 사용할 경우의 loss함수
    '''
    def __init__(self):
        super(loss_softmax).__init__()
    
    def forward(pred, gt):
        softmax_loss = nn.CrossEntropyLoss()
        loss = softmax_loss(pred, gt)
        
        return loss


class loss_svm(nn.Module):
    
    def __init__(self):
        super(loss_svm).__init__()
        
    def forward(self, pred, gt):
        hinge_loss = 1 - torch.mul(pred, gt)
        loss = torch.max(torch.tensor(0), hinge_loss)
        
        return loss