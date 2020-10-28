import torch
from torch import nn, optim
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np


class loss_softmax(nn.Module):
    '''
    Softmax를 사용할 경우의 loss함수
    '''
    def __init__(self):
        super(loss_softmax, self).__init__()
    
    def forward(self, pred, gt):
        softmax_loss = nn.CrossEntropyLoss()
        loss = softmax_loss(pred, gt)
        
        return loss

class loss_svm(nn.Module):
    
    def __init__(self):
        super(loss_svm, self).__init__()

    def forward(self, pred, gt):
        pred_label = torch.argmax(pred, axis = 1)
        hinge_loss = 1 - torch.mul(pred_label, gt)
        print(hinge_loss)
        loss = torch.max(torch.zeros_like(hinge_loss), hinge_loss)
        print(loss)
        
        return loss