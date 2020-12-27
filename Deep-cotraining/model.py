import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import numpy as np
import random
import sys, os

# Source: https://github.com/benathi/fastswa-semi-sup/blob/master/mean_teacher/architectures.py
class CNN(nn.Module):
    def __init__(self, num_classes = 10, dropout = 0.5):
            super(CNN, self).__init__()

            self.convlayer1 = nn.Sequential(
                weight_norm(nn.Conv2d(3, 128, 3, padding = 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(128, 128, 3, padding = 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(128, 128, 3, padding = 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                
                nn.MaxPool2d(2, stride = 2, padding = 0),
                nn.Dropout(dropout),
            )
            
            self.convlayer2 = nn.Sequential(
                weight_norm(nn.Conv2d(128, 256, 3, padding = 1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(256, 256, 3, padding = 1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(256, 256, 3, padding = 1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                
                nn.MaxPool2d(2, stride = 2, padding = 0),
                nn.Dropout(dropout),
            )
            
            self.convlayer3 = nn.Sequential(
                weight_norm(nn.Conv2d(256, 512, 3, padding = 0)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(512, 256, 1, padding = 0)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(256, 128, 1, padding = 0)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                
                nn.AvgPool2d(6, stride = 2, padding = 0)
            )
            
            self.fclayer = nn.Sequential(
                weight_norm(nn.Linear(128, num_classes))
            )
        
    def forward(self, x):

            # ---------
            #  layer 1
            # ---------
            out = self.convlayer1(x)

            # ---------
            #  layer 2
            # ---------
            out = self.convlayer2(out)
            
            # ---------
            #  layer 3
            # ---------
            out = self.convlayer3(out)
            
            # ---------
            #  fc layer
            # ---------
            out = out.view(-1, 128)
            out = self.fclayer(out)
            
            return out

# Dropout value setting!
def co_train_classifier(num_classes = 10, dropout = 0.0):
    model = CNN(num_classes = num_classes, dropout = dropout)
    return model