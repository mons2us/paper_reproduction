import torch, torchvision
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F

'''
CNN model used in the paper,
(1) INPUT: 32 × 32 × 1

# Block 1
(2) CONV5: 5 × 5 size, 32 filters, 1 stride
(3) ReLU: max(0,hθ(x))
(4) POOL: 2 × 2 size, 1 stride

# Block 2
(5) CONV5: 5 × 5 size, 64 filters, 1 stride
(6) ReLU: max(0,hθ(x))
(7) POOL: 2 × 2 size, 1 stride

# Block 3 (FC)
(8) FC: 1024 Hidden Neurons
(9) DROPOUT: p = 0.5
(10) FC: 10 Output Clas
'''

class CNN(nn.Module):
    
    def __init__(self, in_channels = 1, class_num = 10):
        super(CNN, self).__init__()
    
        # Common Function
        self.maxpool_22 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Block 1
        b1_conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 5, stride = 1, padding = 1)
        
        # Res: (32-5+2)/1 + 1 = 30
        self.b1_block = nn.Sequential(b1_conv1,
                                    nn.ReLU(),
                                    self.maxpool_22)
        
        # Block 2
        b2_conv1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 1)
        
        # res: (30-5+2)/1 + 1 = 28
        self.b2_block = nn.Sequential(b2_conv1,
                                    nn.ReLU(),
                                    self.maxpool_22)
        
        # FC layer
        fc_1 = nn.Linear(28 * 28 * 64, class_num)
        
        
    def forward(self, x):
        
        out = self.b1_block(x)
        out = self.b2_block(out)
        out = self.fc_1(out)
        
        return out
    
cnn_model = CNN()