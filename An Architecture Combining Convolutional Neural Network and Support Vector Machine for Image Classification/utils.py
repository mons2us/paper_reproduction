import os
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
    
    def forward(self, pred, gt, reg_term, device, current_batch_size):
        softmax_loss = nn.CrossEntropyLoss()
        loss = softmax_loss(pred, gt)
        
        return loss


class loss_svm(nn.Module):
    '''
    Softmax를 사용할 경우의 loss함수
    '''
    def __init__(self, penalty_param = 1):
        super(loss_svm, self).__init__()
        self.penalty_param = penalty_param
    
    def forward(self, pred, gt, reg_term, device, current_batch_size):
        hinge_loss = nn.MultiMarginLoss(p = 2, margin = 1)

        # Regularization term
        reg_loss = torch.mean(torch.square(reg_term))

        loss = self.penalty_param * reg_loss + hinge_loss(pred, gt)
        
        return loss

# custom/
# class loss_svm(nn.Module):
#     '''
#     SVM loss: Squared hinge loss with L2 regularization term
#     '''
#     def __init__(self, penalty_param = 1):
#         super(loss_svm, self).__init__()
#         self.penalty_param = penalty_param

#     def forward(self, pred, gt, reg_term, device, current_batch_size):
#         gt_dummy = self.onehot_tensor(gt, batch_size = current_batch_size)
#         gt_dummy = gt_dummy.to(device)

#         # Regularization term
#         #reg_loss = torch.mean(torch.matmul(reg_term, reg_term.t()))
#         reg_loss = torch.mean(torch.square(reg_term))

#         # hinge loss (1)
#         hinge_loss = 1 - torch.mul(pred, gt_dummy)

#         # squared hinge loss (2)
#         squared_hinge_loss = torch.square(torch.max(torch.zeros_like(hinge_loss), hinge_loss))

#         loss = reg_loss + self.penalty_param * torch.mean(squared_hinge_loss)

#         return loss

    @staticmethod
    def onehot_tensor(gt, batch_size):
        batch_size = batch_size
        one_hot = torch.zeros(batch_size, 10)
        one_hot[torch.arange(batch_size), gt] = 1
        
        # Rest classes to -1
        one_hot[one_hot == 0] = -1
        
        return one_hot


def save_plot(save_pth, loss_type, *acc_loss):

    # 학습 결과 확인 (Plot)
    if not os.path.exists(save_pth):
        print("Directory to save plots does not exist. Make one? [y | n]")
        dir_yn = str(input())

        if dir_yn == 'y':
            os.makedirs(save_pth)

        elif dir_yn == 'n':
            print("Please check directory. Not saving plot this time!")
            return

        else:
            raise Exception("Input should be either y or n")

    for i, data in enumerate(acc_loss):

        fig = plt.figure(figsize = (12, 6))
        plt.plot(data)
        
        plot_name = ''.join(('train_plot_acc_', loss_type, '.jpg')) if i == 0 else ''.join(('train_plot_loss_', loss_type, '.jpg'))
        plt.savefig(os.path.join(save_pth, plot_name))
        plt.close(fig)