import torch
#import random
import numpy as np
import math
import torch.nn as nn

def set_seed(configs):
    seed = configs['train']['SEED']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss() 
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2) 
    return (loss1+loss2)

def loss_cot(U_p1, U_p2, U_batch_size):
# the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    a1 = 0.5 * (S(U_p1) + S(U_p2))
    loss1 = -torch.sum(a1 * torch.log(a1))
    loss2 = -torch.sum(S(U_p1) * LS(U_p1))
    loss3 = -torch.sum(S(U_p2) * LS(U_p2))

    return (loss1 - 0.5 * (loss2 + loss3))/U_batch_size

def loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2, logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2, batch_size):
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    
    a = torch.sum(S(logit_S2) * LS(perturbed_logit_S1))
    b = torch.sum(S(logit_S1) * LS(perturbed_logit_S2))
    c = torch.sum(S(logit_U2) * LS(perturbed_logit_U1))
    d = torch.sum(S(logit_U1) * LS(perturbed_logit_U2))

    return -(a+b+c+d)/batch_size