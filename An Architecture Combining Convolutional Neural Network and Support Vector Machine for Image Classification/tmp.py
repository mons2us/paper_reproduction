import os
import torch


a = torch.tensor([1, 2, 0, 0, 0, 1])
b = torch.tensor([1, 2, 1, 0, 0, 1])

torch.sum(torch.eq(a, b)) / a.shape[0]


