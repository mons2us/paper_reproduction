import torch.utils.data
import torch
import numpy as np
from torchvision import datasets, transforms

# train/test loader
def load_datasets(dir = './data', data_type = 'MNIST', batch_size = None, train_label = None, test_label = None):
    
    transform = transforms.Compose([
        #transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    if data_type == 'MNIST':

        train_tmp = datasets.MNIST(dir, train = True, download = True, transform = transform)
        train_idx = train_tmp.train_labels == train_label

        train_tmp.__dict__['data'] = train_tmp.__dict__['data'][train_idx]
        train_tmp.__dict__['targets'] = train_tmp.__dict__['targets'][train_idx]

        test_tmp = datasets.MNIST(dir, train = False, download = True, transform = transform)
        test_idx = torch.tensor(np.isin(test_tmp.test_labels.numpy(), test_label))

        test_tmp.__dict__['data'] = test_tmp.__dict__['data'][test_idx]
        test_tmp.__dict__['targets'] = test_tmp.__dict__['targets'][test_idx]
        
        # MNIST Digit dataset
        TRAIN_DATASETS = torch.utils.data.DataLoader(
            train_tmp,
            batch_size = batch_size,
            shuffle = True, 
            drop_last = True)

        TEST_DATASETS = torch.utils.data.DataLoader(
            test_tmp,
            batch_size = batch_size,
            shuffle = True,
            drop_last = True)

    return TRAIN_DATASETS, TEST_DATASETS