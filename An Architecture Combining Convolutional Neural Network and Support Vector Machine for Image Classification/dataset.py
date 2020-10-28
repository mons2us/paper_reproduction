import torch.utils.data
from torchvision import datasets, transforms

# train/test loader
def load_datasets(dir = './data', data_type = 'MNIST', batch_size = 128):
    
    if data_type == 'MNIST':
        # MNIST Digit dataset
        TRAIN_DATASETS = torch.utils.data.DataLoader(
            datasets.MNIST(dir, train = True, download = True, transform = transforms.ToTensor()),
            batch_size = batch_size,
            shuffle = True)

        TEST_DATASETS = torch.utils.data.DataLoader(
            datasets.MNIST(dir, train = False, download = True, transform = transforms.ToTensor()),
            batch_size = batch_size,
            shuffle = True)
    
    else:
        # MNIST Fashion dataset
        TRAIN_DATASETS = torch.utils.data.DataLoader(
            datasets.FashionMNIST(dir, train = True, download = True, transform = transforms.ToTensor()),
            batch_size = batch_size,
            shuffle = True)

        TEST_DATASETS = torch.utils.data.DataLoader(
            datasets.FashionMNIST(dir, train = False, download = True, transform = transforms.ToTensor()),
            batch_size = batch_size,
            shuffle = True)
    
    return TRAIN_DATASETS, TEST_DATASETS