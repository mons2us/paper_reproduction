import torch.utils.data
from torchvision import datasets, transforms

# train/test loader
def load_datasets(dir = './data', batch_size = 256):
    
    # MNIST Digit dataset
    TRAIN_DATASETS = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train = True, download = True, transform = transforms.ToTensor()),
        batch_size = batch_size,
        shuffle = True)

    TEST_DATASETS = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train = False, download = True, transform = transforms.ToTensor()),
        batch_size = batch_size,
        shuffle = True)
    
    # MNIST Fashion dataset
    FASHION_TRAIN_DATASETS = torch.utils.data.DataLoader(
        datasets.FashionMNIST(dir, train = True, download = True, transform = transforms.ToTensor()),
        batch_size = batch_size,
        shuffle = True)

    FASHION_TEST_DATASETS = torch.utils.data.DataLoader(
        datasets.FashionMNIST(dir, train = False, download = True, transform = transforms.ToTensor()),
        batch_size = batch_size,
        shuffle = True)
    
    return TRAIN_DATASETS, TEST_DATASETS