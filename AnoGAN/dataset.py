import torch.utils.data
from torchvision import datasets, transforms

# train/test loader
def load_datasets(dir = './data', data_type = 'MNIST', batch_size = 8192):
    
    transform = transforms.Compose([
        #transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    if data_type == 'MNIST':
        # MNIST Digit dataset
        TRAIN_DATASETS = torch.utils.data.DataLoader(
            datasets.MNIST(dir, train = True, download = True, transform = transform),
            batch_size = batch_size,
            shuffle = True, 
            drop_last = True)

        TEST_DATASETS = torch.utils.data.DataLoader(
            datasets.MNIST(dir, train = False, download = True, transform = transform),
            batch_size = batch_size,
            shuffle = True,
            drop_last = True)

    return TRAIN_DATASETS, TEST_DATASETS