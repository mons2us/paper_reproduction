import torch
import torchvision
import torchvision.transforms as transforms
import yaml


def load_dataset(data_type, dataset_dir, batch_size, num_class):
    '''
    label/unlabeled data: 4000/46000 for cifar10 dataset
    '''
    if data_type == 'cifar10':
        
        U_batch_size = int(batch_size * 0.92)
        S_batch_size = batch_size - U_batch_size
        
        transform_train = transforms.Compose([
            transforms.RandomAffine(0, translate=(1/16,1/16)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    else:
        raise ValueError('no such dataset')

    S_idx = []
    U_idx = []
    dataiter = iter(trainloader)
    trainlist = [[] for x in range(num_class)]
    step = int(len(trainset)/batch_size)
    
    return trainset, trainloader, testset, testloader, S_batch_size, U_batch_size, S_idx, U_idx, dataiter, trainlist, step