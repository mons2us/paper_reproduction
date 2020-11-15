import os
import argparse
import torch, torchvision
import matplotlib.pyplot as plt

from dataset import load_datasets
from model import Generator, Discriminator
from train import train


parser = argparse.ArgumentParser(description = "AnoGAN Implementation")

parser.add_argument('--mode', default = 'train')
parser.add_argument('--data_type', type = str, default = 'MNIST',
                    help = "Data type to train or test: CIFAR10 available")
parser.add_argument('--batch_size', type = int, default = 128, metavar = 'N',
                    help = "Batch size to be used for training (default: 128)")
parser.add_argument('--epochs', type = int, default = 10, metavar = 'N',
                    help = "Number of epochs to be used for training (default: 10)")
parser.add_argument('--use_cuda', action='store_true', default = True,
                    help = "Whether to use cuda in training. If you don't want to use cuda, set this to False")
parser.add_argument('--seed', type = int, default = 2020, metavar = 'S',
                    help = "Random seed (default: 2020)")
parser.add_argument('--log_interval', type = int, default = 100, metavar = 'N',
                    help = "Logging interval in training (default: 10)")
parser.add_argument('--model_pth', type = str, default = './model',
                    help = "Path for the model to be saved or loaded from. Default is ./model;\
                        If using svm loss, model will automatically be saved in ./model with name: model_cnn_svm.pkl")
parser.add_argument('--plot_pth', type = str, default = './plot',
                    help = "Path for the result plot to be saved at. Default is ./plot")



if __name__ == "__main__":

    args = parser.parse_args()
    
    # whether to use cuda
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    
    # set device (gpu/cpu) according to args.use_cuda
    device = torch.device("cuda" if args.use_cuda else "cpu")
    torch.manual_seed(args.seed)
    
    # load dataset: CIFAR10
    assert args.data_type in ('MNIST'), "Choose data_type in [CIFAR10]"
    dataset_train, dataset_test = load_datasets(data_type = args.data_type, batch_size = args.batch_size)
    

    if args.use_cuda:
        model_g = Generator().cuda()
        model_d = Discriminator().cuda()

    if args.mode == 'train':
        gen_loss, dis_loss = train(model_g = model_g,
                                   model_d = model_d,
                                   trainset = dataset_train,
                                   lr = 1e-3,
                                   epochs = args.epochs,
                                   log_interval = args.log_interval,
                                   cuda = args.use_cuda,
                                   save_path = args.model_pth)