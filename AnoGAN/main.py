import os
import argparse
import easydict
import torch, torchvision
import matplotlib.pyplot as plt

from dataset import load_datasets
from model import Generator, Discriminator
from train import train
from anomaly import *

parser = argparse.ArgumentParser(description = "AnoGAN Implementation")

parser.add_argument('--mode', default = 'train')
parser.add_argument('--data_type', type = str, default = 'MNIST',
                    help = "Data type to train or test: CIFAR10 available")
parser.add_argument('--batch_size', type = int, default = 1024, metavar = 'N',
                    help = "Batch size to be used for training (default: 128)")
parser.add_argument('--train_label', type = int, default = 999, metavar = 'N',
                    help = "Label to be used in training DCGAN (considered as normal data)")
parser.add_argument('--test_label', type = str, default = "None", metavar = 'N',
                    help = "Normal label and anomaly label")
parser.add_argument('--epochs', type = int, default = 10, metavar = 'N',
                    help = "Number of epochs to be used for training (default: 10)")
parser.add_argument('--use_cuda', action='store_true', default = True,
                    help = "Whether to use cuda in training. If you don't want to use cuda, set this to False")
parser.add_argument('--log_interval', type = int, default = 100, metavar = 'N',
                    help = "Logging interval in training (default: 10)")
parser.add_argument('--model_pth', type = str, default = './model',
                    help = "Path for the model to be saved or loaded from. Default is ./model;\
                        If using svm loss, model will automatically be saved in ./model with name: model_cnn_svm.pkl")
parser.add_argument('--plot_pth', type = str, default = './plot',
                    help = "Path for the result plot to be saved at. Default is ./plot")



if __name__ == "__main__":

    args = parser.parse_args()

    assert args.train_label != 999 and args.test_label != "None", "Target labels must be provided."
    print(f"Training Label: {args.train_label}, Test Label: {args.test_label}")

    # whether to use cuda
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    
    # set device (gpu/cpu) according to args.use_cuda
    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    # load dataset: CIFAR10
    assert args.data_type in ('MNIST'), "Choose data_type in [CIFAR10]"
    dataset_train, dataset_test = load_datasets(data_type = args.data_type,
                                                batch_size = args.batch_size,
                                                train_label = args.train_label,
                                                test_label = list(map(int, args.test_label.split(','))))

    if args.mode == 'train':

        gen_loss, dis_loss = train(model_g = Generator(),
                                   model_d = Discriminator(),
                                   trainset = dataset_train,
                                   batch_size = args.batch_size,
                                   lr = 0.0004,
                                   epochs = args.epochs,
                                   log_interval = args.log_interval,
                                   cuda = args.use_cuda,
                                   save_path = args.model_pth)

    if args.mode == 'anogan':

        # Restore DCGAN model
        model_g = Generator()
        model_g.load_state_dict(torch.load('./model/generator.pkl'))
        
        model_d = Discriminator()
        model_d.load_state_dict(torch.load('./model/discriminator.pkl'))

        # Data to calculate anomaly score
        for i, (images, _) in enumerate(dataset_test):
            inf_data = images[:16, :, :, :]
            break
        
        train_anogan(model_g,
                     model_d,
                     inf_data,
                     inf_data.shape[0],
                     lr = 0.0002,
                     max_iter = 5000,
                     cuda = True)

