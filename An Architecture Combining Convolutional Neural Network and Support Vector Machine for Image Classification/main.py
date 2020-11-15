import os
import argparse
import torch, torchvision
import matplotlib.pyplot as plt

from dataset import load_datasets
from model import CNN
from train import train
from test import test
from utils import save_plot


#from visualize import visualize_reconstructed, visualize_encoder, visualize_decoder



parser = argparse.ArgumentParser(description = "Architecture combining CNN and SVM on (Fashion) MNIST")

parser.add_argument('--mode', default = 'train')
parser.add_argument('--loss_type', type = str, default = 'softmax',
                    help = "Do you want to use softmax or svm as the loss function? (Default: Softmax)")
parser.add_argument('--data_type', type = str, default = 'MNIST',
                    help = "Data type to train or test: MNIST or Fashion_MNIST available")
parser.add_argument('--batch_size', type = int, default = 128, metavar = 'N',
                    help = "Batch size to be used for training (default: 128)")
parser.add_argument('--epochs', type = int, default = 3, metavar = 'N',
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

if __name__ == '__main__':
    args = parser.parse_args()
    
    # whether to use cuda
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    
    # set device (gpu/cpu) according to args.use_cuda
    device = torch.device("cuda" if args.use_cuda else "cpu")
    torch.manual_seed(args.seed)
    
    # load dataset: MNIST or Fashion_MNIST
    assert args.data_type in ('MNIST', 'Fashion_MNIST'), "Choose data_type between MNIST or Fashion_MNIST"
    dataset_train, dataset_test = load_datasets(data_type = args.data_type, batch_size = args.batch_size)
    
    # Model path(save or load)
    if (not os.path.exists(args.model_pth)) & (args.mode == 'train'):
        print("Directory to save the model does not exist. Make one? [y | n]")
        dir_yn = str(input())
        if dir_yn == 'y':
            os.makedirs(args.model_pth)
            save_load_flag = True

        elif dir_yn == 'n':
            print("Please check directory. Not able to save model this time!")
            save_load_flag = False

        else:
            raise Exception("Input should be either y or n")

    else:
        save_load_flag = True

    if args.use_cuda:
        model = CNN().cuda()
    else:
        model = CNN()   

    if args.mode == 'train':
        train_losses, train_accs = train(model = model,
                                         trainset = dataset_train,
                                         train_type = args.loss_type,
                                         lr = 1e-3,
                                         epochs = args.epochs,
                                         log_interval = args.log_interval,
                                         cuda = args.use_cuda,
                                         save = save_load_flag,
                                         save_path = args.model_pth)

        save_plot(args.plot_pth, args.loss_type, train_accs, train_losses)

    else:
        # Test mode
        # Check model existence
        if not save_load_flag:
            raise Exception("No path set to load model from!")

        else:
            # load model
            model_name = os.path.join(args.model_pth, 'model_cnn_softmax.pkl') if args.loss_type == 'softmax' else os.path.join(args.model_pth, 'model_cnn_svm.pkl')
            model = torch.load(model_name)

        test_acc = test(model = model,
                        test_type = args.loss_type,
                        testset = dataset_test,
                        cuda = args.use_cuda,
                        model_path = args.model_pth)
        
        print("Test Accuracy: {:.4f}".format(test_acc))