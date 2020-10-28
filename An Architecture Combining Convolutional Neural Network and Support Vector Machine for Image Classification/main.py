import argparse
import torch, torchvision
from dataset import load_datasets
from model import CNN
from train import train
from test import test
import matplotlib.pyplot as plt

#from visualize import visualize_reconstructed, visualize_encoder, visualize_decoder



parser = argparse.ArgumentParser(description = "Architecture combining CNN and SVM on (Fashion) MNIST")

parser.add_argument('--mode', default = 'train')
parser.add_argument('--train_type', type = str, default = 'softmax',
                    help = "Do you want to use softmax or svm as the loss function?")
parser.add_argument('--data_type', type = str, default = 'MNIST',
                    help = "Data type to train or test: MNIST or Fashion_MNIST available")
parser.add_argument('--batch_size', type = int, default = 128, metavar = 'N',
                    help = "Batch size to be used for training (default: 64)")
parser.add_argument('--epochs', type = int, default = 3, metavar = 'N',
                    help = "Number of epochs to be used for training (default: 10)")
parser.add_argument('--use_cuda', action='store_true', default = True,
                    help = "Whether to use cuda in training. If you don't want to use cuda, set this to False")
parser.add_argument('--seed', type = int, default = 2020, metavar = 'S',
                    help = "Random seed (default: 2020)")
parser.add_argument('--log_interval', type = int, default = 100, metavar = 'N',
                    help = "Logging interval in training (default: 10)")
parser.add_argument('--model_pth', type = str, default = './model/trained_cnn_svm.pkl',
                    help = "Path for the model to be saved or loaded from")


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
    
    if args.use_cuda:
        model = CNN().cuda()    

    if args.mode == 'train':
        train_losses, train_accs = train(model = model,
                                         trainset = dataset_train,
                                         train_type = args.train_type,
                                         lr = 1e-3,
                                         epochs = args.epochs,
                                         log_interval = args.log_interval,
                                         cuda = args.use_cuda,
                                         save_path = args.model_pth)
        
        # 학습 결과 확인
        plt.figure(figsize = (12, 6))
        plt.plot(train_accs)
        plt.plot(train_losses)
        plt.show()
    
    else:
        model = torch.load(args.model_pth)
        
        # test
        # MNIST testset에 대한 loss 출력
        if args.mode == 'test':
            test_loss = test(model = model,
                             test_type = 'softmax',
                             testset = dataset_test,
                             cuda = args.use_cuda)
            
            print("Test loss: {:.4f}".format(test_loss))
            
        # visualize
        # MNIST testset에 대해 vae 복원 시각화
        #  * 학습된 latent_dim = 2가 아니면 복원결과/latent space
        #  * 학습된 latent_dim = 2이면 복원결과/latent space/decoded 시각화
        # elif args.mode == 'visualize':
        #     visualize_reconstructed(model = model, dataset = dataset_test)
        #     visualize_encoder(model = model, dataset = dataset_test)
        #     if args.latent_dims == 2:
        #         visualize_decoder(model = model, cuda = dataset_test) # 2차원일 때 표현이 가능


