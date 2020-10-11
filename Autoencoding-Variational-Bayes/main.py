import argparse
import torch, torchvision
from dataset import load_datasets
from model import VAE
from train import loss_function, train
from test import test
import matplotlib.pyplot as plt

from visualize import visualize_reconstructed, visualize_encoder, visualize_decoder


parser = argparse.ArgumentParser(description = "Variational Autoencoder on MNIST")

parser.add_argument('--mode', default = 'train')
parser.add_argument('--batch_size', type = int, default = 64, metavar = 'N',
                    help = "Batch size to be used for training (default: 64)")
parser.add_argument('--epochs', type = int, default = 10, metavar = 'N',
                    help = "Number of epochs to be used for training (default: 10)")
parser.add_argument('--latent_dims', type = int, default = 20, metavar = 'N',
                    help = "Size of VAE's latent dimension (default: 20)")
parser.add_argument('--use_cuda', action='store_true', default = True,
                    help = "Whether to use cuda in training. If you don't want to use cuda, set this to False")
parser.add_argument('--seed', type = int, default = 2020, metavar = 'S',
                    help = "Random seed (default: 2020)")
parser.add_argument('--log_interval', type = int, default = 100, metavar = 'N',
                    help = "Logging interval in training (default: 10)")
parser.add_argument('--model_pth', type = str, default = './model/trained_vae.pkl',
                    help = "Path for the model to be saved or loaded from")


if __name__ == '__main__':
    args = parser.parse_args()
    
    # whether to use cuda
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    
    # set device (gpu/cpu) according to args.use_cuda
    device = torch.device("cuda" if args.use_cuda else "cpu")
    torch.manual_seed(args.seed)
    
    dataset_train, dataset_test = load_datasets(batch_size = args.batch_size)
    
    if args.use_cuda:
        model = VAE(latent_dim = args.latent_dims).cuda()
    else:
        model = VAE(latent_dim = args.latent_dims)
        
        
    if args.mode == 'train':
        train_losses = train(model = model,
                             trainset = dataset_train,
                             lr = 1e-3,
                             epochs = args.epochs,
                             log_interval = args.log_interval,
                             cuda = args.use_cuda,
                             save_path = args.model_pth)
        
        # 학습 결과 확인
        plt.figure(figsize = (12, 6))
        plt.plot(train_losses)
        plt.show()
    
    else:
        model = torch.load(args.model_pth)
        
        # test
        # MNIST testset에 대한 loss 출력
        if args.mode == 'test':
            test_loss = test(model = model,
                             testset = dataset_test,
                             cuda = args.use_cuda)
            
            print("Test loss: {:.4f}".format(test_loss))
            
        # visualize
        # MNIST testset에 대해 vae 복원 시각화
        #  * 학습된 latent_dim = 2가 아니면 복원결과/latent space
        #  * 학습된 latent_dim = 2이면 복원결과/latent space/decoded 시각화
        elif args.mode == 'visualize':
            visualize_reconstructed(model = model, dataset = dataset_test)
            visualize_encoder(model = model, dataset = dataset_test)
            if args.latent_dims == 2:
                visualize_decoder(model = model, cuda = dataset_test) # 2차원일 때 표현이 가능