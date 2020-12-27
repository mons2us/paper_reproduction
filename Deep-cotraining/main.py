import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from random import shuffle
import os
import math
import pickle
import argparse
import random
import yaml
import logging

from tqdm import tqdm
from tensorboardX import SummaryWriter 
from model import co_train_classifier
from advertorch.attacks import GradientSignAttack

from utils import loss_cot, loss_diff, loss_sup
from dataset import load_dataset


# ---------------------
#   Settings/options
# ---------------------
np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def load_model_configs(args):
    with open(args.config_dir, encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs


def set_seed(configs):
    seed = configs['train']['SEED']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, epoch):
    """cosine scheduling"""
    epoch = epoch + 1
    lr = args.base_lr*(1.0 + math.cos((epoch-1)*math.pi/args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lamda(epoch):
    epoch = epoch + 1
    global lambda_cot
    global lambda_diff
    lambda_cot = lambda_cot_max*math.exp(-5*(1-epoch/args.warm_up)**2) if epoch <= args.warm_up else lambda_cot_max
    lambda_diff = lambda_diff_max*math.exp(-5*(1-epoch/args.warm_up)**2) if epoch <= args.warm_up else lambda_diff_max


def checkpoint(epoch, option):
    # Save checkpoint.
    logger.info(f'Saving checkpoint at epoch: {epoch}')
    state = {
        'net1': net1,
        'net2': net2,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state':torch.cuda.get_rng_state(),
        'np_state': np.random.get_state(), 
        'random_state': random.getstate()
    }
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if(option=='best'):
        torch.save(state, './'+ args.checkpoint_dir +'/ckpt.best.' + str(args.seed))
    else:
        torch.save(state, './'+ args.checkpoint_dir +'/ckpt.last.' + str(args.seed))


def train(net1, net2, args, U_batch_size, step_size, *loaders):
    
    net1 = net1
    net2 = net2
    global writer
    
    tot_epoch = args.epochs
    batch_size = args.batchsize
    U_batch_size = U_batch_size
    step = step_size
    
    S_loader1 = loaders[0]
    S_loader2 = loaders[1]
    U_loader = loaders[2]
    testloader = loaders[3]
    
    params = list(net1.parameters()) + list(net2.parameters())
    optimizer = optim.SGD(params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.decay)
    
    for e in range(tot_epoch):
        
        logger.info(f'epoch: {e+1}')
        
        adjust_learning_rate(optimizer, e)
        adjust_lamda(e)
        
        # Error(loss) initialization
        total_S1, total_S2, total_U1, total_U2 = 0, 0, 0, 0
        train_correct_S1, train_correct_S2, train_correct_U1, train_correct_U2 = 0, 0, 0, 0
        running_loss = 0.0
        ls, lc, ld = 0.0, 0.0, 0.0
        
        # create iterator for b1, b2, bu
        S_iter1 = iter(S_loader1)
        S_iter2 = iter(S_loader2)
        U_iter = iter(U_loader)
        
        for i in tqdm(range(step)):
            
            net1.train()
            net2.train()
            
            inputs_S1, labels_S1 = S_iter1.next()
            inputs_S2, labels_S2 = S_iter2.next()
            inputs_U, labels_U = U_iter.next()

            inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()
            inputs_S2, labels_S2 = inputs_S2.cuda(), labels_S2.cuda()
            inputs_U = inputs_U.cuda()    

            logit_S1 = net1(inputs_S1)
            logit_S2 = net2(inputs_S2)
            logit_U1 = net1(inputs_U)
            logit_U2 = net2(inputs_U)

            _, predictions_S1 = torch.max(logit_S1, 1)
            _, predictions_S2 = torch.max(logit_S2, 1)

            # pseudo labels of U 
            _, predictions_U1 = torch.max(logit_U1, 1)
            _, predictions_U2 = torch.max(logit_U2, 1)

            # fix batchnorm
            net1.eval()
            net2.eval()
            
            #generate adversarial examples
            perturbed_data_S1 = adversary1.perturb(inputs_S1, labels_S1)
            perturbed_data_U1 = adversary1.perturb(inputs_U, predictions_U1)

            perturbed_data_S2 = adversary2.perturb(inputs_S2, labels_S2)
            perturbed_data_U2 = adversary2.perturb(inputs_U, predictions_U2)
            
            net1.train()
            net2.train()

            perturbed_logit_S1 = net1(perturbed_data_S2)
            perturbed_logit_S2 = net2(perturbed_data_S1)

            perturbed_logit_U1 = net1(perturbed_data_U2)
            perturbed_logit_U2 = net2(perturbed_data_U1)

            # zero out the parameter gradients
            optimizer.zero_grad()
            net1.zero_grad()
            net2.zero_grad()
            
            Loss_sup = loss_sup(logit_S1, logit_S2, labels_S1, labels_S2)
            Loss_cot = loss_cot(logit_U1, logit_U2, U_batch_size)
            Loss_diff = loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2, logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2, batch_size)
            
            total_loss = Loss_sup + lambda_cot*Loss_cot + lambda_diff*Loss_diff
            total_loss.backward()
            optimizer.step()


            train_correct_S1 += np.sum(predictions_S1.cpu().numpy() == labels_S1.cpu().numpy())
            total_S1 += labels_S1.size(0)

            train_correct_U1 += np.sum(predictions_U1.cpu().numpy() == labels_U.cpu().numpy())
            total_U1 += labels_U.size(0)

            train_correct_S2 += np.sum(predictions_S2.cpu().numpy() == labels_S2.cpu().numpy())
            total_S2 += labels_S2.size(0)

            train_correct_U2 += np.sum(predictions_U2.cpu().numpy() == labels_U.cpu().numpy())
            total_U2 += labels_U.size(0)
            
            running_loss += total_loss.item()
            ls += Loss_sup.item()
            lc += Loss_cot.item()
            ld += Loss_diff.item()
            
            # using tensorboard to monitor loss and acc
            writer.add_scalars('data/loss', {'loss_sup': Loss_sup.item(), 'loss_cot': Loss_cot.item(), 'loss_diff': Loss_diff.item()}, (e)*(step)+i)
            writer.add_scalars('data/training_accuracy', {'net1 acc': 100. * (train_correct_S1+train_correct_U1) / (total_S1+total_U1), 'net2 acc': 100. * (train_correct_S2+train_correct_U2) / (total_S2+total_U2)}, (e)*(step)+i)
            if (i+1)%50 == 0:
                # print statistics
                tqdm.write('net1 training acc: %.3f%% | net2 training acc: %.3f%% | total loss: %.3f | loss_sup: %.3f | loss_cot: %.3f | loss_diff: %.3f  '
                    % (100. * (train_correct_S1+train_correct_U1) / (total_S1+total_U1), 100. * (train_correct_S2+train_correct_U2) / (total_S2+total_U2), running_loss/(i+1), ls/(i+1), lc/(i+1), ld/(i+1)))
        
        # Test after each epoch
        test(net1, net2, e, testloader)
        
        # Save model
        if e % 5 == 0:
            checkpoint(e, 'last')


def test(net1, net2, epoch, testloader):
    
    net1.eval()
    net2.eval()
    
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs1 = net1(inputs)
            predicted1 = outputs1.max(1)
            total1 += targets.size(0)
            correct1 += predicted1[1].eq(targets).sum().item()

            outputs2 = net2(inputs)
            predicted2 = outputs2.max(1)
            total2 += targets.size(0)
            correct2 += predicted2[1].eq(targets).sum().item()

    print('\nnet1 test acc: %.3f%% (%d/%d) | net2 test acc: %.3f%% (%d/%d)'
        % (100.*correct1/total1, correct1, total1, 100.*correct2/total2, correct2, total2))

    writer.add_scalars('data/testing_accuracy', {'net1 acc': 100.*correct1/total1, 'net2 acc': 100.*correct2/total2}, epoch)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Deep Co-Training for Semi-Supervised Image Recognition')
    
    parser.add_argument('--batchsize', default=100, type=int)
    parser.add_argument('--lambda_cot_max', default=10, type=int)
    parser.add_argument('--lambda_diff_max', default=0.5, type=float)
    parser.add_argument('--seed', default=2020011135, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--warm_up', default=80.0, type=float)
    parser.add_argument('--momentum', default=0.85, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--epsilon', default=0.02, type=float)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--dataset_dir', default='./data', type=str)
    parser.add_argument('--config_dir', default='./config.yaml', type=str)
    parser.add_argument('--tensorboard_dir', default='./tensorboard', type=str)
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str)
    parser.add_argument('--base_lr', default=0.035, type=float)
    parser.add_argument('--resume', action='store_true', help='If added, resume from checkpoint')
    parser.add_argument('--data_type', default='cifar10', type=str, help='Only cifar10 available')
    
    args = parser.parse_args()
    
    # load model configs
    configs = load_model_configs(args)

    # set seeds for reproduction
    set_seed(configs)

    # basic settings
    start_epoch = 0
    end_epoch = args.epochs
    class_num = args.num_class 
    batch_size = args.batchsize


    lambda_cot_max = args.lambda_cot_max
    lambda_diff_max = args.lambda_diff_max
    lambda_cot = 0.0
    lambda_diff = 0.0
    best_acc = 0.0

    # -------------------
    #    Load Dataset 
    # -------------------
    trainset, trainloader, testset, testloader, S_batch_size, U_batch_size, S_idx, U_idx, dataiter, trainlist, step_size = load_dataset(args.data_type, args.dataset_dir, args.batchsize, args.num_class)

    # -------------------
    #  Load/Build Model
    # -------------------
    if args.resume:
        logger.info(f'Resuming from checkpoint')
        assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found!'
        
        checkpoint = torch.load('./'+ args.checkpoint_dir + '/ckpt.last.' + args.sess + '_' + str(args.seed))
        
        net1 = checkpoint['net1']
        net2 = checkpoint['net2']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['np_state'])
        random.setstate(checkpoint['random_state'])

        with open("cifar10_labelled_index.pkl", "rb") as fp:
            S_idx = pickle.load(fp)

        with open("cifar10_unlabelled_index.pkl", "rb") as fp:
            U_idx = pickle.load(fp)
            
    else:
        logger.info('Building model...')
        start_epoch = 0
        net1 = co_train_classifier()
        net2 = co_train_classifier()

        for i in range(len(trainset)):
            inputs, labels = dataiter.next()
            trainlist[labels].append(i)

        for i in range(class_num):
            shuffle(trainlist[i])
            S_idx = S_idx + trainlist[i][0:400]
            U_idx = U_idx + trainlist[i][400:]

        # save the indexes in case we need the exact ones to resume
        with open("cifar10_labelled_index.pkl","wb") as fp:
            pickle.dump(S_idx,fp)

        with open("cifar10_unlabelled_index.pkl","wb") as fp:
            pickle.dump(U_idx,fp)

    net1.cuda()
    net2.cuda()
    
    # If using more than 1 gpu, activate codes under;
    #net1 = torch.nn.DataParallel(net1)
    #net2 = torch.nn.DataParallel(net2)
    
    logger.info(f'Using {torch.cuda.device_count()} GPUs.')

    S_sampler = torch.utils.data.SubsetRandomSampler(S_idx)
    U_sampler = torch.utils.data.SubsetRandomSampler(U_idx)

    S_loader1 = torch.utils.data.DataLoader(trainset, batch_size=S_batch_size, sampler=S_sampler)
    S_loader2 = torch.utils.data.DataLoader(trainset, batch_size=S_batch_size, sampler=S_sampler)
    U_loader = torch.utils.data.DataLoader(trainset, batch_size=U_batch_size, sampler=U_sampler)

    # adversary object for net 1: p_1(x)
    adversary1 = GradientSignAttack(
    net1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon, clip_min=-math.inf, clip_max=math.inf, targeted=False)

    # adversary object for net 2: p_2(x)
    adversary2 = GradientSignAttack(
    net2, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon, clip_min=-math.inf, clip_max=math.inf, targeted=False)

    # tensorboard
    writer = SummaryWriter(args.tensorboard_dir)
    
    # -----------------------
    #     Start training
    # -----------------------
    train(net1, net2, args, U_batch_size, step_size, S_loader1, S_loader2, U_loader, testloader)
    
    # close tensorboard writer
    writer.export_scalars_to_json('./'+ args.tensorboard_dir + 'output.json')
    writer.close()