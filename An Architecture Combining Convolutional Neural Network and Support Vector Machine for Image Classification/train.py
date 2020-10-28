import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tnrange

from utils import loss_softmax, loss_svm

# 학습 모델 정의
def train(model,
          trainset,
          train_type = 'softmax',
          lr = 1e-3,
          epochs = 10,
          log_interval = 10,
          cuda = True,
          save_path = './model/trained_softmax.pkl'):
    
    model = model
    trainset = trainset
    
    device = torch.device("cuda" if cuda else "cpu")
    #print("Using: {}".format(device))
    
    train_losses = []
    train_loss = 0

    train_accs = []
    train_acc = 0
    
    model.train()
    
    # optimizer는 Adam 사용
    optimizer = optim.Adam(
        model.parameters(),
        lr = lr
    )
    
    # loss function: softmax or svm?
    loss_function = loss_softmax() if train_type == 'softmax' else loss_svm()


    for epoch in tnrange(epochs, desc = 'Training Process'):
        
        for batch_idx, (images, labels) in enumerate(trainset):
            
            images = Variable(images).to(device) if cuda else Variable(images)
            labels = Variable(labels).to(device) if cuda else Variable(labels)

            optimizer.zero_grad()
            
            pred = model(images)
            
            # 정확도 계산
            pred_label = torch.argmax(pred, axis = 1)
            
            pred_tf = torch.sum(torch.eq(pred_label, labels))
            pred_acc = pred_tf/pred.shape[0]

            # Loss 계산
            loss = loss_function(pred, labels)
            loss.backward() # Backpropagation
            
            # loss at training
            train_losses.append(loss.item() / len(images)) # 배치별로 backprop하여 loss를 loss list에 담는다.
            train_loss += loss.item()

            # acc at training
            train_accs.append(pred_acc)
            train_acc += pred_acc

            optimizer.step()
            
            if batch_idx % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.4f}  Acc: {:.4f}".format(
                    epoch + 1,
                    batch_idx * len(images),
                    len(trainset.dataset),
                    batch_idx * 100. / len(trainset),
                    loss.item(),
                    pred_acc
                ))

        print(len(trainset), len(trainset.dataset))
        print("======= Epoch: {}  Average Loss: {:.4f}  Average Acc: {:.4f} =======\n".format(
            epoch + 1,
            train_loss / len(trainset),
            train_acc / len(trainset)
        ))
        
        train_loss = 0
        train_acc = 0
    
    # save model
    if not os.path.exists(os.path.dirname(save_path)):
        print("Directory to save the model does not exist. Make one? [y | n]")
        if str(input()) == 'y':
            os.makedirs(os.path.dirname(save_path))
        elif str(input()) == 'n':
            print("Then pleas check directory!")
        else:
            print("Input should be either y or n")
    
    torch.save(model, save_path)
    
    return train_losses, train_accs
