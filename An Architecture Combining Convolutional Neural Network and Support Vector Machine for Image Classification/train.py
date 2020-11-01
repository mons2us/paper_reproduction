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
          save = True,
          save_path = './model'):
    
    model = model
    trainset = trainset
    
    device = torch.device("cuda" if cuda else "cpu")
    #print("Using: {}".format(device))
    
    # Loss: Softmax or SVM?
    loss_function = loss_softmax() if train_type == 'softmax' else loss_svm()

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

    for epoch in tnrange(epochs, desc = 'Training Process'):
        
        for batch_idx, (images, labels) in enumerate(trainset):
            
            images = Variable(images).to(device) if cuda else Variable(images)
            labels = Variable(labels).to(device) if cuda else Variable(labels)

            optimizer.zero_grad()
            
            # w: weight of the last fully connected layer
            pred, w = model(images)
            
            # 정확도 계산
            pred_label = torch.argmax(pred, axis = 1)
            
            pred_tf = torch.sum(torch.eq(pred_label, labels))
            pred_acc = pred_tf/pred.shape[0]

            loss = loss_function(pred, labels, reg_term = w, device = device, current_batch_size = len(images))
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

        print("======= Epoch: {}  Average Loss: {:.4f}  Average Acc: {:.4f} =======\n".format(
            epoch + 1,
            train_loss / len(trainset),
            train_acc / len(trainset)
        ))
        
        train_loss = 0
        train_acc = 0
    
    if save:
        try:
            model_name = os.path.join(save_path, 'model_cnn_softmax.pkl') if train_type == 'softmax' else os.path.join(save_path, 'model_cnn_svm.pkl')
            torch.save(model, model_name)
        except Exception as e:
            print(f"Model saving failed from the following error: {e}")
    
    return train_losses, train_accs
