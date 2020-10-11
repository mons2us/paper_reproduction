import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tnrange

from utils import loss_function

# 학습 모델 정의
def train(model,
          trainset,
          lr = 1e-3,
          epochs = 10,
          log_interval = 10,
          cuda = True,
          save_path = './model/trained_vae.pkl'):
    
    model = model
    trainset = trainset
    
    device = torch.device("cuda" if cuda else "cpu")
    #print("Using: {}".format(device))
    
    train_losses = []
    train_loss = 0
    
    model.train()
    
    # optimizer는 Adam 사용
    optimizer = optim.Adam(
        model.parameters(),
        lr = lr
    )
    
    for epoch in tnrange(epochs, desc = 'Training Process'):
        
        for batch_idx, (images, _) in enumerate(trainset):
            
            images = Variable(images).to(device) if cuda else Variable(images)
            optimizer.zero_grad()
            
            reconstructed, mu, logvar = model(images)
            loss = loss_function(images, reconstructed, mu, logvar) # loss 계산
            loss.backward() # Backpropagation
            
            train_losses.append(loss.item() / len(images)) # 배치별로 backprop하여 loss를 loss list에 담는다.
            train_loss += loss.item()
            
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.4f}".format(
                    epoch + 1,
                    batch_idx * len(images),
                    len(trainset.dataset),
                    batch_idx * 100. / len(trainset),
                    loss.item() / len(images)
                ))
                
        print("======= Epoch: {}  Average Loss: {:.4f} =======".format(
            epoch + 1,
            train_loss / len(trainset.dataset)
        ))
        
        train_loss = 0
    
    # save model
    torch.save(model, save_path)
    
    return train_losses