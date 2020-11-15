import os
import torch
from torch import nn, optim
from torch.nn import functional as F, init
from torch.autograd import Variable
import torchvision.utils as tv_utils

from tqdm import tqdm, tnrange

# 학습 모델 정의
def train(model_g,
          model_d,
          trainset,
          lr = 1e-3,
          epochs = 10,
          log_interval = 10,
          cuda = True,
          save_path = './model'):
    
    model_g, model_d = model_g, model_d
    trainset = trainset
    
    device = torch.device("cuda" if cuda else "cpu")
    #print("Using: {}".format(device))
    
    # Loss function
    loss_function = nn.MSELoss()
    
    model_g.train()
    model_d.train()
    
    # optimizers
    g_optimizer = optim.Adam(
        model_g.parameters(),
        lr = lr
    )

    d_optimizer = optim.Adam(
        model_g.parameters(),
        lr = lr
    )

    for epoch in tnrange(epochs, desc = 'Training Process'):
        
        for batch_idx, (images, labels) in enumerate(trainset):
            
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            ##################
            ## Generator #####
            ##################

            # Generate using random noise z
            z = init.normal_(torch.Tensor(len(images), 100), mean = 0, std = 0.1)
            z = Variable(z).to(device)
            #print(f"z: {z.size()}")

            generated = model_g(z)
            #print(f"generated: {generated.size()}")
            discrm, med_feature = model_d(generated)
            
            gen_loss = torch.sum(loss_function(discrm, torch.ones_like(discrm)))
            gen_loss.backward()
            g_optimizer.step()


            ##################
            ## Discriminator #
            ##################
            d_optimizer.zero_grad()

            z = init.normal_(torch.Tensor(len(images), 100), mean = 0, std = 0.1)
            z = Variable(z).to(device)

            generated = model_g(z)
            discrm, med_feature = model_d(generated)
            discrm_real, med_feature_real = model_d(images)
            dis_loss = torch.sum(loss_function(discrm, torch.zeros_like(discrm))) + torch.sum(loss_function(discrm_real, torch.ones_like(discrm_real)))

            dis_loss.backward()
            d_optimizer.step()
            
            # loss at training
            if batch_idx % 50 == 0:
                torch.save(model_g.state_dict(), './model/generator.pkl')
                torch.save(model_d.state_dict(), './model/discriminator.pkl')
                print(f"Gen_loss: {gen_loss}, Dis_loss: {dis_loss}")


        print("======= Epoch: {}  Average Gen_loss: {:.4f}  Average Dis_loss: {:.4f} =======\n".format(
            epoch + 1,
            gen_loss / len(trainset),
            dis_loss / len(trainset)
        ))
        
        tv_utils.save_image(generated.data[0:25], f"./model/gen_{epoch}.png", nrow = 5)

    return gen_loss, dis_loss