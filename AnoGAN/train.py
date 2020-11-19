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
          batch_size = None,
          lr = None,
          epochs = 50,
          log_interval = 10,
          cuda = True,
          save_path = './model'):


    trainset = trainset
    
    device = torch.device("cuda" if cuda else "cpu")
    print("Using device: {}".format(device))

    model_g = model_g.to(device)
    model_d = model_d.to(device)


    # Loss function
    loss_function = nn.BCELoss()

    # optimizers
    gen_optimizer = optim.Adam(model_g.parameters(), lr = lr*4, betas = (0.6, 0.999))
    dis_optimizer = optim.Adam(model_d.parameters(), lr = lr, betas = (0.6, 0.999))

    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    for epoch in range(epochs):
        
        for batch_idx, (images, _) in enumerate(trainset):
            
            label_real = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad = False).to(device)
            label_fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad = False).to(device)

            images = Variable(images).to(device)

            # ----------------
            #     Generator
            # ----------------
            gen_optimizer.zero_grad()
            
            # Generate using random noise z
            z = Variable(init.normal_(torch.Tensor(batch_size, 100), mean = 0, std = 0.1)).to(device)
            
            gen_fake = model_g(z)
            dis_fake, mediate_fake = model_d(gen_fake)

            gen_loss = loss_function(dis_fake, label_real)
            
            gen_loss.backward(retain_graph = True)
            gen_optimizer.step()

            # -----------------
            #   Discriminator
            # -----------------
            dis_optimizer.zero_grad()

            dis_fake, mediate_fake = model_d(gen_fake.detach())
            dis_real, mediate_real = model_d(images)

            dis_real_loss = loss_function(dis_real, label_real)
            dis_fake_loss = loss_function(dis_fake, label_fake)
            
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            dis_loss.backward()
            dis_optimizer.step()

            # loss at training
            if batch_idx % 30 == 0:
                torch.save(model_g.state_dict(), './model/generator.pkl')
                torch.save(model_d.state_dict(), './model/discriminator.pkl')

                print(f"Gen_loss: {gen_loss.data}, Dis_loss: {dis_loss.data}")
                tv_utils.save_image(gen_fake.data[0:25], f"./result/gen_{epoch}_{batch_idx}.png", nrow = 5)
                

        print(f"======= {epoch+1}th epoch done training =======\n")
        # if epoch % 2 == 0:
        #     tv_utils.save_image(gen_fake.data[0:25], f"./result/gen_{epoch}.png", nrow = 5)
        
    return gen_loss, dis_loss