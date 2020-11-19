import torch
from torch import nn, optim
from model import Generator, Discriminator
from torch.nn import init
from torch.autograd import Variable
import torchvision.utils as tv_utils
from tqdm import tqdm 
import matplotlib.pyplot as plt

# ---------------
#   AnoGAN Loss
# ---------------
def residual(x, z):
    _loss = torch.sum(torch.abs(x - z))
    return _loss

def anomaly_score(x, z, discriminator):
    
    dis_real, mediate_real = discriminator(x)
    dis_fake, mediate_fake = discriminator(z)

    ano_loss = residual(x, z)
    dis_loss = residual(mediate_real, mediate_fake)

    _anomaly_score = 0.9 * ano_loss + 0.1 * dis_loss

    return _anomaly_score


def train_anogan(model_g,
                 model_d,
                 inf_data,
                 inf_size,
                 lr = 0.0002,
                 max_iter = 500,
                 cuda = True):

    device = torch.device("cuda" if cuda else "cpu")

    model_g = model_g.to(device)
    model_d = model_d.to(device)

    # -----------------------------------
    #    Fix generator & discriminator
    # -----------------------------------
    model_g.eval()
    model_d.eval()

    z = init.normal_(torch.Tensor(inf_size, 100), mean = 0, std = 0.1).to(device)
    z = Variable(z, requires_grad = True)

    z_optimizer = optim.Adam([z], lr = lr)

    for i in tqdm(range(max_iter)):
        gen_fake = model_g(z)
        ano_loss = anomaly_score(Variable(inf_data).to(device), gen_fake, model_d)
        ano_loss.backward()

        z_optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Anomaly Score: {ano_loss.data}")

    # ----------------------
    #     Plot results
    # ----------------------
    tv_utils.save_image(gen_fake.data[:], f"./result/anomaly_gen.png", nrow = 4)
    tv_utils.save_image(inf_data.data[:], f"./result/anomaly_real.png", nrow = 4)