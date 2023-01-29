import torch
import torch.nn as nn

from utils.utils import init_weights

device = None


class Pix2PixGAN(nn.Module):
    def __init__(self, generator, discriminator, init_std=0.02, pool_size=50):
        super(Pix2PixGAN, self).__init__()
        self.gen = generator()
        self.dis = discriminator()
        
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.gan_loss = nn.MSELoss()
        self.gen_loss = nn.L1Loss()

        init_weights(self, init_std)
        
    def fit_epoch(self, loader, max_iter=None):
        avg_loss_g = 0
        avg_loss_d = 0
        
        for i, (X_real, Y_real) in enumerate(loader):
            if max_iter is not None and i * loader.batch_size > max_iter:
                break

            X_real = X_real.to(device)
            Y_real = Y_real.to(device)
            
            Y_fake = self.gen(X_real)

            self.dis.requires_grad_(False)
            
            self.gen_optim.zero_grad()
            
            XY_fake = torch.cat((X_real, Y_fake), 1)            
            preds = self.dis(XY_fake)
            loss_g = self.gan_loss(preds, torch.ones_like(preds, device=device))
            loss_g += self.gen_loss(Y_fake, Y_real)
            
            loss_g.backward()
            self.gen_optim.step()
            
            avg_loss_g += loss_g.item() / len(loader)
            
            self.dis.requires_grad_(True)

            self.dis_optim.zero_grad()
            loss_d = self.gan_loss(self.dis(torch.cat((X_real, Y_real), 1)), torch.ones_like(preds, device=device)) + \
                     self.gan_loss(self.dis(torch.cat((X_real, Y_fake.detach()), 1)), torch.zeros_like(preds, device=device))
            loss_d *= 0.5
            loss_d.backward()
            self.dis_optim.step()
            
            avg_loss_d += loss_d.item() / len(loader)
            
        return avg_loss_g, avg_loss_d
    
    def forward(self, X):
        return self.gen(X)
