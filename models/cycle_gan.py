import torch
import torch.nn as nn
from itertools import chain

from utils.utils import init_weights
from utils.image_pool import ImagePool

device = None


class CycleGAN(nn.Module):
    def __init__(self, generator, discriminator, cycle_lambda=10, idt_lambda=1, init_std=0.02, pool_size=50):
        super(CycleGAN, self).__init__()
        self.gen_A = generator()
        self.gen_B = generator()
        self.dis_A = discriminator()
        self.dis_B = discriminator()
        self.cycle_lambda = cycle_lambda
        self.idt_lambda = idt_lambda
        
        self.gen_optim = torch.optim.Adam(chain(self.gen_A.parameters(), self.gen_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.dis_optim = torch.optim.Adam(chain(self.dis_A.parameters(), self.dis_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        
        self.gan_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.idt_loss = nn.L1Loss()

        self.pool_A = ImagePool(pool_size)
        self.pool_B = ImagePool(pool_size)

        init_weights(self, init_std)
        
    def fit_epoch(self, loader_A, loader_B, max_iter=None):
        avg_loss_g = 0
        avg_loss_d = 0
        
        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            if max_iter is not None and i * loader_A.batch_size > max_iter:
                break

            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            fake_A = self.gen_A(real_B)
            fake_B = self.gen_B(real_A)
            
            rec_A = self.gen_A(fake_B)
            rec_B = self.gen_B(fake_A)

            idt_A = self.gen_A(real_A)
            idt_B = self.gen_B(real_B)

            self.dis_A.requires_grad_(False)
            self.dis_B.requires_grad_(False)
            
            self.gen_optim.zero_grad()
            
            loss_g = self.cycle_lambda * (self.cycle_loss(rec_A, real_A) + self.cycle_loss(rec_B, real_B))
            loss_g += self.idt_lambda * (self.idt_loss(idt_A, real_A) + self.idt_loss(idt_B, real_B))
            
            preds = self.dis_A(fake_A)
            loss_g += self.gan_loss(preds, torch.ones_like(preds, device=device))
            preds = self.dis_B(fake_B)
            loss_g += self.gan_loss(preds, torch.ones_like(preds, device=device))
            
            loss_g.backward()
            self.gen_optim.step()
            
            avg_loss_g += loss_g.item() / min(len(loader_A), len(loader_B))
            
            self.dis_A.requires_grad_(True)
            self.dis_B.requires_grad_(True)

            self.dis_optim.zero_grad()
            loss_d = self.gan_loss(self.dis_A(real_A), torch.ones_like(preds, device=device)) + \
                     self.gan_loss(self.dis_A(self.pool_A.query(fake_A).detach()), torch.zeros_like(preds, device=device)) + \
                     self.gan_loss(self.dis_B(real_B), torch.ones_like(preds, device=device)) + \
                     self.gan_loss(self.dis_B(self.pool_B.query(fake_B).detach()), torch.zeros_like(preds, device=device))
            loss_d *= 0.5
            loss_d.backward()
            self.dis_optim.step()
            
            avg_loss_d += loss_d.item() / min(len(loader_A), len(loader_B))
            
        return avg_loss_g, avg_loss_d
    
    def forward(self, A, B):
        return self.gen_A(B), self.gen_B(A)
