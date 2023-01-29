import torch
from IPython.display import clear_output
from tqdm import tqdm

from utils.plot import plot_images

device = None


def fit(gan, train_loader_A, train_loader_B, test_loader_A, test_loader_B, epochs, max_iter=None):
    gan.train()

    # lr линейно затухает до 0 после половины эпох
    rule = lambda epoch: 1.0 - max(0, epoch - epochs // 2) / (epochs // 2)
    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gan.gen_optim, lr_lambda=rule)
    dis_scheduler = torch.optim.lr_scheduler.LambdaLR(gan.dis_optim, lr_lambda=rule)
    
    test_A = next(iter(test_loader_A)).to(device)
    test_B = next(iter(test_loader_B)).to(device)

    losses_g = []
    losses_d = []
    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        avg_loss_g, avg_loss_d = gan.fit_epoch(train_loader_A, train_loader_B, max_iter)

        gen_scheduler.step()
        dis_scheduler.step()

        losses_g.append(avg_loss_g)
        losses_d.append(avg_loss_d)
            
        clear_output(wait=True)
        gan.eval()
        with torch.no_grad():
            fake_A, fake_B = gan(test_A, test_B)
            plot_images([*test_A.cpu(), *test_B.cpu(), *fake_B.detach().cpu(), *fake_A.detach().cpu()], 
                        w=test_loader_A.batch_size*2, h=2)
        gan.train()
        
        pbar.set_description(f'Epoch: {epoch+1}. Gen loss: {avg_loss_g:.8f}. Discr loss: {avg_loss_d:.8f}')
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        
    gan.pool_A.images.clear()
    gan.pool_B.images.clear()
    
    return losses_g, losses_d


def fit_pix2pix(gan, train_loader, test_loader, epochs, max_iter=None):
    gan.train()

    # lr линейно затухает до 0 после половины эпох
    rule = lambda epoch: 1.0 - max(0, epoch - epochs // 2) / (epochs // 2)
    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gan.gen_optim, lr_lambda=rule)
    dis_scheduler = torch.optim.lr_scheduler.LambdaLR(gan.dis_optim, lr_lambda=rule)
    
    test_real = [i.to(device) for i in next(iter(test_loader))]

    losses_g = []
    losses_d = []
    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        avg_loss_g, avg_loss_d = gan.fit_epoch(train_loader, max_iter)

        gen_scheduler.step()
        dis_scheduler.step()

        losses_g.append(avg_loss_g)
        losses_d.append(avg_loss_d)
            
        clear_output(wait=True)
        gan.eval()
        with torch.no_grad():
            test_fake = gan(test_real[0])
            plot_images([*test_real[0].cpu(), *test_real[1].cpu(), *test_fake.detach().cpu()], 
                        w=test_loader.batch_size, h=3)
        gan.train()
        
        pbar.set_description(f'Epoch: {epoch+1}. Gen loss: {avg_loss_g:.8f}. Discr loss: {avg_loss_d:.8f}')
    
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    return losses_g, losses_d
