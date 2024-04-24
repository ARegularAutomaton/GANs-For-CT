import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from physics.ct import CT
from transforms.rotate import Rotate
from utils.logger import LOG, get_timestamp
from utils.metric import cal_psnr, cal_mse, cal_psnr_complex
from model.generator import Generator
from model.discriminator import Discriminator
import datetime
from dataset.dataset import get_dataloader, get_datasets
from utils.weight_init import weights_init

######################################################################
# Inputs
# ------
# Batch size during training
real_batch_size = 10
noise_batch_size = 10

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5000

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# get data loader
dataloader, noise_dataloader = get_dataloader(image_size, real_batch_size, noise_batch_size)
real_set, fake_set = get_datasets(image_size, real_batch_size, noise_batch_size)

def plot_real_batch(dataloader):
     # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

######################################################################
# Implementation
# --------------

def init_net():
    ######################################################################
    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    ######################################################################
    # Create the Reconstructor
    netR = Generator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netR = nn.DataParallel(netR, list(range(ngpu)))

    ######################################################################
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)

    return netG, netR, netD

# init networks
netG, netR, netD = init_net()

######################################################################
# Loss Functions and Optimizers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def loss_and_optimisers():
     # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()
    mse = nn.MSELoss()
    physics = CT(img_width=image_size, radon_view=90, noise_model={'noise_type':'g','sigma':0.1,})
    transform = Rotate(n_trans=1)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerR = optim.Adam(netR.parameters(), lr=lr, betas=(beta1, 0.999))
    
    return criterion, mse, physics, transform, optimizerG, optimizerR, optimizerD

# prepare loss and optimisers
criterion, mse, physics, transform, optimizerG, optimizerR, optimizerD = loss_and_optimisers()

def prepare_fixed_noise():
     # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = next(iter(noise_dataloader))[0].to(device)
    fixed_noise = physics.forw(fixed_noise, add_noise=True)
    fixed_noise = physics.pseudo(fixed_noise)
    # print(fixed_noise.shape)
    return fixed_noise

# fixed noise for visualising progress
fixed_noise = prepare_fixed_noise()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

######################################################################
# Training
# ~~~~~~~~
# Training Loop

def paths_and_logs():
    dt = datetime.datetime.now()
    now = "{}{}{}{}{}".format(dt.month, dt.day, dt.hour, dt.minute, dt.second)
    save_path = './ckp/{}_ei_{}_{}_sigma{}' \
                .format(now, physics.name,
                        physics.noise_model['noise_type'],
                        physics.noise_model['sigma']) + '_I0_{}'.format(int(physics.I0))
    os.makedirs(save_path, exist_ok=True)

    log = LOG(save_path, filename='training_loss',
                    field_name=['epoch', 'loss_G', 'loss_D', 'loss_mc', 'loss_eq',
                                'loss_eq', 'psnr'])
    return now, save_path, log

# create paths and log objects
now, save_path, log = paths_and_logs()

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    loss_G_seq, loss_D_seq, loss_mc_seq, loss_eq_seq, loss_seq, psnr_seq = [], [], [], [], [], []
    real_dataloader = torch.utils.data.DataLoader(real_set, batch_size=real_batch_size, shuffle=True)
    fake_dataloader = torch.utils.data.DataLoader(fake_set, batch_size=real_batch_size, shuffle=True)

    for i, data in enumerate(dataloader):
        for index, batch in enumerate(noise_dataloader):
            if index == i:
                noise_batch = batch
                break

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()

        real = data[0].to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = netD(real).view(-1)

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        #-----------------------------
        # Train with all-fake batch
        noise_batch = noise_batch[0].to(device)
        z = physics.forw(noise_batch, add_noise=True)
        z = physics.pseudo(z).to(device)
        z = (z - torch.min(z)) / (torch.max(z) - torch.min(z))
        
        # Generate fake image batch with G
        fake = netG(z)
        b_size = fake.size(0)
        label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

        # Classify fake batch with D
        output = netD(fake.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)

        # Calculate G's loss based on this output
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()
        errG.backward()
        
        #-------------------------
        # minise distance between generated and latent variable
        fake = netG(z)
        conformity_loss = mse(fake, real)
        conformity_loss.backward()

        #-------------------------
        # domain consistency
        meas0 = physics.forw(fake.detach())
        fbp_g = physics.pseudo(meas0)
        x1 = netG(fbp_g)
        loss_domain = mse(x1, fake)
        loss = loss_domain + conformity_loss
        loss.backward()

        #-------------------------
        # eq
        x2 = transform.apply(x1.detach())
        meas2 = physics.forw(x2)
        fbp_2 = physics.pseudo(meas2)
        x3 = netG(fbp_2)

        loss_eq = mse(x3, x2)
        loss_eq.backward()

        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # losses for all epoch and batch
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # losses in this epoch
        loss_G_seq.append(errG.item())
        loss_D_seq.append(errD.item())

        loss_eq_seq.append(loss_eq.item())

        loss_mc_seq.append(0)
        loss_seq.append(0)

        psnr_seq.append(cal_psnr(noise_batch, fake))
        
        if epoch % 5000 == 0 or (epoch == num_epochs-1):
            fake = torch.zeros(fixed_noise.shape)
            fake = ((netG(fixed_noise))).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=5))

        # save model
        if epoch > 0 and epoch % 10 == 0:
                    state = {'epoch': epoch,
                             'state_dict': netG.state_dict(),}
                             
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(state,
                               os.path.join(save_path, 'ckp_{}_EIGANLatent.pth.tar'.format(now)))
        
        iters += 1

    loss_closure = [np.mean(loss_G_seq), np.mean(loss_D_seq), np.mean(loss_mc_seq), np.mean(loss_eq_seq), np.mean(loss_seq)]
    loss_closure.append(np.mean(psnr_seq))
    log.record(epoch + 1, *loss_closure)
    print(
        '{}\tEpoch[{}/{}]\tG={:.4e}\tD={:.4e}\tmc={:.4e}\teq={:.4e}\tR={:.4e}\tpsnr={:.4f}'
        .format(get_timestamp(), epoch+1, num_epochs, *loss_closure))
log.close()

######################################################################
# Results
# -------

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.style.use('seaborn-v0_8')
plt.axis('off')
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

######################################################################
# **Visualization of G’s progression**

fig = plt.figure(figsize=(4,4))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
os.makedirs("gifs/", exist_ok=True)
ani.save(f'gifs/GAN{num_epochs}epochs.gif', writer='imagemagick')

######################################################################
# **Real Images vs. Fake Images**

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Pseudo Images")
plt.imshow(np.transpose(vutils.make_grid(fixed_noise.to(device)[:64], padding=5, normalize=True, nrow=5).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()