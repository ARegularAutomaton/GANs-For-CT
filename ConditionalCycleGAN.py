import os
import piq
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
from model.discriminator import Discriminator, ConditionalDiscriminator, NLayerDiscriminator
import datetime
from dataset.dataset import get_datasets
from utils.weight_init import weights_init

######################################################################
# Inputs
# ------
# Batch size during training
Y_batch_size = 10
X_batch_size = 10
assert X_batch_size == Y_batch_size

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

# Get Dataloader
Y_dataset, X_dataset = get_datasets(image_size, 1, 1)

def plot_real_batch(dataloader):
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
    # Create the Discriminator A
    netDA = NLayerDiscriminator(input_nc=1).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netDA = nn.DataParallel(netDA, list(range(ngpu)))

    netDA.apply(weights_init)

    # Create the Discriminator B
    netDB = NLayerDiscriminator(input_nc=1).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netDB = nn.DataParallel(netDB, list(range(ngpu)))

    netDB.apply(weights_init)

    # Create the Conditional Discriminator
    netDAConditional = NLayerDiscriminator(input_nc=2).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netDAConditional = nn.DataParallel(netDAConditional, list(range(ngpu)))

    netDAConditional.apply(weights_init)

    return netG, netR, netDA, netDB, netDAConditional

# init networks
netGA, netGB, netDA, netDB, netDAConditional = init_net()

######################################################################
# Loss Functions and Optimizers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def loss_and_optimisers():
     # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss().to(device)
    noise_model = {'noise_type':'g', 'sigma':0.1,}
    physics = CT(img_width=image_size, radon_view=90)
    transform = Rotate(n_trans=1)

    # Setup Adam optimizers for both G and D
    optimizerDA = optim.Adam(netDA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDB = optim.Adam(netDB.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDAConditional = optim.Adam(netDA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerGA = optim.Adam(netGA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerGB = optim.Adam(netGB.parameters(), lr=lr, betas=(beta1, 0.999))
    
    return criterion, physics, transform, optimizerGA, optimizerGB, optimizerDA, optimizerDB,optimizerDAConditional

def GANLoss(input, netD, label):
    criterion = nn.BCELoss().to(device)
    b_size = input.size(0)
    feature_size = 2 * 2 # image size in the output layer 2 * 2 for 64,  6 * 6 for 128
    label_tensor = torch.full((feature_size * b_size,), label, dtype=torch.float, device=device)
    
    # Forward pass real batch through D
    output = netD(input).view(-1)

    # Calculate loss on all-real batch
    return criterion(output, label_tensor)

# prepare loss and optimisers
criterion, physics, transform, optimizerGA, optimizerGB, optimizerDA, optimizerDB, optimizerDAConditional = loss_and_optimisers()
criterionCycle = nn.L1Loss().to(device)
criterionIdt = nn.L1Loss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionGAN = GANLoss
mse = nn.MSELoss().to(device)
lambda_idt = 0.5
lambda_A = 10
lambda_B = 10
lambda_L1 = 1
3
def prepare_fixed_noise():
     # Create batch of latent vectors that we will use to visualize the progression of the generator
    X_dataloader = torch.utils.data.DataLoader(X_dataset, batch_size=X_batch_size, shuffle=False)
    fixed_noise = next(iter(X_dataloader))[0].to(device)
    fixed_noise = physics.forw(fixed_noise, add_noise=True)
    fixed_noise = physics.pseudo(fixed_noise)
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
DA_losses = []
DB_losses = []
iters = 0

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    loss_G_seq, loss_DA_seq, loss_DB_seq, loss_eq_seq, loss_seq, psnr_seq = [], [], [], [], [], []
    X_dataloader = torch.utils.data.DataLoader(X_dataset, batch_size=X_batch_size, shuffle=True)
    Y_dataloader = torch.utils.data.DataLoader(Y_dataset, batch_size=Y_batch_size, shuffle=True)

    # For each batch in the dataloader
    for i, fake_batch in enumerate(X_dataloader):
        # Generate latent variable (z) of noise
        real_batch = list(Y_dataloader)[i][0].to(device)
        fake_batch = fake_batch[0].to(device)
        z = physics.forw(fake_batch, add_noise=True)
        z = physics.pseudo(z)
        z = (z - torch.min(z)) / (torch.max(z) - torch.min(z))
        
        # Get real and generated images
        realA = z
        realB = real_batch
        fakeB = netGA(realA)
        recoA = netGB(fakeB)
        fakeA = netGB(realB)
        recoB = netGA(fakeA)

        # Ds require no gradients when optimizing Gs
        set_requires_grad([netDA, netDB, netDAConditional], False)

        # set GA and GB's gradients to zero
        optimizerGA.zero_grad()
        optimizerGB.zero_grad()

        #-------------------------
        # maxmise cycle consistency
        # |GA(B) - B|
        idA = netGA(realB)
        loss_idt_A = criterionIdt(idA, realB) * lambda_B * lambda_idt
        # |GB(A) - A|
        idB = netGB(realA)
        loss_idt_B = criterionIdt(idB, realA) * lambda_A * lambda_idt
        # GAN loss D_A(G_A(A))
        loss_G_A = criterionGAN((fakeB), netDA, real_label)
        # GAN loss D_B(G_B(B))
        loss_G_B = criterionGAN((fakeA), netDB, real_label)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = criterionCycle(recoA, realA) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = criterionCycle(recoB, realB) * lambda_B
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        # #-------------------------
        # # structure preserving loss
        # fakeB = netGA(realA)
        # ssim_loss: torch.Tensor = piq.SSIMLoss(data_range=1.)(fakeB, realA)
        # conformity_loss = ssim_loss

        # #-------------------------
        # # cycle consistency loss for generated
        # fakeB_cycle = netGA(fakeB)
        # cycle_loss = 0 * mse(fakeB_cycle, fakeB)

        #-------------------------
        # domain consistency loss
        meas0 = physics.forw(fakeB)
        fbp0 = physics.pseudo(meas0)
        x1 = netGA(fbp0)
        loss_domain = mse(x1, fakeB.detach())

        #-------------------------
        # conditional adverserial
        fakeB = netGA(realA)
        fake_AB = torch.cat((realA, fakeB), 1)
        loss_G_GAN = criterionGAN(fake_AB, netDAConditional, real_label)
        loss_G_L1 = criterionL1(fakeB, realB) * lambda_L1
        loss_GConditional = loss_G_GAN + loss_G_L1

        #-------------------------
        # eq loss
        x2 = transform.apply(x1.detach())
        meas2 = physics.forw(x2)
        fbp_2 = physics.pseudo(meas2)
        x3 = netGA(fbp_2)
        loss_eq = mse(x3, x2)

        loss = loss_domain  + loss_GConditional + loss_eq
        loss.backward()

        # Update Gs
        optimizerGA.step()
        optimizerGB.step()

        # Update Ds
        set_requires_grad([netDA, netDB, netDAConditional], True)
        optimizerDA.zero_grad()
        optimizerDB.zero_grad()
        optimizerDAConditional.zero_grad()

        #-------------------------
        # adverserial
        loss_DA = 0.5 * (criterionGAN(realB.detach(), netDA, real_label) + criterionGAN(fakeB.detach(), netDA, fake_label))
        loss_DA.backward()
        loss_DB = 0.5 * (criterionGAN(realA.detach(), netDB, real_label) + criterionGAN(fakeA.detach(), netDB, fake_label))
        loss_DB.backward()

        #-------------------------
        # conditional adverserial
        fake_AB = torch.cat((realA.detach(), fakeB.detach()), 1)
        loss_DConditional_fake = criterionGAN(fake_AB, netDAConditional, fake_label)
        real_AB = torch.cat((realA.detach(), realB.detach()), 1)
        loss_DConditional_real = criterionGAN(real_AB, netDAConditional, real_label)
        loss_DConditional = (loss_DConditional_fake + loss_DConditional_real) * 0.5
        
        optimizerDAConditional.step()
        optimizerDA.step()
        optimizerDB.step()

        # losses for all epoch and batch
        G_losses.append(loss_G.item())
        DA_losses.append(loss_DA.item())
        DB_losses.append(loss_DB.item())

        # losses in this epoch
        loss_G_seq.append(loss_G.item())
        loss_DA_seq.append(loss_DA.item())
        loss_DB_seq.append(loss_DB.item())
        loss_eq_seq.append(loss_eq.item())
        loss_seq.append(loss.item())
        psnr_seq.append(cal_psnr(fake_batch, netGA(z)))
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if epoch % 500 == 0 or (epoch == num_epochs-1): 
            fake = torch.zeros(fixed_noise.shape)
            fake = ((netGA(fixed_noise))).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=5))

        # save model
        if epoch > 0 and epoch % 10 == 0:
                    state = {'epoch': epoch,
                             'state_dict': netGA.state_dict(),}

                             
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(state,
                               os.path.join(save_path, 'ckp_{}_ConditionalCycleGAN.pth.tar'.format(now)))
        
        iters += 1
    
    # Averaged over epochs
    loss_closure = [np.mean(loss_G_seq), np.mean(loss_DA_seq), np.mean(loss_DB_seq), np.mean(loss_seq), np.mean(loss_eq_seq)]
    loss_closure.append(np.mean(psnr_seq))
    log.record(epoch + 1, *loss_closure)
    print(
        '{}\tEpoch[{}/{}]\tG={:.4e}\tDA={:.4e}\tDB={:.4e}\tLoss={:.4e}\tEQ={:.4e}\tpsnr={:.4f}'
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
plt.plot(DA_losses,label="D_A")
plt.plot(DB_losses,label="D_B")

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
ani.save(f'gifs/ConditionalcGAN{num_epochs}epochs.gif', writer='imagemagick')

######################################################################
# **Real Images vs. Fake Images**

# Grab a batch of real images from the dataloader
real_batch = next(iter(Y_dataloader))

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