import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.generator import Generator

from dataset.dataset import get_test_dataloader

from physics.ct import CT

from utils.metric import cal_psnr, cal_ssim

def test_ct(net_name, net_ckp, device):
    image_size = 64
    radon_view = 90
    I0 = 1e5
    sigma = 0.1

    noise_model = {'noise_type': 'mpg',
                   'sigma': sigma}

    generator = Generator(input_channels=1, output_channels=1, residual=True).to(device)

    dataloader = get_test_dataloader(image_size, 1)

    ct = CT(image_size, radon_view, circle=False, device=device, I0=I0, noise_model=noise_model)

    psnr_fbp, psnr_net = [],[]
    ssim_fbp, ssim_net = [],[]

    for _, x in enumerate(dataloader):
        simulation = 'astra'
        if simulation == 'radon':
            x = x[0] if isinstance(x, list) else x
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            x = x.type(torch.float).to(device)

            x = x * (forw.MAX - forw.MIN) + forw.MIN
            y = forw.A(x, add_noise=True)
            fbp = forw.iradon(torch.log(forw.I0 / y))

            psnr_fbp.append(cal_psnr(fbp, x))
            ssim_fbp.append(cal_ssim(fbp, x))


            checkpoint = torch.load(net_ckp, map_location=device)
            unet.load_state_dict(checkpoint['state_dict'])
            unet.to(device).eval()
            x_net = f(fbp)

            psnr_net.append(cal_psnr(x_net, x))
            ssim_net.append(cal_ssim(x_net, x))

        elif simulation == 'astra':
            x = x[0] if isinstance(x, list) else x
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            x = x.type(torch.float).to(device)
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

            y = ct.forw(x, add_noise=True)
            fbp = ct.pseudo(y)

            psnr_fbp.append(cal_psnr(fbp, x))
            ssim_fbp.append(cal_ssim(fbp, x))

            checkpoint = torch.load(net_ckp, map_location=device)
            generator.load_state_dict(checkpoint['state_dict'])
            generator.to(device).eval()
            x_net = generator(fbp)

            psnr_net.append(cal_psnr(x_net, x))
            ssim_net.append(cal_ssim(x_net, x))
        # show(x, fbp, x_net)

    print('AVG-PSNR (views={}\tI0={}\tsigma={})\t FBP={:.3f} + {:.3f}\t{}={:.3f} + {:.3f}'.format(
        radon_view,I0,sigma, np.mean(psnr_fbp),np.std(psnr_fbp), net_name, np.mean(psnr_net), np.std(psnr_net)))
    print('AVG-SSIM (views={}\tI0={}\tsigma={})\t FBP={:.3f} + {:.3f}\t{}={:.3f} + {:.3f}'.format(
        radon_view,I0,sigma, np.mean(ssim_fbp),np.std(ssim_fbp), net_name, np.mean(ssim_net), np.std(ssim_net)))

def show(x, fbp, x_net):
    # if get_display_metric(x_net, x, 0).astype(np.float64) > 0.76:
        plt.subplot(1,3,1)
        plt.axis('off')
        plt.imshow(fbp.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.title('A')
        plt.text(63, 6, f'PSNR: {get_display_metric(fbp, x)}\n SSIM: {get_display_metric(fbp, x, 0)}', color='white', fontsize=12, ha='right')
        
        plt.subplot(1,3,2)
        plt.axis('off')
        plt.imshow(x_net.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.title('B')
        plt.text(63, 6, f'PSNR: {get_display_metric(x_net, x)}\n SSIM: {get_display_metric(x_net, x, 0)}', color='white', fontsize=12, ha='right')

        plt.subplot(1,3,3)
        plt.axis('off')
        plt.imshow(x.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.title('C')
        plt.text(63, 6, f'', color='white', fontsize=12, ha='right')

        plt.show()

def get_display_metric(bp, x, name='psnr'):
    if name == 'psnr':
        return np.around(cal_psnr(bp, x), decimals=2).astype(str)
    else:
        return np.around(cal_ssim(bp, x), decimals=2).astype(str)

def plot(path):
    files = list_files(path)
    headers = ["epoch","mc loss","eq loss","Total loss","SSIM","PSNR","PSNR","GPU Memory"]
    # print(plt.style.available)
    plt.style.use('seaborn-v0_8')
    for f in range(len(files)):
        for c in range(7):
            if is_csv_file(files[f]) and c == 6:
                arr = np.genfromtxt(files[f], delimiter=",", skip_header=1, usecols=(0,c))
                arr = arr.swapaxes(0,1)
                # print(arr)
                # for i in range(len(arr[1])):
                #     arr[1][i] = arr[1][i] # log(abs(arr[1][i]))
                plt.plot(arr[0], arr[1], label=headers[c])
    plt.xlabel("epochs", fontsize=16)
    plt.ylabel("value", fontsize=16)
    plt.title("PSNR during training", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    plt.style.use('default')

def list_files(path):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def is_csv_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == '.csv'

if __name__ == '__main__':
    device = 'cuda:0'

    # plot metrics
    path = 'ckp\EIGANRealMSE'
    plot(path)
    
    # test network
    net_ckp_ct = 'ckp\EIGANRealMSE\ckp_423102828_EIGANLatent.pth.tar'
    test_ct(net_name='rei',net_ckp=net_ckp_ct, device=device)