
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Root directory for dataset
real_dataroot = "dataset/real"
noise_dataroot = "dataset/noise"
test_dataroot = "dataset/test"

######################################################################
# Data
# ----
def get_datasets(image_size, real_batch_size, noise_batch_size):
    real_set = dset.ImageFolder(root=real_dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                            ]))
    real_dataloader = torch.utils.data.DataLoader(real_set, batch_size=real_batch_size, shuffle=True)

    fakes_set = dset.ImageFolder(root=noise_dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                            ]))
    # Create the dataloader
    fakes_dataloader = torch.utils.data.DataLoader(fakes_set, batch_size=noise_batch_size, shuffle=True)

    return real_set, fakes_set

def get_dataloader(image_size, real_batch_size, noise_batch_size):
    real_set = dset.ImageFolder(root=real_dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                            ]))
    real_dataloader = torch.utils.data.DataLoader(real_set, batch_size=real_batch_size, shuffle=True)

    fakes_set = dset.ImageFolder(root=noise_dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                            ]))
    # Create the dataloader
    fakes_dataloader = torch.utils.data.DataLoader(fakes_set, batch_size=noise_batch_size, shuffle=True)

    return real_dataloader, fakes_dataloader


def get_test_dataloader(image_size, test_batch_size=1):
    dataset = dset.ImageFolder(root=test_dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
    return dataloader