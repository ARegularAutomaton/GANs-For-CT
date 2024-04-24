import torch
import torch.nn as nn
import torch.nn.parallel

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, residual=False, ngpu=1):
        super().__init__()
        self.name = 'unet'
        self.ngpu = ngpu
        self.residual = residual
        
        self.Conv1 = Conv(input_channels, out_channels=64)
        self.Conv2 = Conv(in_channels=64, out_channels=128)
        self.Conv3 = Conv(in_channels=128, out_channels=256)
        self.Conv4 = Conv(in_channels=256, out_channels=512)
        self.Conv5 = Conv(in_channels=512, out_channels=1024)
        self.MaxPool2D = nn.MaxPool2d(kernel_size=2, stride=2)

        self.UpConv5 = UpConv(in_channels=1024, out_channels=512)
        self.ConvUp4 = Conv(in_channels=1024, out_channels=512)
        self.UpConv4 = UpConv(in_channels=512, out_channels=256)
        self.ConvUp3 = Conv(in_channels=512, out_channels=256)
        self.UpConv3 = UpConv(in_channels=256, out_channels=128)
        self.ConvUp2 = Conv(in_channels=256, out_channels=128)
        self.UpConv2 = UpConv(in_channels=128, out_channels=64)
        self.ConvUp1 = Conv(in_channels=128, out_channels=64)
        self.ConvOut = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.MaxPool2D(x1)
        x2 = self.Conv2(x2)
        x3 = self.MaxPool2D(x2)
        x3 = self.Conv3(x3)
        x4 = self.MaxPool2D(x3)
        x4 = self.Conv4(x4)
        x5 = self.MaxPool2D(x4)
        x5 = self.Conv5(x5)
        y4 = self.UpConv5(x5)
        y4 = torch.cat((x4,y4), dim=1)
        y4 = self.ConvUp4(y4)
        y3 = self.UpConv4(y4)
        y3 = torch.cat((x3,y3), dim=1)
        y3 = self.ConvUp3(y3)
        y2 = self.UpConv3(y3)
        y2 = torch.cat((x2,y2), dim=1)
        y2 = self.ConvUp2(y2)
        y1 = self.UpConv2(y2)
        y1 = torch.cat((x1,y1), dim=1)
        y1 = self.ConvUp1(y1)
        y1 = self.ConvOut(y1)
        y1 = nn.Sigmoid()(y1)
        return y1 + x if self.residual else y1

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv.apply(weights_init)

    def forward(self, x):
        return self.conv(x)
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up_conv.apply(weights_init)
    
    def forward(self, x):
        return self.up_conv(x)

# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)