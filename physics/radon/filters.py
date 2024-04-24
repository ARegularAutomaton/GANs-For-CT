import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import numpy.fft
fftmodule = numpy.fft
from .utils import PI, fftfreq

class AbstractFilter(nn.Module):
    def __init__(self):
        super(AbstractFilter, self).__init__()

    def forward(self, x):
        input_size = x.shape[2]
        sinogram_fft = torch.fft.fft(x, dim=-2)
        filter = self._get_fourier_filter(input_size)
        sinogram_fft_filtered = sinogram_fft.transpose(2,3) * filter.to(x.device)
        filtered_sinogram = torch.fft.ifft(sinogram_fft_filtered.transpose(2,3), dim=-2).real
        return filtered_sinogram
    
        # print(filtered_sinogram.shape)
        # exit()
        # input_size = x.shape[2]
        # projection_size_padded = max(64, int(2 ** (2 * torch.tensor(input_size)).float().log2().ceil()))
        # pad_width = projection_size_padded - input_size
        # padded_tensor = F.pad(x, (0,0,0,pad_width))
        # f = self._get_fourier_filter(padded_tensor.shape[2]).to(x.device)
        # fourier_filter = self.create_filter(f)
        # projection = torch.fft.fft(padded_tensor.transpose(2,3), dim=-2).real * fourier_filter
        # return torch.fft.ifft(projection.transpose(2,3), dim=-1)[:,:,:input_size,:].real

    def _get_fourier_filter(self, size):
        freq_axis = torch.fft.fftfreq(size)
        fourier_filter = torch.abs(freq_axis)
        return fourier_filter
    
        # n = torch.cat([
        #     torch.arange(1, size / 2 + 1, 2),
        #     torch.arange(size / 2 - 1, 0, -2)
        # ])
        # f = torch.zeros(size)
        # f[0] = 0.25
        # f[1::2] = -1 / (PI * n) ** 2

    def create_filter(self, f):
        raise NotImplementedError

class RampFilter(AbstractFilter):
    def __init__(self):
        super(RampFilter, self).__init__()

    def create_filter(self, f):
        return f

class HannFilter(AbstractFilter):
    def __init__(self):
        super(HannFilter, self).__init__()

    def create_filter(self, f):
        n = torch.arange(0, f.shape[0])
        hann = 0.5 - 0.5*(2.0*PI*n/(f.shape[0]-1)).cos()
        return f*hann.roll(hann.shape[0]//2,0).unsqueeze(-1)

class LearnableFilter(AbstractFilter):
    def __init__(self, filter_size=363):
        super(LearnableFilter, self).__init__()
        self.filter = nn.Parameter(2*fftfreq(filter_size).abs())

    def forward(self, x):
        fourier_filter = self.filter.to(x.device)
        print(torch.fft.fft(x.transpose(2,3), dim=-2).shape)
        projection = torch.fft.fft(x.transpose(2,3), dim=-2) * fourier_filter
        return torch.fft.ifft(projection.transpose(2,3), dim=-1).real