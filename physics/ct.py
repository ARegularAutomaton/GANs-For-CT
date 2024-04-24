import torch
import numpy as np
import astra
import random
from .radon import Radon, IRadon


class CT():
    def __init__(self, img_width, radon_view, uniform=True, circle = False, device='cuda:0', I0=1e5, noise_model=None):
        if uniform:
            theta = np.linspace(35, 145, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.theta = theta
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)

        self.name='ct'
        self.I0 = I0

        # used for normalzation input
        self.MAX = 0.032 / 5
        self.MIN = 0

        # astra
        self.proj_id = None

        if noise_model is None:
            self.noise_model = {'noise_type':'g',
                                'sigma':0.1,}
        else:
            self.noise_model = noise_model

    def noise(self, m):
        if self.noise_model['sigma'] > 0:
            noise = torch.randn_like(m) * self.noise_model['sigma']
            m = m + noise
        return m

    def forw(self, x, add_noise=False, angles=None):
        simulation = 'astra'
        # self.theta = np.linspace(0, 180, random.randint(5,20), endpoint=False) if angles is None else angles # random angles
        
        if simulation == 'radon':
            m = self.I0 * torch.exp(-self.radon(x)) # clean GT measurement
        elif simulation == 'astra':
            length = int((x.shape[-2]**2 + x.shape[-1]**2)**0.5 + 0.5)
            m = torch.zeros(x.shape[0], x.shape[1], len(self.theta), length).to(x.device)
            for i in range(x.shape[0]):
                m[i] = self.radon_astra(x[i], length)

        if add_noise:
            m = self.noise(m)
        return m
        
    def radon_astra(self, x, length):
        device = x.device
        tensor = np.zeros((x.shape[0], len(self.theta), length))
        x = x.detach().cpu().numpy()

        for i in range(x.shape[0]):
            # Specify projection geometry
            proj_geom = astra.create_proj_geom('parallel', 1, length, self.theta)

            # Specify volume geometry
            vol_geom = astra.create_vol_geom(x.shape[1], x.shape[2])

            # Create a projector object
            self.proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

            # Create a sinogram
            _, tensor[i] = astra.create_sino(x[i], self.proj_id)

            x = torch.from_numpy(tensor).to(device)
        return x

    def pseudo(self, y):
        simulation = 'astra'

        if simulation == 'radon':
            x = self.iradon(y)
            return x
        elif simulation == 'astra':
            assert self.proj_id != None and type(self.proj_id) == int
            vol_geom = astra.projector.volume_geometry(self.proj_id)
            x = torch.zeros((y.shape[0], y.shape[1], vol_geom['GridRowCount'], vol_geom['GridColCount'])).to(y.device)
            for i in range(y.shape[0]):
                x[i] = self.iradon_astra(y[i])
            return x

    def iradon_astra(self, x):
        algorithm = 'FBP'
        device = x.device
        x = x.detach().cpu().numpy()

        proj_geom = astra.projector.projection_geometry(self.proj_id)
        vol_geom = astra.projector.volume_geometry(self.proj_id)
        tensor = np.zeros((x.shape[0], vol_geom['GridRowCount'], vol_geom['GridColCount']))

        for i in range(x.shape[0]):
            # id of the reoncstructed data
            recon_id = astra.data2d.create('-vol', vol_geom)
            
            # run reconstruction algorithm
            cfg = astra.astra_dict(algorithm + '_CUDA')
            cfg['ProjectionDataId'] = astra.data2d.create('-sino', proj_geom, x[i])
            cfg['ReconstructionDataId'] = recon_id
            fbp_id = astra.algorithm.create(cfg)
            astra.algorithm.run(fbp_id)
            astra.algorithm.delete(fbp_id)
            tensor[i] = astra.data2d.get(recon_id)

            x = torch.from_numpy(tensor).to(device)
        return x
            
