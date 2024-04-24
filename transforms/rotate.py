import torch
import random
import numpy as np


class Rotate():
    def __init__(self, n_trans):
        self.n_trans = n_trans

    def apply(self, x):
        return rotate_random(x, self.n_trans)

def rotate_random(x, n_rotations=3):
    rotated_tensors = []

    rotations = range(1,n_rotations+1, 1)
    for rotation in rotations:
        rotated_tensor = x.clone()
        rotated_tensor = torch.rot90(rotated_tensor, k=rotation, dims=(-2, -1))
        
        rotated_tensors.append(rotated_tensor)
    
    x = torch.cat(rotated_tensors, dim=0)
    
    return x