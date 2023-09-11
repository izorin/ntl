import torch 
from torch import nn 
import torch.nn.functional as F

import numpy as np



# def compose_transforms(trans, transforms_kwargs):
#     transforms_list = []
#     for transform, transform_kwargs in zip(trans, transforms_kwargs):
#         tr = getattr(transforms, transform)(**transforms_kwargs)
#         transforms_list.append(tr)
        
#     return tr

class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = sigma
        
    def __call__(self, x):
        x_noisy = x[1] + np.random.randn(*x[1].shape) * self.sigma
        return (x[0], x_noisy, x[2])

class AddConstant:
    def __init__(self, eps):
        self.eps = eps
    
    def __call__(self, x):
        return (x[0], x[1] + self.eps, x[2])
    