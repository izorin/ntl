import torch 
from torch import nn 
import torch.nn.functional as F

import numpy as np


class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = sigma
        
    def __call__(self, x):
        x_noisy = x[1] + np.random.randn(*x[1].shape) * self.sigma
        return (x[0], x_noisy, x[2])