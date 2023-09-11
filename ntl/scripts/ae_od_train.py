import numpy as np 
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, robust_scale
from sktime.transformations.series.impute import Imputer

from functools import partial
from types import SimpleNamespace

import sys
sys.path.append('/Users/ivan_zorin/Documents/DEV/code/ntl/')
from ntl.data import SGCCDataset, data_train_test_split
from ntl.data import FillNA, Scale, Reshape, ToTensor, Cutout
from ntl.models import AE2dCNN
from ntl.trainer import ArgsTrainer
from ntl.utils import fix_seed

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():
    
    fix_seed(42)
    
    path = '/Users/ivan_zorin/Documents/DEV/data/sgcc/data.csv'
    transforms = [FillNA('drift'), 
                Cutout(256), 
                Scale('robust'), 
                Reshape((16, 16)),
                lambda x: x[None],
                ToTensor()
    ]
    normal_data = SGCCDataset(path, label=0, nan_ratio=0.75, transforms=transforms, year=2016)
    anomal_data = SGCCDataset(path, label=1, nan_ratio=1.0, transforms=transforms, year=2016)

    train, test = data_train_test_split(normal_data, anomal_data)

    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    test_loader = DataLoader(test, batch_size=4, shuffle=False)
    
    model = AE2dCNN()
    loss = nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=2)
    logger = SummaryWriter(log_dir='/Users/ivan_zorin/Documents/DEV/runs/debug/trainer') 
    
    config = SimpleNamespace(**{
        'debug': True,
        'n_debug_batches': 5,
        'log_step': 5,
        'n_epochs': 10
    })
    
    trainer = ArgsTrainer(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        loss=loss,
        optim=optim,
        scheduler=scheduler,
        config=config,
        logger=logger
    )
    
    trainer.train()
    
    
if __name__ == '__main__':
    main()
    print('all done')
    