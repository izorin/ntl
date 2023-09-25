import numpy as np  
from types import SimpleNamespace
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# LOCAL
# PROJECT_PATH = '/Users/ivan_zorin/Documents/DEV/code/ntl/'
# DATA_PATH = '/Users/ivan_zorin/Documents/DEV/data/sgcc/data.csv'
# LOG_DIR = '/Users/ivan_zorin/Documents/DEV/runs/debug/trainer'

# ZHORES
PROJECT_PATH = '/trinity/home/ivan.zorin/dev/code/ntl/'
DATA_PATH = '/trinity/home/ivan.zorin/dev/data/sgcc/data.csv'
LOG_DIR = '/trinity/home/ivan.zorin/dev/logs/CNNAE/'


import sys
sys.path.append(PROJECT_PATH)
from ntl.data import SGCCDataset, data_train_test_split
from ntl.data import FillNA, Scale, Reshape, ToTensor, Cutout, Diff
from ntl.models import AE2dCNN
from ntl.trainer import ArgsTrainer
from ntl.utils import fix_seed, get_date


def main():
    
    fix_seed(42)
    
    transforms = [
        FillNA('drift'), 
        Cutout(256), 
        Scale('minmax'), 
        # Diff(1),
        Reshape((16, 16)),
        lambda x: x[None],
        ToTensor()
    ]
    
    normal_data = SGCCDataset(DATA_PATH, label=0, nan_ratio=0.75, transforms=transforms, year=2016)
    anomal_data = SGCCDataset(DATA_PATH, label=1, nan_ratio=1.0, transforms=transforms, year=2016)

    train, test = data_train_test_split(normal_data, anomal_data)

    train_loader = DataLoader(train, batch_size=256, drop_last=False, shuffle=True)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)
    
    model = AE2dCNN()
    loss = nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=2)
    
    date = get_date()
    print(f'experiment folder: {date}')
    # logger = SummaryWriter(log_dir=os.path.join(LOG_DIR, date))
    
    config = SimpleNamespace(**{
        'debug': False,
        'n_debug_batches': np.nan,
        'log_step': 5,
        'n_epochs': 50,
        'split_val_losses': True,
        'LOG_DIR': LOG_DIR
    })
    
    trainer = ArgsTrainer(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        loss=loss,
        optim=optim,
        scheduler=scheduler,
        config=config,
        # logger=logger
    )
    print('start training')
    trainer.train()
    
    
if __name__ == '__main__':
    main()
    print('all done')
    