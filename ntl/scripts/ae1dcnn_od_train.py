import numpy as np  
from types import SimpleNamespace
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# LOCAL
# PROJECT_PATH = '/Users/ivan_zorin/Documents/DEV/code/ntl/'
# DATA_PATH = '/Users/ivan_zorin/Documents/DEV/data/sgcc/data.csv'
# LOG_DIR = '/Users/ivan_zorin/Documents/DEV/runs/debug/trainer'

# ZHORES
PROJECT_PATH = '/trinity/home/ivan.zorin/dev/code/ntl/'
DATA_PATH = '/trinity/home/ivan.zorin/dev/data/sgcc/data.csv'
LOG_DIR = '/trinity/home/ivan.zorin/dev/logs/debug/ae1dcnn_1batch/'


import sys
sys.path.append(PROJECT_PATH)
from ntl.data import SGCCDataset, data_train_test_split
from ntl.data import FillNA, Scale, ToTensor
from ntl.models import AE1dCNN
from ntl.trainer import ArgsTrainer
from ntl.utils import fix_seed, get_date


def main():
    
    fix_seed(42)
    
    transforms = [
        FillNA('drift'), 
        Scale('minmax'), 
        ToTensor(),
        lambda x: x.transpose(1,2)
    ]
    
    normal_data = SGCCDataset(DATA_PATH, label=0, nan_ratio=0.75, transforms=transforms, year=2016)
    anomal_data = SGCCDataset(DATA_PATH, label=1, nan_ratio=1.0, transforms=transforms, year=2016)

    batch_size = 64
    full_dataset_training = False 
    
    if full_dataset_training:
        ########################## full-dataset ##########################
        train, test = data_train_test_split(normal_data, anomal_data)

        train_loader = DataLoader(train, batch_size=batch_size, drop_last=False, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
        
    else:
        ######################### one-batch #############################
        normal_idxs = np.random.choice(np.arange(len(normal_data)), (2, batch_size), replace=False)
        anomal_idxs = np.random.choice(np.arange(len(anomal_data)), batch_size, replace=False)
        
        train_data = torch.utils.data.Subset(normal_data, normal_idxs[0, :])
        test_normal = torch.utils.data.Subset(normal_data, normal_idxs[1, :])
        test_anomal = torch.utils.data.Subset(anomal_data, anomal_idxs)
        test_data = ConcatDataset([test_normal, test_anomal])
        
        train_loader = DataLoader(train_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        ##################################################################
    
    
    model = AE1dCNN()
    loss = nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=2)
    
    # date = get_date()
    # print(f'experiment folder: {date}')
    # logger = SummaryWriter(log_dir=os.path.join(LOG_DIR, date))
    
    config = SimpleNamespace(**{
        'debug': True,
        'n_debug_batches': 5,
        'log_step': 5,
        'n_epochs': 2,
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
    