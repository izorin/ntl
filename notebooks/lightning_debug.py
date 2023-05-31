import os 
import sys 
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import wandb
from lightning.pytorch.loggers import WandbLogger


sys.path.append('/Users/ivan_zorin/Documents/DEV/code/ntl/')

from models import *
from data.data import SGCCDataset, sgcc_train_test_split, DummyDataset
from utils.utils import load_config
from train import main


def main_run(config, wandb_logger):

    train_data, val_data, test_data = sgcc_train_test_split(config)
    num_workers = 2
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False,  num_workers=num_workers)


    lstmae = LSTMAE_old(**config.model_kwargs)

    model = LitLSTMAE(
        model=lstmae,
        loss_fn=getattr(torch.nn.functional, config.loss),
        optimizer=getattr(torch.optim, config.optimizer),
        config=config
    )
    

    trainer = pl.Trainer(fast_dev_run=False,
                    check_val_every_n_epoch=config.val_every_n_steps,
                    limit_train_batches=10,
                    limit_val_batches=10,
                    max_epochs=2,
                    #  num_sanity_val_steps=1,
                    accelerator=config.device,
                    #  profiler='simple',
                    logger=wandb_logger,
                    log_every_n_steps=1,
)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        
    )

    trainer.test(model, test_loader)
    
    model.clear_mem()


    
def dummy_run(config, wandb_logger):
    
    dim = 64
    train_dummy = DataLoader(DummyDataset(dim), batch_size=16)
    val_dummy = DataLoader(DummyDataset(dim), batch_size=16)
    
    dummy_model = LitDummyModel(dim=dim)
    
    trainer = pl.Trainer(fast_dev_run=False,
                    check_val_every_n_epoch=1,
                    limit_train_batches=10,
                    limit_val_batches=10,
                    max_epochs=2,
                    #  num_sanity_val_steps=1,
                    accelerator=config.device,
                    #  profiler='simple',
                    logger=wandb_logger,
                    log_every_n_steps=1,
                )
    
    trainer.fit(
        model=dummy_model,
        train_dataloaders=train_dummy,
        val_dataloaders=val_dummy  
    )

####################################################################################        
if __name__ == '__main__':
    
    dummy = False
    
    pathes_file = './configs/local_pathes.yaml'
    
    if dummy:
        config = load_config('./configs/dummy_config.yaml', pathes_file)
        wandb_logger = WandbLogger(**config.logger)
        # wandb_logger = WandbLogger(
        #     name='dummy-run',
        #     project='NTL',
        #     config=config.__dict__,
        # )
        
        dummy_run(config, wandb_logger)
    
    else: 
        config_path = './configs/config.yaml'
        config = load_config(config_path, pathes_file)
        main(config_path, pathes_file)
    
    
    wandb.finish()