import os 
import sys 
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import wandb
from lightning.pytorch.loggers import WandbLogger
import fire


sys.path.append('/Users/ivan_zorin/Documents/DEV/code/ntl/')

import models
from models import *
from data.data import SGCCDataset, sgcc_train_test_split, DummyDataset
from utils.utils import load_config


def main(config):
    # config and logger
    config_path = config
    config = load_config(config_path)
    wandb_logger = WandbLogger(**config.logger)
    wandb.save(config_path) # save config file
    # data
    train_data, val_data, test_data = sgcc_train_test_split(config)
    num_workers = config.num_workers
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False,  num_workers=num_workers)

    # base model
    nn_model = getattr(models, config.model_name)
    nn_model = nn_model(**config.model_kwargs)
    # Lightning model
    model = LitLSTMAE(
        model=nn_model,
        loss_fn=F.l1_loss,
        optimizer=torch.optim.Adam,
        config=config
    )
    # Trainer
    trainer = pl.Trainer(fast_dev_run=False,
                         check_val_every_n_epoch=1,
                         max_epochs=config.n_epochs,
                         accelerator=config.device,
                         logger=wandb_logger,
                         log_every_n_steps=1,
)
    # fit loop
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # tests
    trainer.test(model, test_loader)
    
    # upload logs
    model.clear_mem()
    wandb.finish()


if __name__ == '__main__':
    fire.Fire(main)
    
    