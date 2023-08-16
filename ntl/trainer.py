import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, ConcatDataset

import numpy as np
import matplotlib.pyplot as plt
import os 
import sys

from data import SGCCDataset 
import models
from utils import compute_roc_auc, load_config



class Trainer:
    
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        
        # data        
        normal_dataset = SGCCDataset(path=self.config.data_path, label=0, scale=self.config.scale, nan_ratio=self.config.nan_ratio)
        anomal_dataset = SGCCDataset(path=self.config.data_path, label=1, scale=self.config.scale)

        train_data, val_data, test_normal_data = random_split(normal_dataset, [len(normal_dataset) - 2*len(anomal_dataset), len(anomal_dataset), len(anomal_dataset)])
        test_data = ConcatDataset([test_normal_data, anomal_dataset])

        self.train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=self.config.debug)
        
        # models
        self.model = getattr(models, self.config.model)(**self.config.model_kwargs).to(self.device)

        # optimzers ans schedulers
        self.optim = getattr(torch.optim, self.config.optimizer)(self.model.parameters(), **self.config.optimizer_kwargs) # torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, **self.config.scheduler_kwargs)
# mode='min', factor=factor, patience=patience, verbose=True)

        # loss function
        self.loss_fn = getattr(nn, self.config.loss_fn)(**self.config.loss_kwargs) # nn.L1Loss()

        # loggers
        self.logger = SummaryWriter(self.config.save_dir)
        
        # buffer
        self.buffer = {
            'scores': [],
            'labels': []
        }
        
    def _clear_mem(self):
        for key in self.buffer.keys():
            self.buffer[key] = []
            
     
    def fix_seed(self):
        # TODO fix seed for 
        # np.random
        # random 
        # torch
        # torch.cuda
        # recurrent models ?
        raise NotImplemented
    
    def model_step(self, batch, embeddings, labels):
        y, x, _ = batch
        x = x.to(self.device)
        z, x_hat = self.model(x)
        loss = self.loss_fn(x, x_hat)
        
        embeddings.append(z.detach().cpu().numpy().squeeze())
        labels.append(y)
        
        return loss
    
    def shared_step(self, epoch, step_name):
        losses = []
        embeddings = []
        labels = []
        loader = self.train_loader if step_name == 'train' else self.val_loader
        loader_len = len(loader)
        
        for i, batch in enumerate(loader):
            if self.config.debug and i >= self.config.n_debug_batches:
                break
            step = i + loader_len * epoch
            loss = self.model_step(batch, embeddings, labels)
            loss = torch.mean(loss, dim=(1,2))
            losses.append(loss.detach().cpu().numpy())
            loss = torch.mean(loss)
            
            if self.model.training:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            self.logger.add_scalar(f'{step_name}/loss', loss.item(), step)
        
        losses = np.concatenate(losses)
        embeddings = np.concatenate(embeddings)
        labels = torch.cat(labels)
        
        if epoch % self.config.log_step == 0:
            self.logger.add_embedding(tag=f'{step_name}/embs', mat=embeddings, metadata=labels, global_step=epoch)
        
        return losses, embeddings, labels
        
    def train_step(self, epoch):
        self.model.train()
        losses, _, _ = self.shared_step(epoch, step_name='train')
        return losses.mean()
    
    def val_step(self, epoch):
        self.model.eval()
        with torch.no_grad():
            losses, embeddings, labels = self.shared_step(epoch, step_name='val')
            self.buffer['scores'] += losses.tolist() # save all losses and lables 
            self.buffer['labels'] += labels.tolist() # to compute roc-auc later
        
        return losses.mean()
    
    def supervised_validation(self, scores, labels, epoch):
        (_, fig), (FPR, TPR, auc_score) = compute_roc_auc(scores, labels, pyplot=True)
        self.logger.add_scalar('val/auc-score', auc_score, epoch)
        self.logger.add_figure(tag='val/roc-auc', figure=fig, global_step=epoch)
        
        self._clear_mem() # empty buffer
        
    
    def train(self):
        # TODO add tqdm
        train_losses, val_losses = [], []
        for epoch in range(self.config.n_epochs):
            # train_loss = self.train_step(epoch)
            train_loss = -1
            val_loss = self.val_step(epoch)
            self.scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.logger.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)  
            self.supervised_validation(self.buffer['scores'], self.buffer['labels'], epoch)
            
        
    def save(self):
        # save experiment results: model checkpoint
        raise NotImplemented
    
    
    
    
    
    
    
def main(experiment_config, path_config):
    config = load_config(experiment_config, path_config)
    trainer = Trainer(config)
    trainer.train()
    
    # trainer.save()
    
    
if __name__ == '__main__':
    # fire.Fire(main)
    experiment_config = '/Users/ivan_zorin/Documents/DEV/code/ntl/configs/trainer_debug.yaml'
    path_config = '/Users/ivan_zorin/Documents/DEV/code/ntl/configs/local_pathes.yaml'
    main(experiment_config, path_config)
    