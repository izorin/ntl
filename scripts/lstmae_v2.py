import torch
from torch import nn
import torch.nn.functional as F
from torch import utils
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm, trange
import os 
from datetime import datetime

# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import sys
sys.path.append('/home/ivan.zorin/dev/code/ntl/')

from data.data import sgcc_train_test_split, SGCCDataset
from models.models import LSTMAE_old
from utils.utils import compute_roc_auc


def inspect_grad_norm(model, norm_type=2):
    name_norm = {}
    with torch.no_grad():
        for p in model.named_parameters():
            if p[1].grad is not None and p[1].requires_grad:
                name_norm[p[0]] = torch.norm(p[1], norm_type).item()
    
    return name_norm


def main():
    data_path = '/home/ivan.zorin/dev/data/sgcc/data.csv'
    experiment_name = 'lstm_ae_grad_norms_watch'
    date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    run_path = os.path.join('/home/ivan.zorin/dev/logs/', experiment_name, date)
    scale = 'minmax'
    nan_ratio = 0.7
    batch_size = 32

    input_size = 1
    hidden_size = [64]
    lr = 0.0001
    factor = 0.5
    patience = 3

    N_epochs = 20
    val_logging_step = 5

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        assert False, 'cuda is not available'


    # data
    normal_dataset = SGCCDataset(path=data_path, label=0, scale=scale, nan_ratio=nan_ratio)
    anomal_dataset = SGCCDataset(path=data_path, label=1, scale=scale)

    train_data, val_data, test_normal_data = utils.data.random_split(normal_dataset, [len(normal_dataset) - 2*len(anomal_dataset), len(anomal_dataset), len(anomal_dataset)])
    test_data = utils.data.ConcatDataset([test_normal_data, anomal_dataset])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # model and train utils
    model = LSTMAE_old(input_size, hidden_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=factor, patience=patience, verbose=True)
    loss_fn = nn.L1Loss()
    logger = torch.utils.tensorboard.SummaryWriter(run_path)
    
    
    # training
    train_len = len(train_loader)
    val_len = len(val_loader)

    for epoch in trange(N_epochs, total=N_epochs):
    # for epoch in range(N_epochs):
        
        train_losses, val_losses = [], []
        train_embeddings, val_embeddings = [], []
        val_labels = []
        val_scores = []

        train_iterator = tqdm(train_loader, leave=False, desc='Train')
        val_iterator = tqdm(val_loader, leave=False, desc='Val')
        
        model.train()
        for i, batch in enumerate(train_iterator):
            optim.zero_grad()
            y, x, _ = batch
            x = x.to(device)
            z, x_hat = model(x)
            loss = loss_fn(x, x_hat)
            
            loss.backward()
            optim.step()
            
            train_losses.append(loss.item())
            train_embeddings.append(z.detach().cpu().numpy().squeeze())
            step = i + train_len * epoch
            logger.add_scalar('train/loss', loss.item(), step)
        
        
        train_embeddings = np.concatenate(train_embeddings)
        train_loss = sum(train_losses) / len(train_losses)
        logger.add_embedding(tag='train/embs', mat=train_embeddings, global_step=epoch)
        
        # inspect_grad_norm(model)
        # log gradient norms 
        # logger.

        model.eval()
        for i, batch in enumerate(val_iterator):
            with torch.no_grad():
                y, x, _ = batch
                x = x.to(device)
                z, x_hat = model(x)
                loss = loss_fn(x, x_hat)
                scores = F.l1_loss(x, x_hat, reduction='none').mean(axis=1).cpu().numpy()
                
                val_labels.append(y)
                val_losses.append(loss.item())
                val_scores.append(scores)
                val_embeddings.append(z.detach().cpu().numpy().squeeze())
                step = i + train_len * epoch
                logger.add_scalar('val/loss', loss.item(), step)
                
        val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(val_loss)
        logger.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        
        val_embeddings = np.concatenate(val_embeddings)
        if epoch % val_logging_step == 0:
            logger.add_embedding(tag='val/embs', mat=val_embeddings, global_step=epoch)
        
        
        val_labels = torch.concat(val_labels, dim=0).numpy()
        val_scores = np.concatenate(val_scores, axis=0)
        (_, fig), (FPR, TPR, auc_score) = compute_roc_auc(val_scores, val_labels, pyplot=True)
        logger.add_scalar('val/auc-score', auc_score, epoch)
        logger.add_figure(tag='val/roc-auc', figure=fig, global_step=epoch)
        
        grad_norms = inspect_grad_norm(model)
        logger.add_scalars('grad_norm', grad_norms, epoch)
        
        
        
        
if __name__ == '__main__':
    main()