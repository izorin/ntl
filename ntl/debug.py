#%%
# %load_ext autoreload
# %autoreload 2

#%%
from symbol import encoding_decl
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

import fairseq 
import sequitur as seq
from sequitur.models import LSTM_AE


import os 
import sys
sys.path.append('/Users/ivan_zorin/Documents/DEV/code/ntl/')

from data.data import SGCCDataset
import models
from models import *

from sequitur.models.lstm_ae import Encoder, Decoder


#%%
experiment_config = '/Users/ivan_zorin/Documents/DEV/code/ntl/configs/trainer_debug.yaml'
path_config = '/Users/ivan_zorin/Documents/DEV/code/ntl/configs/local_pathes.yaml'

config = load_config(experiment_config, path_config)

device = config.device

normal_dataset = SGCCDataset(path=config.data_path, label=0, scale=config.scale, nan_ratio=config.nan_ratio)
anomal_dataset = SGCCDataset(path=config.data_path, label=1, scale=config.scale)

train_data, val_data, test_normal_data = random_split(normal_dataset, [len(normal_dataset) - 2*len(anomal_dataset), len(anomal_dataset), len(anomal_dataset)])
test_data = ConcatDataset([test_normal_data, anomal_dataset])

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=config.debug)

#%%
batch = next(iter(train_loader))
y, x, _ = batch
x = x.to(device)
x.shape


#%%
model = SequiturLSTMAE(**config.model_kwargs).to(device)
model

#%%
z = model.encoder(x)
z.shape

#%% 
seq_len = x.shape[1]
x_hat = model.decoder(z, seq_len)
x_hat.shape
#%% 
out = model(x)
print(out[0].shape, out[1].shape)
# %%
