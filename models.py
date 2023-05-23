from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sympy import plot
import torch
from torch import nn
import lightning.pytorch as pl
import torch.nn.functional as F
from utils.utils import reduce_embed_dim,  plot_embeddings, plot_prediction
import wandb
import numpy as np



class LSTMAE_old(nn.Module):
    def __init__(self, input_size=1, hidden_size=[64], n_lstms=1, **lstm_kwargs):
        super().__init__()
        assert n_lstms == len(hidden_size)

        # encoder
        encoder, decoder = [], []
        self.enc_dims = [input_size] + hidden_size
        self.dec_dims = self.enc_dims[::-1]

        for i in range(len(self.enc_dims) - 1):
            encoder += [nn.LSTM(input_size=self.enc_dims[i], hidden_size=self.enc_dims[i+1], num_layers=1, batch_first=True, **lstm_kwargs)]

            decoder += [nn.LSTM(input_size=self.dec_dims[i], hidden_size=self.dec_dims[i+1], num_layers=1, batch_first=True, **lstm_kwargs)]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x):
        N, L, H = x.shape # Batch x Length x Feature size
        # encoding
        _, (hn, _) = self.encoder(x) # latent representation
        hn.transpose_(0, 1) # batch first
        # decoding
        x_hat, (_, _) = self.decoder(hn.expand(-1, L, -1)) # expand to stratch embedding over length of the input

        return (hn.squeeze(1), x_hat.flip(1))


class LSTMEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=[64], n_lstms=1, **lstm_kwargs):
        super().__init__
        
        assert n_lstms == len(hidden_size)
        layers = []
        self.dims = [input_size] + hidden_size
        for i in range(len(self.dims)):
            layers += [nn.LSTM(input_size=self.dims[i], hidden_size=self.dims[i+1], num_layers=1, batch_first=True)]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, input):
        _, (hn, _) = self.layers(input)
        # hn.transpose_(0, 1) # batch first
        
        return hn
        

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_lstms=1, **lstm_kwargs):
        super().__init__()
        assert n_lstms == len(hidden_size)
        
        self.dims = [input_size] + hidden_size 
        layers = []
        for i in range(len(self.dims)):
            layers += [nn.LSTM(input_size=self.dims, hidden_size=self.dims[i+1], num_layers=1, batch_first=True)]
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x, (_, _) = self.layers(x)
        return x


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, ):
        super(LSTMAE, self).__init__()

        self.encoder = LSTMEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x):
        N, L, H = x.shape # Batch x Length x Feature size
        _, (hn, _) = self.encoder(x)
        x_hat, (_, _) = self.decoder(hn.expand(-1, L, -1)) # expand to stratch embedding over length of the input

        return (hn.squeeze(1), x_hat.flip(1))


class LitDummyModel(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Linear(dim, dim, bias=True)
        self.embs = torch.randn([32, 64])
        
    def forward(self, x):
        return self.net(x)
        
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)
    
    def _shared_step(self, batch, batch_idx):
        y = self.net(batch)
        loss = F.mse_loss(batch, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        embs = torch.randn([32, 64])
        fig = plot_embeddings(embs, pca_dim=2, title='validation')
        wandb.log({'val embs': wandb.Image(fig)})

    def on_validation_batch_end(self, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:

        if batch_idx == 0:
            x = np.random.randn(2, 1024)
            fig = plot_prediction(x[0, :], x[1, :])
            wandb.log({'val_x_hat': wandb.Image(fig)})
        
        
class LitLSTMAE(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, logger, config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # self.logger = logger
    
        self.config = config
        
        self.embs = []
        self.val_embs = []
        self.test_embs = []

        self.save_hyperparameters(ignore=['model'])
        
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.config.lr)
        return optimizer
    
    
    def training_step(self, batch, batch_idx):
        _, x, _ = batch
        _, x_hat = self.model(x)
        loss = self.loss_fn(x, x_hat)
        self.log('train_loss', loss)
        return loss
        
        
    def _shared_model_eval(self, batch, batch_idx):
        y, x, _ = batch
        z, x_hat = self.model(x)
        loss = self.loss_fn(x, x_hat)
        self.embs.append(z)
        return y, z, loss
        
    def validation_step(self, batch, batch_idx):
        labels, z, loss = self._shared_model_eval(batch, batch_idx)
        self.log('val_loss', loss)
        
    def test_step(self, batch, batch_idx):
        labels, z, loss = self._shared_model_eval(batch, batch_idx)
        self.log('test_loss', loss)
        
        
    def _shared_on_epoch_end(self, step_name):
        embs = torch.concat(self.embs, dim=0)
        embs = reduce_embed_dim(embs, pca_dim=self.config.pca_dim)
        fig = plot_embeddings(embs, title=f'embeddings of {step_name}', log=True)
    
        # self.embs.clear()
        return fig
    
    def on_validation_epoch_end(self):
        fig = self._shared_on_epoch_end(step_name='val')
        # self.logger.log('val_embeddings', fig)
        # wandb.log({'val_embeddings': wandb.Image(fig)})
        self.embs.clear()
        
    def on_test_epoch_end(self):
        fig = self._shared_on_epoch_end(step_name='test')
        # self.logger.log_image('test_embeddings', fig)
        # wandb.log({'test_embeddings': wandb.Image(fig)})
        self.embs.clear()
        
        
    def _shared_on_batch_end(self, batch, step=''):
        _, x, _ = batch
        x = x[0]
        x_hat = self.model(x)
        fig = plot_prediction(x, x_hat)
        wandb.log({f'{step}_x_hat': wandb.Image(fig)})
    
    def on_validation_batch_end(self, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        
        if batch_idx == 0:
            self._shared_on_batch_end(batch, step='val')
            
        
    

        
        
        
        
    
    

    
    