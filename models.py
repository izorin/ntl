import torch
from torch import nn
import lightning.pytorch as pl
import torch.nn.functional as F
from utils.utils import *
import wandb
import numpy as np
import plotly.express as px



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
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('val/loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        embs = torch.randn([32, 64])
        labels = np.random.choice(['b','r'], size=32)
        embs = reduce_embed_dim(embs, pca_dim=2)
        
        # plt plot
        # fig = plot_embeddings(embs, title=f'val_embeddings', log=True)
        # wandb.log({'val embs': wandb.Image(fig)})
        
        # wandb plot
        # table = wandb.Table(columns=['x', 'y'], data=embs)
        # wandb.log({'val/embs_2D': wandb.plot.scatter(table, 'x', 'y')})
        
        # plotly plot 
        fig = plot_embeddings(embs, labels, title='plots/val/embs')
        wandb.log({'plots/val/embs': fig})
        
        
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):

        if batch_idx == 0:
            _len = 1024
            x = np.random.randn(2, _len)
            
            fig = plot_prediction(x[0, :], x[1, :])
            wandb.log({'plots/val/x_x_hat': fig})
            
            # wandb.log({'val/x_hat': wandb.Image(fig)})
            # data = [[i, x, x_hat, x_x_hat] for i, x, x_hat, x_x_hat in zip(range(_len), x[0, :], x[1, :], x[0, :] - x[1, :])]
            # table = wandb.Table(data=data, columns=['i', 'x', 'x_hat', 'x - x_hat'])
            # line_x = wandb.plot.line_series(
            #     xs = range(_len),
            #     ys=[x[0, :], x[1, :]],
            #     keys=['x', 'x_hat'],
            #     title='GT and prediction',
            #     xname=''
            # )
            # line_x = wandb.plot.line(table, x='i', y='x')
            # line_x_hat = wandb.plot.line(table, x='i', y='x_hat')
            # line_x_x_hat = wandb.plot.line(table, x='i', y='x_x_hat')
            # wandb.log({'val/x_prediction': line_x})
        
class LitLSTMAE(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, logger, config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # self.logger = logger
    
        self.config = config
        
        self.embs = []
        self.labels = []
        self.losses = []
        self.save_hyperparameters(ignore=['model'])
        
    def _clear_mem(self):
        self.embs.clear()
        self.labels.clear()
        self.losses.clear()
        
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.config.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        _, x, _ = batch
        _, x_hat = self.model(x)
        loss = self.loss_fn(x, x_hat)
        self.log('train/loss', loss)
        return loss
         
    def _shared_model_eval(self, batch, batch_idx):
        y, x, _ = batch
        z, x_hat = self.model(x)
        loss = self.loss_fn(x, x_hat)
        self.embs.append(z)
        self.losses.append(loss.item())
        return y, z, loss
        
    def validation_step(self, batch, batch_idx):
        labels, z, loss = self._shared_model_eval(batch, batch_idx)
        self.log('val/loss', loss)
        
    def test_step(self, batch, batch_idx):
        labels, z, loss = self._shared_model_eval(batch, batch_idx)
        self.log('test/loss', loss)
        
        
    def _shared_on_epoch_end(self, step_name):
        embs = torch.concat(self.embs, dim=0)
        embs = reduce_embed_dim(embs, pca_dim=self.config.pca_dim) # 2D coordinates
        # TODO process self.labels to be in ['norm', 'bad'] instead of [0, 1]
        fig = plot_embeddings(embs, self.labels)
        wandb.log({f'plots/{step_name}/embs': fig})
        


        # TODO supervised model validation, using reconstruction errors and labels. 
        # TODO plot ROC-AUC
    
    def on_validation_epoch_end(self):
        fig = self._shared_on_epoch_end(step_name='val')
        self._clear_mem()
        
    def on_test_epoch_end(self):
        fig = self._shared_on_epoch_end(step_name='test')
        self._clear_mem()
        
        
    def _shared_on_batch_end(self, batch, step_name=''):
        _, x, _ = batch
        x = x[0]
        x_hat = self.model(x)
        fig = plot_prediction(x, x_hat)
        wandb.log({f'plots/{step_name}/GT and prediction': fig})
        
        # log_predicted_signals(x, x_hat, step_name)
        
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self._shared_on_batch_end(batch, step_name='val')
            
        
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self._shared_on_batch_end(batch, step_name='test')
        
        
        
    
    

    
    