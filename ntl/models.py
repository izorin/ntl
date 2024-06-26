from typing import DefaultDict
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# import lightning.pytorch as pl
import pytorch_lightning as pl
import torch.nn.functional as F

import wandb
import numpy as np
import plotly.express as px

from sequitur.models import LSTM_AE
from sequitur.models.lstm_ae import Encoder, Decoder

from ntl.utils import conv2d_shape, convtraspose2d_shape




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
        # roc_auc_table = wandb.roc_auc_table(columns=['x', 'y'], data=embs)
        # wandb.log({'val/embs_2D': wandb.plot.scatter(roc_auc_table, 'x', 'y')})
        
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
            # roc_auc_table = wandb.roc_auc_table(data=data, columns=['i', 'x', 'x_hat', 'x - x_hat'])
            # line_x = wandb.plot.line_series(
            #     xs = range(_len),
            #     ys=[x[0, :], x[1, :]],
            #     keys=['x', 'x_hat'],
            #     title='GT and prediction',
            #     xname=''
            # )
            # line_x = wandb.plot.line(roc_auc_table, x='i', y='x')
            # line_x_hat = wandb.plot.line(roc_auc_table, x='i', y='x_hat')
            # line_x_x_hat = wandb.plot.line(roc_auc_table, x='i', y='x_x_hat')
            # wandb.log({'val/x_prediction': line_x})
        
class LitLSTMAE(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, scheduler, config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
    
        self.config = config
        self.tb_logger = SummaryWriter(log_dir=self.config.save_dir, comment='debug')
        
        self.embs = DefaultDict(list)
        self.labels = DefaultDict(list)
        self.losses = DefaultDict(list)
        # self.roc_auc_table = wandb.Table(columns=['epoch', 'FPR', 'TPR', 'AUC']) # epoch == self.current_epoch
        
        self.save_hyperparameters(ignore=['model'])
        wandb.watch(self.model, log='all')
        
        
        
    def _clear_mem(self, step_name):
        self.embs[step_name] = []
        self.labels[step_name] = []
        self.losses[step_name] = []
    
    def clear_mem(self):
        for key in self.embs.keys(): # key in ['train', 'val', 'test']
            self._clear_mem(key)
            
        
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.config.optimizer_kwargs)
        scheduler = self.scheduler(optimizer, **self.config.scheduler_kwargs)

        return {
            'optimizer': optimizer,
            'lr_scheduler' : {
                'scheduler': scheduler,
                'monitor': 'train/loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': False,
                'name': 'LR'
            }
        }
    
    
    def _shared_model_step(self, batch, batch_idx, step_name):
        y, x, _ = batch
        z, x_hat = self.model(x)
        loss = self.loss_fn(x, x_hat, reduction='none')
        self.embs[step_name].append(z.detach().cpu())
        self.losses[step_name].append(loss.detach().cpu())
        self.labels[step_name].append(y.detach().cpu())
        return loss.mean()
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_model_step(batch, batch_idx, step_name='train')
        self.log('train/loss', loss, batch_size=self.config.batch_size)
        self.tb_logger.add_scalar('train/loss', loss.item(), self.global_step)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self._shared_model_step(batch, batch_idx, step_name='val')
        self.log('val/loss', loss, batch_size=self.config.batch_size)
        self.tb_logger.add_scalar('val/loss', loss.item(), self.global_step)
        
        
    def test_step(self, batch, batch_idx):
        loss = self._shared_model_step(batch, batch_idx, step_name='test')
        self.log('test/loss', loss, batch_size=self.config.batch_size)
        self.tb_logger.add_scalar('test/loss', loss.item(), self.global_step)
        
        
    def _shared_on_epoch_end(self, step_name):
        self.embs[step_name] = torch.concat(self.embs[step_name], dim=0).numpy()
        self.losses[step_name] = torch.concat(self.losses[step_name], dim=0).numpy().squeeze().sum(axis=1) # sum over seq length
        self.labels[step_name] = torch.concat(self.labels[step_name], dim=0).numpy()
        
        # plot embeddings
        self.embs[step_name] = reduce_embed_dim(self.embs[step_name], pca_dim=self.config.pca_dim) # 2D coordinates
        fig, fig_plotly = plot_embeddings(self.embs[step_name], self.labels[step_name], pyplot=True)
        wandb.log({f'{step_name}/embs': fig})
        self.tb_logger.add_figure(f'{step_name}/embs', fig_plotly, self.global_step)
        
        # plot reconstruction errors hist
        fig = rec_error_hist(self.losses[step_name], self.labels[step_name])
        wandb.log({f'{step_name}/error_hist': fig})
        # self.tb_logger.add_figure(f'{step_name}/embs', fig, self.global_step) # FIXME
        
        
    def on_train_epoch_end(self): 
        # NOTE
        # train_epoch_start() -> val_epoch_start() -> val_epoch_end() -> train_epoch_end()
        # therefore self.embs() of train stage are reset in on_val_epoch_end() 
        self._shared_on_epoch_end(step_name='train')
        self._clear_mem(step_name='train')
    
    def _unsupervised_validation(self, step_name):
        '''
        validates with unlabeled data; only computes and plots embeddings and reconstruction errors
        '''
        self._shared_on_epoch_end(step_name)
    
    def _supervised_validation(self, step_name):
        '''
        does unsupervised validation and in addition utilize labels to compute roc-auc using reconstruction errors
        '''
        self._shared_on_epoch_end(step_name)
        
        # supervised tests
        fig, (FPR, TPR, auc_score) = compute_roc_auc(self.losses[step_name], self.labels[step_name])
        wandb.log({f'{step_name}/roc-auc': fig})
        wandb.log({f'{step_name}/auc': auc_score})

        self.tb_logger.add_scalar(f'{step_name}/auc', auc_score, self.global_step)
        # self.tb_logger.add_figure(f'{step_name}/roc-auc', fig, self.global_step) # FIXME
        # self.roc_auc_table.add_data(self.current_epoch, FPR, TPR, auc_score)
         
    def on_validation_epoch_end(self):
        if self.config.supervised_validation:
            self._supervised_validation('val')
        else:
            self._unsupervised_validation('val')
            
        self._clear_mem('val')
            
    def on_test_epoch_end(self):
        self._supervised_validation('test')
        self._clear_mem(step_name='test')

        
    def _shared_on_batch_end(self, batch, step_name=''):
        _, x, _ = batch
        idx = np.random.randint(low=0, high=x.shape[0])
        x = x[idx:idx+1] # == x[idx] keeping shape
        _, x_hat = self.model(x)
        x = x.detach().cpu().numpy().squeeze()
        x_hat = x_hat.detach().cpu().numpy().squeeze()
        fig = plot_prediction(x, x_hat)
        wandb.log({f'{step_name}/GT and prediction': fig})
        # self.tb_logger.add_figure(f'{step_name}/GT and prediction', fig, self.global_step) # FIXME
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self._shared_on_batch_end(batch, step_name='train')
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self._shared_on_batch_end(batch, step_name='val')
            
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self._shared_on_batch_end(batch, step_name='test')
        

    # def on_test_end(self):
        # wandb.log({'roc-auc-table': self.roc_auc_table})
    



class SeqEncoder(Encoder):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        
        super().__init__(input_dim, out_dim, h_dims, h_activ, out_activ)
        
    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()
        
    
class SeqDecoder(Decoder):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super().__init__(input_dim, out_dim, h_dims, h_activ)
        
    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1) # unsqueeze sequence length's dimension, and repeat embedding over it
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.matmul(x, self.dense_matrix)
        
    
    
class SequiturLSTMAE(nn.Module):
    def __init__(self, input_dim, encoding_dim, h_dims=[], h_activ=nn.Sigmoid(), out_activ=nn.Tanh()):
        super().__init__()

        h_activ = getattr(nn, h_activ)()
        out_activ = getattr(nn, out_activ)()
        
        self.encoder = SeqEncoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        
        self.decoder = SeqDecoder(encoding_dim, input_dim, h_dims[::-1], h_activ)
        
    def forward(self, x):
        seq_len = x.shape[1]
        z = self.encoder(x)
        x = self.decoder(z, seq_len)
        
        return z, x
        
        
    
class BaseAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, emb):
        return self.decoder(emb)
        
    def forward(self, x):
        emb = self.encoder(x)
        x = self.decoder(emb)
        emb = emb.reshape(emb.shape[0], -1)
        
        return emb, x
        
    
class AE2dCNN(BaseAE):
    def __init__(self, bias=True):
        super().__init__()
        self.bias = bias
        channels = [1, 4, 16, 32, 64]
        layers = []
        for i in range(len(channels) - 1):

            layers += [
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], stride=1, kernel_size=3, padding=0, bias=bias),
                nn.BatchNorm2d(num_features=channels[i+1]),
                nn.Dropout(0.15),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2, padding=1)
            ]
            
        self.encoder = nn.Sequential(*layers)
        
        channels.reverse()

        layers = []
        for i in range(len(channels) - 1):

            layers += [
                nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i+1], stride=1, kernel_size=3, padding=0, bias=bias),
                nn.BatchNorm2d(num_features=channels[i+1]),
                nn.Dropout(0.15),
                nn.ReLU()
            ]
            
        self.decoder = nn.Sequential(*layers)
            
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, emb):
        return self.decoder(emb)
        
    def forward(self, x):
        emb = self.encoder(x)
        x = self.decoder(emb)
        emb = emb.reshape(emb.shape[0], -1)
        
        return emb, x
    
    
class AE1dCNN(BaseAE):
    def __init__(self):
        super().__init__()
        
        conv1d_layers = []
        conv1d_strided_layers = []
        conv1d_dims = [1, 4, 8, 16, 32]
        conv1d_strided_dims = [32, 64, 128, 256, 512]
        kernels = [3, 5, 7, 5]

        for i in range(len(conv1d_dims) - 1):
            conv1d_layers += [
                nn.Conv1d(in_channels=conv1d_dims[i], out_channels=conv1d_dims[i+1],kernel_size=7, stride=1, dilation=3, bias=False),
                nn.Dropout(0.15),
                nn.BatchNorm1d(conv1d_dims[i+1]),
                nn.LeakyReLU()
            ]
            
        for i in range(len(conv1d_strided_dims) - 1):
            conv1d_strided_layers += [
                nn.Conv1d(in_channels=conv1d_strided_dims[i], out_channels=conv1d_strided_dims[i+1], kernel_size=kernels[i], stride=2, dilation=1, bias=False),
                nn.Dropout(0.15),
                nn.BatchNorm1d(conv1d_strided_dims[i+1]),
                nn.LeakyReLU()    
            ]

        linear_layers = [nn.Linear(in_features=11, out_features=1),
                         nn.Dropout(0.15),
                         nn.ReLU(),
                         nn.Flatten(1,2),
                         nn.Linear(512, 512),
                         nn.Dropout(0.15),
                         nn.ReLU()
                         ]
        
        encoder = conv1d_layers + conv1d_strided_layers + linear_layers
        self.encoder = nn.Sequential(*encoder)

        reverse_linear_layers = [
            nn.Linear(512, 512),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1)),
            nn.Linear(1, 11),
            nn.Dropout(0.15),
            nn.ReLU()
            ]

        kernels.reverse()
        conv1d_strided_dims.reverse()
        conv1d_dims.reverse()
        output_paddings = [0, 1, 0, 1]
        reverse_conv1d_strided_layers = []
        reverse_conv1d_layers = []
        
        for i in range(len(conv1d_strided_dims) - 1):
            reverse_conv1d_strided_layers += [
                nn.ConvTranspose1d(in_channels=conv1d_strided_dims[i], out_channels=conv1d_strided_dims[i+1], kernel_size=kernels[i], stride=2, output_padding=output_paddings[i], bias=False),
                nn.Dropout(0.15),
                nn.BatchNorm1d(conv1d_strided_dims[i+1]),
                nn.LeakyReLU()
            ]
            
        for i in range(len(conv1d_dims) - 1):
            reverse_conv1d_layers += [
                nn.ConvTranspose1d(in_channels=conv1d_dims[i], out_channels=conv1d_dims[i+1], kernel_size=7, dilation=3, bias=False),
                nn.Dropout(0.15),
                nn.BatchNorm1d(conv1d_dims[i+1]),
                nn.LeakyReLU()
            ]
            
        decoder = reverse_linear_layers + reverse_conv1d_strided_layers + reverse_conv1d_layers
        self.decoder = nn.Sequential(*decoder)
            
    def encode(self, x):
        if x.shape[1] != 1:
            x.transpose_(1,2)
        return self.encoder(x)
    
    def decode(self, emb):
        return self.decoder(emb)
        
    def forward(self, x):
        if x.shape[1] != 1:
            x.transpose_(1,2)
            
        #####################
        emb = self.encoder(x)
        x = self.decoder(emb)
        emb = emb.reshape(emb.shape[0], -1)
        # probably can be replaces with the following line 
        # emb, x = super().forward(x)
        
        return emb, x
    
    
class VAE2dCNN(BaseAE):
    def __init__(self, bias=False):
        super().__init__()
        self.bias = bias
        channels = [1, 4, 16, 32, 64]
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], stride=1, kernel_size=3, padding=0, bias=bias),
                nn.BatchNorm2d(num_features=channels[i+1]),
                nn.Dropout(0.15),
                nn.ReLU(),

            ]
            
        self.encoder = nn.Sequential(*layers)
        
        channels.reverse()

        layers = []
        for i in range(len(channels) - 1):

            layers += [
                nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i+1], stride=1, kernel_size=3, padding=0, bias=bias),
                nn.BatchNorm2d(num_features=channels[i+1]),
                nn.Dropout(0.15),
                nn.ReLU()
            ]
            
        self.decoder = nn.Sequential(*layers)
            