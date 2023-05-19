import torch
from torch import nn
import lightning.pytorch as pl
import torch.nn.functional as F




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


class LitLSTMAE(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
        self.config = config
        
        self.embs = []
        self.val_embs = []
        self.test_embs = []
    
    
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
        
        
    def _shared_on_epoch_end(self):
        embs = torch.stack(self.embs)
        self.embs.clear()
        # TODO: reduce to 2D and visualize embeddings 
        # add figure to logger       
        
        # plot some signal and its reconstruction      
    
    def on_validation_epoch_end(self):
        self._shared_on_epoch_end()
        
    def on_test_epoch_end(self):
        self._shared_on_epoch_end()
    
    
        
        
        
        
        
    
    

    
    