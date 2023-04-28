import torch
from torch import nn
import lightning.pytorch as pl



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
        hn.transpose_(0, 1) # batch first
        
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
    def __init__(self, encoder, decoder):
        super(LSTMAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        N, L, H = x.shape # Batch x Length x Feature size
        _, (hn, _) = self.encoder(x)
        x_hat, (_, _) = self.decoder(hn.expand(-1, L, -1)) # expand to stratch embedding over length of the input

        return (hn.squeeze(1), x_hat.flip(1))


class LitLSTMAE(pl.LightningModule):
    def __init__(self,):
        pass

    
    