import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout = 0.1, alpha = 0.2):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.layers = [nn.BatchNorm1d(self.d_in), nn.Linear(d_in, d_hidden[0])]
        for i in range(1, len(d_hidden)):
            self.layers = self.layers + [
                nn.BatchNorm1d(d_hidden[i-1]),
                nn.Linear(d_hidden[i-1], d_hidden[i])
            ]
        self.layers = self.layers + [nn.BatchNorm1d(d_hidden[-1]), nn.Linear(d_hidden[-1], d_out)]
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=1.414)
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = x.view(-1, self.d_in)
        for layer in self.layers[:-1]:
            x = self.dropout(x)
            x = layer(x)
            x = self.relu(x)
        x = self.layers[-1](x)
        #if torch.isnan(x).any():
        #    raise RuntimeError('Found nan in forward pass.')
        x = x.clip(min=1e-6)
        return F.log_softmax(x, dim=1)

