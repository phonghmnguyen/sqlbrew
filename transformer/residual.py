import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return F.layer_norm(x + self.dropout(sublayer(x)), self.weight.shape, self.weight, self.bias)