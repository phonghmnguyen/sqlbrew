import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .ffn import FeedForwardNetwork
from .residual import ResidualConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn_hidden, dropout)
        self.resid_conns = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])


    def forward(self, x):
        x = self.resid_conns[0](x, self.attn)
        return self.resid_conns[1](x, self.ffn)

        

class Encoder(nn.Module):
    def __init__(self, d_model, n_stack, n_head, d_ffn_hidden, dropout):
        self.enc_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ffn_hidden, dropout)
              for _ in range(n_stack)
            ]
        )

    def forward(self, x):
        for enc in self.enc_stack:
            x = enc(x)

        return x


        