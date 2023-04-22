import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .ffn import FeedForwardNetwork
from .residual import ResidualConnection


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn_hidden, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn_hidden)
        self.resid_conns = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(3)
        ])

    def forward(self, x, enc_out, mask=None):
        x = self.resid_conns[0](x, lambda x: self.attn(x, mask=mask))
        x = self.resid_conns[1](x, lambda x: self.enc_attn(x, enc_out, enc_out))
        return self.resid_conns[2](x, self.ffn)
    

class Decoder(nn.Module):
    def __init__(self, d_model, n_stack, n_head, d_ffn_hidden, dropout):
        super(Decoder, self).__init__()
        self.dec_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ffn_hidden, dropout)
              for _ in range(n_stack)
            ]
        )

    def forward(self, x, enc_out, mask=None):
        for dec in self.dec_stack:
            x = dec(x, enc_out, mask=mask)

        return x

