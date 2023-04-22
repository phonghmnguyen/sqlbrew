import torch.nn as nn
import torch.nn.functional as F
from . import (
    MultiHeadAttention,
    FeedForwardNetwork,
    ResidualConnection,
    Embedding,
    PositionalEncoding
)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn_hidden, dropout)
        self.residual = nn.ModuleList([
            ResidualConnection(self.attn),
            ResidualConnection(self.ffn),
        ])

    def forward(self, enc_in):
        enc_out = enc_in
        for layer in self.residual:
            enc_out = layer(enc_out)
        
        return enc_out



class Encoder(nn.Module):
    def __init__(self, d_model, n_stack, n_head, d_ffn_hidden, n_vocab, dropout):
        # I decided to go with separated embeddings for the encoder and decoder approach
        # as opposed to the original transformer paper, because intuitively it gives
        # the model the ability to learn different representations for 2 different corpora (English vs SQL)
        self.embed = Embedding(n_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=n_vocab)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.enc_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ffn_hidden, dropout)
              for _ in range(n_stack)
            ]
        )

        
    def forward(self, src_seq):
        embed = self.embed(src_seq)
        enc_out = self.dropout(self.pos_enc(embed))
        for enc in self.enc_stack:
            enc_out = enc(enc_out)

        return enc_out  


        