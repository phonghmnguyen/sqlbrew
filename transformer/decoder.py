import torch.nn as nn
import torch.nn.functional as F
from . import (
    MultiHeadAttention,
    FeedForwardNetwork,
    ResidualConnection,
    Embedding,
    PositionalEncoding
)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn_hidden, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn_hidden)
        self.residual = nn.ModuleList([
            ResidualConnection(self.attn),
            ResidualConnection(self.enc_attn),
            ResidualConnection(self.ffn),
        ])

    def forward(self, dec_in, enc_out, mask=None):
        dec_out = self.residual[0](dec_in, mask=mask)
        dec_out = self.residual[1](None, dec_out, enc_out, enc_out)
        dec_out = self.residual[2](dec_out)
        return dec_out
    

class Decoder(nn.Module):
    def __init__(self, d_model, n_stack, n_head, d_ffn_hidden, n_vocab, dropout):
        super(Decoder, self).__init__()
        self.embed = Embedding(n_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=n_vocab)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.dec_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ffn_hidden, dropout)
              for _ in range(n_stack)
            ]
        )

    def forward(self, tgt_seq, enc_out, mask):
        embed = self.embed(tgt_seq)
        dec_out = self.dropout(self.pos_enc(embed))
        for dec in self.dec_stack:
            dec_out = dec(dec_out, enc_out, mask=mask)

        return dec_out

