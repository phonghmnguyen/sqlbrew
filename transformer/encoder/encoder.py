import torch.nn as nn
import torch.nn.functional as F
from transformer.common import \
(
    MultiHeadAttention,
    FeedForwardNetwork,
    ResidualConnection,
    Embeddings,
    PositionalEncoding
)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(n_head, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn_hidden)
        self.residual = nn.ModuleList([
            ResidualConnection(self.attn),
            ResidualConnection(self.ffn),
        ])

    def forward(self, x):
        for layer in self.residual:
            x = layer(x)
        
        return x



class Encoder(nn.Module):
    def __init__(self, d_model, n_stack, n_head, d_ffn_hidden, corpus_len, dropout):
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ffn_hidden, dropout)
              for _ in range(n_stack)
            ]
        )

        # I decided to go with separated embeddings for the encoder and decoder approach
        # as opposed to the original transformer paper, because intuitively it gives
        # the model the ability to learn different representations for 2 different corpora (English vs SQL)
        self.emb = Embeddings(d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len=corpus_len)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        

    def forward(self, x):
        pass

        