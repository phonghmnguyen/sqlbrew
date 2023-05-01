import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .ffn import FeedForwardNetwork
from .residual import ResidualConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn_hidden, dropout=0.1):
        """
        A single layer of the Transformer Encoder.

        Args:
            d_model: The dimensionality of the embedding vector.
            n_head: The number of parallel attention layers.
            d_ffn_hidden: The size of the hidden layer in the feedforward network.
            dropout: The dropout regularization rate. Defaults to 0.1.
        """
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn_hidden, dropout)
        self.resid_conns = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])


    def forward(self, x):
        """
        Passes the input tensor through a single layer of the encoder.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.

        Returns:
            The output tensor of shape `(batch_size, seq_len, d_model)`.
        """
        x = self.resid_conns[0](x, self.attn)
        return self.resid_conns[1](x, self.ffn)

        

class Encoder(nn.Module):
    def __init__(self, d_model, n_stack, n_head, d_ffn_hidden, dropout):
        """
        A stack of Transformer Encoder layers.

        Args:
            d_model: The dimensionality of embedding vector.
            n_stack: The number of Encoder layers to stack.
            n_head: The number of parallel attention layers.
            d_ffn_hidden: The size of the hidden layer in the feedforward network.
            dropout: The dropout regularization rate.
        """
        self.enc_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ffn_hidden, dropout)
              for _ in range(n_stack)
            ]
        )

    def forward(self, x, mask=None):
        """
        Passes the input tensor through the encoder stack.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.

        Returns:
            The output tensor of shape `(batch_size, seq_len, d_model)`.
        """
        for enc in self.enc_stack:
            x = enc(x, mask)

        return x


        