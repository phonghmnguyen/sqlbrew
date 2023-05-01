import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .ffn import FeedForwardNetwork
from .residual import ResidualConnection


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ffn_hidden, dropout=0.1):
        """
        A single layer of the Transformer Decoder.

        Args:
            d_model: The dimensionality of embedding vector.
            n_head: The number of parallel attention layers.
            d_ffn_hidden: The size of the hidden layer in the feedforward network.
            dropout: The dropout regularization rate. Defaults to 0.1.
        """
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn_hidden)
        self.resid_conns = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(3)
        ])

    def forward(self, x, enc_out, mask=None, enc_mask=None):
        """
        Passes the input tensor through a single layer of the decoder.


        Args:
            x: The input tensor of shape (batch_size, seq_len, d_model).
            enc_out: The output tensor of the encoder of shape (batch_size, seq_len, d_model).
            mask: The mask tensor for masking out the subsequent positions of shape (batch_size, seq_len, seq_len).

        Returns:
            The output tensor of the decoder of shape (batch_size, seq_len, d_model).
        """
        x = self.resid_conns[0](x, lambda x: self.attn(x, mask=mask))
        x = self.resid_conns[1](x, lambda x: self.enc_attn(None, x, enc_out, enc_out, enc_mask))
        return self.resid_conns[2](x, self.ffn)
    

class Decoder(nn.Module):
    def __init__(self, d_model, n_stack, n_head, d_ffn_hidden, dropout):
        """
        A stack of Transformer Decoder layers.

        Args:
            d_model: The dimensionality of embedding vector.
            n_stack: The number of Decoder layers to stack.
            n_head: The number of parallel attention layers.
            d_ffn_hidden: The size of the hidden layer in the feedforward network.
            dropout: The dropout regularization rate.
        """
        super(Decoder, self).__init__()
        self.dec_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ffn_hidden, dropout)
              for _ in range(n_stack)
            ]
        )

    def forward(self, x, enc_out, mask=None, enc_mask=None):
        """
        Passes the input tensor through the decoder stack.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.

        Returns:
            The output tensor of shape `(batch_size, seq_len, d_model)`.
        """
        for dec in self.dec_stack:
            x = dec(x, enc_out, mask=mask, enc_mask=enc_mask)

        return x

