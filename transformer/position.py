import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        """
        A module for adding positional encoding to input sequences.

        Args:
            d_model: The dimensionality of embedding vector.
            dropout: The dropout regularization rate.
            max_len: The maximum length of a sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Apply positional encoding to an input sequence.

        Args:
            x: The input sequence tensor of shape (batch_size, seq_len, d_model).

        Returns:
            The input sequence tensor with positional encoding added, of shape (batch_size, seq_len, d_model).
        """
       
        x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)
