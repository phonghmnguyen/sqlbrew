import math
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_vocab, d_model, pad_idx):
        """
        A lookup table that stores token embeddings

        Args:
            n_vocab: The size of the corpus (number of distinct tokens).
            d_model: The dimensionality of the embedding vector.
            pad_idx: The index used for padding token.
        """
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(n_vocab, d_model, padding_idx=pad_idx)
        self.d_model = d_model

    def forward(self, x):
        """
        Maps token indices to token embeddings.

        Args:
            x: The input tensor of shape `(batch_size, seq_len)`.

        Returns:
            The output tensor of shape `(batch_size, seq_len, d_model)`.
        """
        return self.lut(x) * math.sqrt(self.d_model)
    