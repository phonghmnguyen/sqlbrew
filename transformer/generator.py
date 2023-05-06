import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, d_model, n_vocab):
        """
        A linear layer that maps from the decoder's output space to the vocabulary space which is then normalized
        with softmax to produce a probability distribution over the vocabulary.

        Args:
            d_model: The dimensionality of the model's output space.
            n_vocab: The size of the vocabulary.
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        """
            Computes the softmax function of the linear transformation applied on the input tensor.

            Args:
                x: The input tensor of shape `(batch_size, seq_len, d_model)`.

            Returns:
                The output tensor of shape `(batch_size, seq_len, n_vocab)`.
        """
        return self.proj(x)