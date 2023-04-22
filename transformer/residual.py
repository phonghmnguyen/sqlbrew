import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        """
        A residual connection used in the Transformer.

        Args:
            d_model: The dimensionality of the embedding vector.
            dropout: The dropout regularization rate. Defaults to 0.1.
        """
        super(ResidualConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Adds a residual connection to the output of a sublayer followed by layer normalization.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.
            sublayer: The sublayer to apply the residual connection to.

        Returns:
            The output tensor of the sublayer with a residual connection added and layer normalized.
        """
        return self.layer_norm(x + self.dropout(sublayer(x)))