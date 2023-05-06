import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ffn_hidden, dropout=0.1):
        """
        A two-layer feedforward neural network used in the Transformer.

        Args:
            d_model: The dimensionality of the embedding vector.
            d_ffn_hidden: The size of the hidden layer.
            dropout: The dropout regularization rate.
        """
        super(FeedForwardNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [   nn.Linear(d_model, d_ffn_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ffn_hidden, d_model)
            ]
        )

    def forward(self, x):
        """
            Passes the input tensor through the feedforward network.

            Args:
                x: The input tensor of shape `(batch_size, seq_len, d_model)`.

            Returns:
                The output tensor of shape `(batch_size, seq_len, d_model)`.
        """
        
        for layer in self.layers:
            x = layer(x)
            
        return x
