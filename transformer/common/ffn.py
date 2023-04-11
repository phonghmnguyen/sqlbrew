import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ffn_hidden, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [   nn.Linear(d_model, d_ffn_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ffn_hidden, d_model)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            y = layer(x)
        return y
