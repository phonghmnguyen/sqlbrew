import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))