import torch.nn as nn
import torch.nn.functional as F

class ResidualConnection(nn.Module):
    def __init__(self, sublayer):
        super(ResidualConnection, self).__init__()
        self.sublayer = sublayer

    def forward(self, x):
        return F.layer_norm(x + self.sublayer(x), x.size(dim=-1))