import torch.nn as nn
import torch.nn.functional as F

class ResidualConnection(nn.Module):
    def __init__(self, sublayer):
        super(ResidualConnection, self).__init__()
        self.sublayer = sublayer

    def forward(self, x, *args, sublayer_mask=None):
        return F.layer_norm(x + self.sublayer(x, args, mask=sublayer_mask), x.size(dim=-1))