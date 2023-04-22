import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, d_model, n_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return F.softmax(self.proj(x), dim=-1)