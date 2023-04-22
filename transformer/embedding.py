import math
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_vocab, d_model):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    