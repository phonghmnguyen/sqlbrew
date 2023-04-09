import torch.nn as nn
import torch.nn.functional as F
from transformer.common import MultiHeadAttention, FeedForwardNetwork



class EncoderLayer(nn.Module):
    def __init__(self, attn, ffn, dropout):
        super(EncoderLayer, self).__init__()
        