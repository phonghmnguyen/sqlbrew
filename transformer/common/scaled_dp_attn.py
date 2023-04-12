import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn_score= torch.matmul(q, k.transpose(2, 3))

        # apply mask to upper diagonal submatrix, mask_dim: (B, n_head, S, S)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask==0, -1e9)

        # note: apply mask before softmax to get proper probability distribution
        attn_score = self.dropout(F.softmax(attn_score, dim=-1))
        return torch.matmul(attn_score, v)  