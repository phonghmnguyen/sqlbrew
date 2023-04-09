import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        # embedding dimension must be divisible by number of heads
        assert cfg.emb_dim % cfg.n_head == 0
        
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim)
        # output projection
        self.c_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)

        self.n_head = cfg.n_head
        self.emb_dim = cfg.emb_dim
        
        # regularization
        self.dropout = cfg.dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)


    def forward(self, x, mask=None):
        # B: batch size, S: sequence length, E: embedding dimension
        B, S, E = x.size()
        # pull out the query, key, value from the concatenated projection
        q, k, v = self.c_attn(x).split(self.emb_dim, dim=2)
        # split heads and transpose to (B, n_head, S, E // n_head)
        q = q.view(B, S, self.n_head, E // self.n_head).transpose(1, 2)
        k = k.view(B, S, self.n_head, E // self.n_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, E // self.n_head).transpose(1, 2)
        # apply attention
        y = F.scaled_dot_product_attention(q, k, v, dropout=self.attn_dropout, mask=mask)
        # concatenate heads and transpose to (B, S, E)
        y = y.transpose(1, 2).contiguous().view(B, S, E)
        # apply drop out to final linear projection
        y = self.resid_dropout(self.c_proj(y))
        return y