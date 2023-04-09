import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super(FeedForwardNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(cfg.emb_dim, cfg.ffn_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.emb_dim)]
        )

    def forward(self, x):
        for layer in self.layers:
            y = layer(x)
        return y
