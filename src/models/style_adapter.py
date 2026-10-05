import torch.nn as nn


class StyleAdapter(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim))
    def forward(self, x):
        return self.net(x)