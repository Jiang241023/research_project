# layer/graphormer_layer.py
import torch
from torch import nn
from .multihead_attention_bias import MultiheadSelfAttentionWithBias

class GraphormerLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0, dropout: float = 0.0, prenorm: bool = True):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttentionWithBias(d_model, n_head, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        if self.prenorm:
            y = self.attn(self.norm1(x), attn_bias, key_padding_mask)
            x = x + self.drop1(y)
            x = x + self.mlp(self.norm2(x))
            return x
        else:
            y = self.attn(x, attn_bias, key_padding_mask)
            x = self.norm1(x + self.drop1(y))
            x = self.norm2(x + self.mlp(x))
            return x
