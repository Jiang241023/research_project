# head/node_head.py
import torch
from torch import nn
from torch_geometric.graphgym.register import register_head

@register_head('node')
class NodeHead(nn.Module):
    """Node-level predictor: per-node linear."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x_tokens, key_padding_mask=None):
        return self.proj(x_tokens)  # [B,N,dim_out]
