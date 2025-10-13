# head/graph_head.py
import torch
from torch import nn
from torch_geometric.graphgym.register import register_head

@register_head('graph')
class GraphHead(nn.Module):
    """
    Graph-level readout: masked mean pool over nodes (or take CLS if present),
    followed by an MLP (1-layer here).
    """
    def __init__(self, dim_in: int, dim_out: int, use_cls_token: bool = False):
        super().__init__()
        self.use_cls = use_cls_token
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x_tokens, key_padding_mask=None):
        # x_tokens: [B,N,d]
        if self.use_cls:
            graph_repr = x_tokens[:, 0]  # first token is CLS
        else:
            if key_padding_mask is None:
                graph_repr = x_tokens.mean(dim=1)
            else:
                mask = key_padding_mask.float()  # [B,N], 1=real
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                graph_repr = (x_tokens * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.proj(graph_repr)
