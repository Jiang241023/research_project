# network/graphormer_backbone.py
import torch
from torch import nn
from torch_geometric.graphgym.register import register_network
from ..layer.graphormer_layer import GraphormerLayer

@register_network('GraphormerBackbone')
class GraphormerBackbone(nn.Module):
    """
    Stacks Graphormer layers. Expects dense inputs [B,N,d_model] and an attn_bias [B,N,N].
    If you use PyG graphs, build these via the collator in loader/collator_graphormer.py.
    """
    def __init__(self, dim_in: int, dim_hidden: int = 256, dim_out: int = 1,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.0,
                 mlp_ratio: float = 4.0, use_cls_token: bool = False):
        super().__init__()
        self.use_cls = use_cls_token
        self.proj_in = nn.Linear(dim_in, dim_hidden)
        self.layers = nn.ModuleList([
            GraphormerLayer(dim_hidden, num_heads, mlp_ratio, dropout, dropout, prenorm=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_hidden)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_hidden))

        # Expose a simple head here if wanted; or use head/ modules instead.
        self.out_head = nn.Linear(dim_hidden, dim_out)

    def forward(self, x_dense, attn_bias, key_padding_mask=None):
        """
        x_dense: [B,N,d_in]
        attn_bias: [B,N,N]
        key_padding_mask: [B,N] (True=real, False=pad)
        """
        B, N, _ = x_dense.shape
        x = self.proj_in(x_dense)

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            # expand attn bias by one token (cls)
            bias_pad = torch.zeros(B, 1, N, device=x.device, dtype=attn_bias.dtype)
            bias_cat_row = torch.zeros(B, N+1, 1, device=x.device, dtype=attn_bias.dtype)
            attn_bias = torch.cat([torch.cat([torch.zeros(B,1,1,device=x.device), bias_pad], dim=2),
                                   torch.cat([bias_cat_row, attn_bias], dim=2)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([torch.ones(B,1, dtype=torch.bool, device=x.device), key_padding_mask], dim=1)

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        return x  # return token representations; use head/ modules for readout
