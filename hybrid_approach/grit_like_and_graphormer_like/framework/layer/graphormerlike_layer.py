import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add, scatter_max
from torch_geometric.graphgym.register import register_layer
from yacs.config import CfgNode as CN

# utils
def pyg_softmax(src, index, num_nodes=None):
    """Sparse softmax over src grouped by 'index' (dst)."""
    num_nodes = int(index.max().item()) + 1 if num_nodes is None else num_nodes
    max_per_index = scatter_max(src, index, dim=0, dim_size=num_nodes)[0]
    out = src - max_per_index[index]
    out = out.exp()
    out_sum = scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16
    return out / out_sum

def resolve_e2e(batch, E: int, device=None):
    e2e = None
    if hasattr(batch, "edge_edge_index"):
        e2e = batch.edge_edge_index
    elif hasattr(batch, "eei"):
        e2e = batch.eei
        if e2e.dim() == 2 and e2e.size(1) == 2:
            e2e = e2e.t().contiguous()
    else:
        z = torch.empty(2, 0, dtype=torch.long, device=device)
        return z[0], z[1]

    if device is None:
        device = e2e.device
    e2e = e2e.to(device=device, dtype=torch.long)

    if not (e2e.dim() == 2 and e2e.size(0) == 2):
        raise RuntimeError(
            f"edge_edge_index must be (2, M). Got {tuple(e2e.size())}. "
            "This usually means your Data class didn't define __cat_dim__/__inc__ "
            "for 'edge_edge_index'. Use LineGraphData."
        )
    return e2e[0], e2e[1]

class Attention(nn.Module):
    """
    Multi-head scaled dot-product attention over edge tokens.
    Returns: (E, embed_dim)
    """
    def __init__(self, in_dim, embed_dim, num_heads, bias=True, attn_dropout=0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.E_dim = embed_dim # The total dimension of the edge/token embedding
        self.H = num_heads # The number of attention heads
        self.D = embed_dim // num_heads # The per-head dimension
        self.scale = self.D ** -0.5 # The scaled dot-product factor

        # Separate Q/K/V (instead of one big qkv layer)
        self.Wq = nn.Linear(in_dim, embed_dim, bias=bias)
        self.Wk = nn.Linear(in_dim, embed_dim, bias=bias)
        self.Wv = nn.Linear(in_dim, embed_dim, bias=bias)

        self.out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(attn_dropout)

    # Reshape a per-edge embedding into multi-head format.
    def split_heads(self, t):
        # (E, E_dim) -> (E, H, D)
        E = t.size(0) # number of edge tokens
        return t.view(E, self.H, self.D)

    def forward(self, edge_features, e2e, attn_bias):
        E = edge_features.size(0) # 
        src_e, dst_e = e2e  # (M,), (M,)

        q = self.split_heads(self.Wq(edge_features))  # (E, H, D)
        k = self.split_heads(self.Wk(edge_features))  # (E, H, D)
        v = self.split_heads(self.Wv(edge_features))  # (E, H, D)

        q_dst = q.index_select(0, dst_e)     # (M, H, D)
        k_src = k.index_select(0, src_e)     # (M, H, D)

        # logits per (src,dst,head): dot over D
        logits = (q_dst * k_src).sum(-1) * self.scale  # (M, H)
        if attn_bias is not None:
            logits = logits + attn_bias                # broadcastable to (M, H)

        # softmax over sources for each destination edge-token
        attn = pyg_softmax(logits, index=dst_e, num_nodes=E)  # (M, H)
        attn = self.dropout(attn)

        # weighted sum of V[src] into each dst
        v_src = v.index_select(0, src_e)              # (M, H, D)
        msg = v_src * attn.unsqueeze(-1)              # (M, H, D)

        out = torch.zeros_like(v)                     # (E, H, D)
        scatter(msg, dst_e, dim=0, out=out, reduce='add')  # sum per dst

        # merge heads
        out = out.reshape(E, self.E_dim)              # (E, embed_dim)
        return self.out(out)  

# Graphormer Edge Layer (pre-LN)
@register_layer("GraphormerEdge")
class GraphormerEdgeLayer(nn.Module):
    """
    Pre-LN Transformer layer on EDGE tokens (line graph), edge→node update.

    Equations (edge stream):
      h'_e = MHA(LN(h_e)) + h_e
      h_e  = FFN(LN(h'_e)) + h'_e
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0, attn_dropout=0.0,
                 layer_norm=True, residual=True,
                 act='Gelu', update_nodes=True, cfg=CN(), **kwargs):
        super().__init__()
        # For strict Graphormer, model dim stays constant across sublayers.
        if in_dim == out_dim:
            self.proj_in  = nn.Identity()
        else:
             self.proj_in = nn.Linear(in_dim, out_dim)
        self.model_dim = out_dim
        self.num_heads = num_heads
        self.residual  = residual
        self.update_nodes = cfg.get("update_nodes", update_nodes)
        self.deg_coef_edges = nn.Parameter(torch.zeros(1, self.model_dim, 2))
        nn.init.xavier_normal_(self.deg_coef_edges)

        # Attention 
        if layer_norm == True:
            self.edge_layernorm_1 = nn.LayerNorm(self.model_dim) # A LayerNorm that will normalize each edge token’s feature vector (length = model_dim) before the attention (typical pre-LN pattern).
        else:
            self.edge_layernorm_1 = nn.Identity()
        self.attn_e = Attention(
            in_dim=self.model_dim, embed_dim=self.model_dim,
            num_heads=num_heads, attn_dropout=attn_dropout
        )
        self.drop_attn = nn.Dropout(dropout)

        # FFN (pre-LN): The FFN expands from model_dim to hidden_dim (typically 4×) then projects back.
        if layer_norm == True:
            self.edge_layernorm_2 = nn.LayerNorm(self.model_dim)
        else:
            self.edge_layernorm_2 = nn.Identity()
        hidden_mult = cfg.get("ffn_mult", 4)  # n set 2 if need d→2d→d
        hidden_dim = self.model_dim * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.GELU() if act == 'relu' else nn.GELU(),  # Graphormer uses GELU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.model_dim),
        )
        self.drop_ffn = nn.Dropout(dropout)

        # edge→node update 
        if layer_norm == True:
            self.layernorm_node = nn.LayerNorm(self.model_dim)
        else:
            self.layernorm_node = nn.Identity()
        self.edge_to_node_projection = nn.Linear(self.model_dim, self.model_dim)

    def forward(self, batch):
        # inputs
        edge_in = self.proj_in(batch.edge_attr)          # (E, d)
        node_in = batch.x                                # (N, d)   (assumed already at 'd')

        E = edge_in.size(0)
        device = edge_in.device
        src_e, dst_e = resolve_e2e(batch, E=E, device=device)

        # Optional Graphormer additive bias on edge attention logits
        # Expect shape (M, H); set to None if don't use it.
        attn_bias = getattr(batch, "edge_edge_bias", None)

        # Eq. (8): h'_e = MHA(LN(h_e)) + h_e 
        edge_residual = edge_in
        edge_norm = self.edge_layernorm_1(edge_in)
        edge_attn = self.attn_e(edge_norm, (src_e, dst_e), attn_bias=attn_bias)  # (E, d)
        edge_attn = self.drop_attn(edge_attn)
        if self.residual:
            updated_edge_features = edge_residual + edge_attn
        else:
            updated_edge_features = edge_attn

        # Eq. (9): h_e = FFN(LN(h'_e)) + h'_e 
        edge_residual_2 = updated_edge_features
        edge_norm2 = self.edge_layernorm_2(updated_edge_features)
        edge_ffn  = self.ffn(edge_norm2)
        edge_ffn  = self.drop_ffn(edge_ffn)
        if self.residual:
            edge_out = edge_residual_2 + edge_ffn
        else:
            edge_out = edge_ffn

        # Edge degree scaler
        log_deg_e = get_edge_log_deg(batch)                             # (E,1)
        h = torch.stack([edge_out, edge_out * log_deg_e], dim=-1)     # (E,d,2)
        edge_out = (h * self.deg_coef_edges).sum(dim=-1)              # (E,d)

        #  edge → node update 
        if self.update_nodes:
            # aggregate updated edge embeddings to both endpoints (undirected)
            if hasattr(batch, "edge_index_undirected"):
                node_i, node_j = batch.edge_index_undirected
            else:
                # if only directed is present, add to dst; duplicate for src if you want symmetry
                node_i, node_j = batch.edge_index

            node_residual  = node_in
            node_norm = self.layernorm_node(node_in)

            # Aggregate updated edge embeddings to both endpoints
            node_msg = torch.zeros_like(node_norm)
            scatter_add(edge_out, node_i, dim=0, out=node_msg) # sum to i
            scatter_add(edge_out, node_j, dim=0, out=node_msg) # sum to j

            node_delta = self.edge_to_node_projection(node_msg)
            if self.residual:
                node_out = node_residual + node_delta
            else: 
                node_out = node_delta
        else:
            node_out = node_in

        batch.edge_attr = edge_out
        batch.x = node_out
        return batch

@torch.no_grad()
def get_edge_log_deg(batch, cache=True):
    E = batch.edge_attr.size(0) if hasattr(batch, "edge_attr") else batch.edge_index.size(1)
    src_e, dst_e = resolve_e2e(batch, E, device=batch.edge_index.device)
    ones = torch.ones_like(dst_e, dtype=torch.float)
    deg_e = scatter_add(ones, dst_e, dim=0, dim_size=E)  # (E,)
    log_deg_e = torch.log1p(deg_e)                       # (E,)
    if cache:
        batch.deg_e = deg_e
        batch.log_deg_e = log_deg_e
    return log_deg_e.view(-1, 1)