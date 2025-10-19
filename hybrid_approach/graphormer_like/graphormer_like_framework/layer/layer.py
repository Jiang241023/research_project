import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_add, scatter_max
from torch_geometric.graphgym.register import register_layer
from yacs.config import CfgNode as CN

def pyg_softmax(src, index, num_nodes=None):
    """Sparse softmax over src grouped by index."""
    num_nodes = int(index.max().item()) + 1 if num_nodes is None else num_nodes
    max_per_index = scatter_max(src, index, dim=0, dim_size=num_nodes)[0]
    out = src - max_per_index[index]
    out = out.exp()
    out_sum = scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16
    return out / out_sum

def _build_edge_edge_index(edge_index: torch.Tensor, num_edges: int, num_nodes: int):
    """
    Build line-graph connectivity (edge->edge) where two original edges are adjacent
    if they share a node. Returns e_edge_index of shape (2, M).
    Works on the directed edge list.
    """
    device = edge_index.device
    E = num_edges
    src, dst = edge_index  # (E,), (E,)

    nodes = torch.cat([src, dst], dim=0)  # (2E,)
    eids  = torch.cat([torch.arange(E, device=device),
                       torch.arange(E, device=device)], 0)

    order = torch.argsort(nodes)
    nodes_sorted = nodes[order]
    eids_sorted  = eids[order]

    _, counts = torch.unique_consecutive(nodes_sorted, return_counts=True)

    e_row, e_col = [], []
    start = 0
    for c in counts.tolist():
        if c > 1:
            group = eids_sorted[start:start + c]      # incident edges at this node
            g1 = group.repeat_interleave(c)
            g2 = group.repeat(c)
            mask = g1 != g2
            e_row.append(g1[mask])
            e_col.append(g2[mask])
        start += c

    if not e_row:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    e_row = torch.cat(e_row, dim=0)
    e_col = torch.cat(e_col, dim=0)

    # Deduplicate pairs that appear from both endpoints
    key = e_row * E + e_col
    uniq = torch.unique(key)
    e_row = (uniq // E).to(torch.long)
    e_col = (uniq %  E).to(torch.long)
    return torch.stack([e_row, e_col], dim=0)

class MultiHeadAttentionGraphormerEdge(nn.Module):
    """
    Multi-head self-attention for edge tokens without relation-aware biases.
    """
    def __init__(self, in_dim, out_dim, num_heads, use_bias=True,
                 dropout=0.0, act=None, clamp=None, **kwargs):
        super().__init__()
        assert out_dim % 1 == 0
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        self.Q = nn.Linear(in_dim, num_heads * out_dim, bias=True)
        self.K = nn.Linear(in_dim, num_heads * out_dim, bias=use_bias)
        self.V = nn.Linear(in_dim, num_heads * out_dim, bias=use_bias)

        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.V.weight)

        # project per-head feature vector -> scalar logit
        self.Aw = nn.Parameter(torch.zeros(out_dim, num_heads, 1))
        nn.init.xavier_normal_(self.Aw)

        self.act = act if act is not None else nn.Identity()
        self.clamp = np.abs(clamp) if clamp is not None else None

    def forward(self, batch):
        E = batch.edge_attr                      # (E, Fe)
        num_edges = E.size(0)

        # Ensure line-graph connectivity exists
        e_ei = getattr(batch, "edge_edge_index", None)
        if e_ei is None:
            e_ei = _build_edge_edge_index(batch.edge_index, num_edges, batch.num_nodes)
            batch.edge_edge_index = e_ei

        Q_h = self.Q(E).view(-1, self.num_heads, self.out_dim)  # (E, H, D)
        K_h = self.K(E).view(-1, self.num_heads, self.out_dim)
        V_h = self.V(E).view(-1, self.num_heads, self.out_dim)

        src_edges = e_ei[0]  # edge-token ids
        dst_edges = e_ei[1]

        score_vec = K_h[src_edges] + Q_h[dst_edges]             # (M, H, D)
        score_vec = self.act(score_vec)

        raw_attn = torch.einsum('ehd,dhc->ehc', score_vec, self.Aw).squeeze(-1)  # (M, H)

        if self.clamp is not None:
            raw_attn = torch.clamp(raw_attn, -self.clamp, self.clamp)

        attn_weights = pyg_softmax(raw_attn, index=dst_edges, num_nodes=num_edges)  # (M, H)
        attn_weights = self.dropout(attn_weights)

        msg = V_h[src_edges] * attn_weights.unsqueeze(-1)       # (M, H, D)
        edge_out = torch.zeros_like(V_h)                         # (E, H, D)
        scatter(msg, dst_edges, dim=0, out=edge_out, reduce='add')

        return edge_out, None

@register_layer("GraphormerEdge")
class GraphormerEdgeLayer(nn.Module):
    """
    Transformer encoder layer that operates on edge tokens; optionally updates nodes.
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0, attn_dropout=0.0,
                 layer_norm=True, batch_norm=True, residual=True,
                 act='relu', update_nodes=True, cfg=CN(), **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.residual = residual
        self.update_nodes = cfg.get("update_nodes", update_nodes)

        attn_kwargs = cfg.get("attn", {})
        self.attention = MultiHeadAttentionGraphormerEdge(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            dropout=attn_dropout,
            act=nn.ReLU() if act == 'relu' else nn.Identity(),
            clamp=attn_kwargs.get("clamp", None),
        )

        self.O_e = nn.Linear(out_dim, out_dim) if cfg.attn.get("O_e", True) else nn.Identity()
        self.O_v = nn.Linear(out_dim, out_dim) if self.update_nodes else nn.Identity()

        self.norm_e = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.norm_v = nn.LayerNorm(out_dim) if (self.update_nodes and layer_norm) else nn.Identity()
        self.bn_e = nn.BatchNorm1d(out_dim, momentum=cfg.get("bn_momentum", 0.1)) if batch_norm else nn.Identity()
        self.bn_v = nn.BatchNorm1d(out_dim, momentum=cfg.get("bn_momentum", 0.1)) if (self.update_nodes and batch_norm) else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU() if act == 'relu' else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

        self.norm_e2 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.bn_e2 = nn.BatchNorm1d(out_dim, momentum=cfg.get("bn_momentum", 0.1)) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        edge_in = batch.edge_attr
        node_in = batch.x
        num_edges = edge_in.shape[0]

        # Edge self-attention
        edge_attn_out, _ = self.attention(batch)          # (E, H, D)
        edge_attn_out = edge_attn_out.view(num_edges, -1) # (E, out_dim)
        edge_attn_out = self.O_e(edge_attn_out)
        edge_attn_out = self.dropout(edge_attn_out)

        edge_out = edge_in + edge_attn_out if self.residual else edge_attn_out
        edge_out = self.norm_e(edge_out)
        if isinstance(self.bn_e, nn.BatchNorm1d):
            edge_out = self.bn_e(edge_out)

        # Optional: edge->node aggregation (add to both endpoints if available)
        if self.update_nodes:
            if hasattr(batch, "edge_index_undirected"):
                u = batch.edge_index_undirected[0]
                v = batch.edge_index_undirected[1]
                node_out = torch.zeros_like(node_in)
                # add each edge contribution to both endpoints (using same E length)
                scatter_add(edge_out, u, dim=0, out=node_out)
                scatter_add(edge_out, v, dim=0, out=node_out)
            else:
                # Fallback: add to target only (consistent with GRIT’s destination aggregation)
                node_out = torch.zeros_like(node_in)
                scatter_add(edge_out, batch.edge_index[1], dim=0, out=node_out)

            node_out = self.O_v(node_out)
            node_out = self.dropout(node_out)
            node_out = node_in + node_out if self.residual else node_out
            node_out = self.norm_v(node_out)
            if isinstance(self.bn_v, nn.BatchNorm1d):
                node_out = self.bn_v(node_out)
        else:
            node_out = node_in

        # FFN on edges
        edge_ffn = self.ffn(edge_out)
        edge_ffn = self.dropout(edge_ffn)
        edge_ffn = edge_out + edge_ffn if self.residual else edge_ffn
        edge_ffn = self.norm_e2(edge_ffn)
        if isinstance(self.bn_e2, nn.BatchNorm1d):
            edge_ffn = self.bn_e2(edge_ffn)

        batch.edge_attr = edge_ffn
        batch.x = node_out
        return batch
