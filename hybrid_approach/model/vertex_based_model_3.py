# model/vertex_based_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- utilities ----

def scatter_softmax_per_dst(scores, dst, num_nodes):
    """
    Stable softmax over incoming edges for each destination node.
    scores: (E,)
    dst:    (E,) int64 dest indices
    returns alpha: (E,)
    """
    # per-node max for stability
    max_per_node = torch.full((num_nodes,), -1e30, device=scores.device)
    # PyTorch 2.0+: scatter_reduce_ supports 'amax'
    max_per_node.scatter_reduce_(0, dst, scores, reduce='amax', include_self=True)
    scores = scores - max_per_node[dst]

    exp_scores = torch.exp(scores)
    denom = torch.zeros(num_nodes, device=scores.device)
    denom.index_add_(0, dst, exp_scores)
    return exp_scores / (denom[dst] + 1e-9)


class FFN(nn.Module):
    def __init__(self, dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden = hidden_mult * dim
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.lin2(self.act(self.lin1(x))))


class VertexFlexibleBlock(nn.Module):
    """
    One GPS-style pre-norm block with GRIT-like flexible attention (vertex-only),
    degree injection after attention, and residual + FFN.
    Structural slice convention (last 4 channels of input features per node):
        [-4] = degree (normalized), [-3:] = xyz coordinates
    """
    def __init__(self, dim, structural_dim=4, heads=4, dropout=0.1, se_ratio=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        # Pre-norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Projections for attention
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)

        # Pairwise structural term (degree-diff + coord-diff → dim)
        self.W_S = nn.Linear(structural_dim, dim, bias=False)

        # Edge-value mixing (as in GRIT flexible attention)
        self.W_E = nn.Linear(dim, dim, bias=False)

        # Scalar scorer for attention logits (one per channel summed)
        self.w_a = nn.Parameter(torch.randn(dim) / (dim ** 0.5))

        # Degree gate (post-attention, channelwise)
        self.w_g = nn.Parameter(torch.zeros(dim))
        self.b_g = nn.Parameter(torch.zeros(dim))

        # FFN
        self.ffn = FFN(dim, hidden_mult=4, dropout=dropout)

        # Optional SE-like graph gate (kept lightweight; can be disabled by not calling)
        r = max(1, dim // se_ratio)
        self.se_fc1 = nn.Linear(dim, r, bias=True)
        self.se_fc2 = nn.Linear(r, dim, bias=True)

    @staticmethod
    def _split_structural(x):
        # last 4 dims: [deg_norm, x, y, z]
        deg = x[:, -4]                    # (n,)
        coords = x[:, -3:]                # (n,3)
        return deg, coords

    def forward(self, x, edge_index):
        """
        x: (n, dim)  — already includes structural channels at the end
        edge_index: (2, E)  — messages j->i (src=j, dst=i). Undirected edges recommended.
        """
        n = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        # ---- (A) Flexible attention (pre-norm) ----
        h = self.ln1(x)                                  # (n, d)
        Q, K, V = self.W_Q(h), self.W_K(h), self.W_V(h)  # (n, d) each

        # Gather per-edge endpoints
        q_i = Q[dst]             # (E, d)
        k_j = K[src]             # (E, d)
        v_j = V[src]             # (E, d)

        # Build simple structural deltas (degree and coordinates)
        deg, coords = self._split_structural(x)          # (n,), (n,3)
        s_ij = torch.stack([deg[dst] - deg[src]], dim=-1)    # (E,1)
        c_ij = coords[dst] - coords[src]                     # (E,3)
        s_concat = torch.cat([s_ij, c_ij], dim=-1)           # (E,4)
        s_term = self.W_S(s_concat)                          # (E,d)

        # GRIT-style pair feature then score
        xi = F.relu(q_i + k_j + s_term)                      # (E,d)
        logits = torch.einsum('ed,d->e', xi, self.w_a)       # (E,)

        alpha = scatter_softmax_per_dst(logits, dst, n)      # (E,)

        # Value path with pair feature injection
        v_mix = v_j + self.W_E(xi)                           # (E,d)
        m_e = alpha.unsqueeze(-1) * v_mix                    # (E,d)

        # Aggregate to destination vertices
        z = torch.zeros_like(x)
        z.index_add_(0, dst, m_e)                            # (n,d)

        # Degree gate after attention (channel-wise)
        gate = torch.sigmoid(deg.unsqueeze(-1) * self.w_g + self.b_g)  # (n,d)
        z = z * (1.0 + gate)

        # Residual 1
        y = x + self.dropout(z)

        # ---- (B) FFN with pre-norm ----
        y2 = y + self.dropout(self.ffn(self.ln2(y)))

        # ---- (C) Optional SE-style global gate (cheap) ----
        g = y2.mean(dim=0, keepdim=True)                    # READOUT: sum/mean over vertices
        s = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(g))))
        y2 = y2 * s                                         # broadcast

        return y2


class VertexFlexibleTransformer(nn.Module):
    """
    Purely vertex-based stack with flexible attention and degree injection.
    Input features are split as:
        base_feats (in_dim_base)  +  structural S (struct_dim = 4: [deg, x, y, z])
    """
    def __init__(
        self,
        in_dim_base: int = 31,
        struct_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        out_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim_base = in_dim_base
        self.struct_dim = struct_dim
        self.hidden_dim = hidden_dim

        # Input projections: base → hidden; structural PE → hidden, then add
        self.in_base = nn.Linear(in_dim_base, hidden_dim)
        self.in_pe = nn.Linear(struct_dim, hidden_dim, bias=False)

        self.blocks = nn.ModuleList([
            VertexFlexibleBlock(hidden_dim, structural_dim=struct_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.out_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        """
        x: (n, in_dim_base + struct_dim). The last 'struct_dim' channels are [deg_norm, x, y, z].
        edge_index: (2, E)
        """
        base = x[:, :self.in_dim_base]
        S = x[:, -self.struct_dim:]              # [deg, x, y, z]
        h = self.in_base(base) + F.silu(self.in_pe(S))   # X^0 = base + phi_pe(S)

        for blk in self.blocks:
            # concatenate S back so the block can slice degree/coords
            h = blk(torch.cat([h, S], dim=-1), edge_index)

            # strip structural tail for next block input
            h = h[:, :self.hidden_dim]

        h = self.out_ln(h)
        return self.head(h)                      # (n, out_dim)


