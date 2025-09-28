import torch
import torch.nn as nn
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# --------------------------------------------
# GIN with row-chunked A@X (A/Ac can stay on CPU as bool)
# --------------------------------------------
class GINConv(nn.Module):
    """
    GIN update with row-chunked A@X so A/Ac can remain CPU-bool.
      Z = (1+eps) * X + A @ X
    """
    def __init__(self, in_dim, out_dim, eps=0.0, row_chunk: int = 512):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor([eps], dtype=torch.float32))
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.row_chunk = int(row_chunk)

    def forward(self, X, A):
        # X: (B,N,C) on GPU; A: (B,N,N) on CPU, dtype=bool
        B, N, C = X.shape
        device, dtype = X.device, X.dtype

        neigh = torch.zeros(B, N, C, device=device, dtype=dtype)
        # stream A rows in small blocks to GPU in the SAME dtype as X (fp16 under autocast)
        for i0 in range(0, N, self.row_chunk):
            i1 = min(N, i0 + self.row_chunk)
            A_blk = A[:, i0:i1, :].to(device=device, dtype=dtype, non_blocking=True)
            neigh[:, i0:i1, :] = torch.bmm(A_blk, X)
            del A_blk

        out = (1.0 + self.eps) * X + neigh
        out = self.linear(out)
        return torch.relu(out)

# --------------------------------------------
# Topological positional embedding (shared MPNN) with low-mem fusion
# --------------------------------------------
class TopoPositionalEmbedding(nn.Module):
    """
    TIGT-style topo PE with *shared* MPNN for A and A_c:
      h_A  = MPNN(X_h, A)
      h_Ac = MPNN(X_h, A_c)
      X0   = X_h + ReLU(h_A * θ[:,0]) + ReLU(h_Ac * θ[:,1])
    (avoid stacking [h_A,h_Ac] to save memory)
    """
    def __init__(self, in_dim, hidden_dim, row_chunk=512):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.mpn_shared = GINConv(hidden_dim, hidden_dim, row_chunk=row_chunk)
        # θ_pe ∈ R^{hidden_dim×2}
        self.theta = nn.Parameter(torch.empty(hidden_dim, 2))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, X, A, Ac):
        X_h  = self.embed(X)                           # (B,N,H)
        h_A  = self.mpn_shared(X_h, A)                 # (B,N,H)
        h_Ac = self.mpn_shared(X_h, Ac)                # (B,N,H)
        # lightweight fusion (no (B,N,H,2) allocation)
        t0 = self.theta[:, 0]
        t1 = self.theta[:, 1]
        X0 = X_h + torch.relu(h_A * t0) + torch.relu(h_Ac * t1)
        return X0

# --------------------------------------------
# Dual MPNN encoder layer (shared or separate weights)
# --------------------------------------------
class DualMPNNLayer(nn.Module):
    def __init__(self, hidden_dim, share_weights=True, row_chunk=512):
        super().__init__()
        self.share = share_weights
        if share_weights:
            self.mpn = GINConv(hidden_dim, hidden_dim, row_chunk=row_chunk)
        else:
            self.mpn_a  = GINConv(hidden_dim, hidden_dim, row_chunk=row_chunk)
            self.mpn_ac = GINConv(hidden_dim, hidden_dim, row_chunk=row_chunk)

    def forward(self, X_prev, A, Ac):
        if self.share:
            X_A  = self.mpn(X_prev, A)
            X_AC = self.mpn(X_prev, Ac)
        else:
            X_A  = self.mpn_a(X_prev, A)
            X_AC = self.mpn_ac(X_prev, Ac)
        return X_A, X_AC

# --------------------------------------------
# GRIT-style flexible attention with 2D tiling + online softmax
# (small tiles to keep peak memory low)
# --------------------------------------------
def _rho_signed_sqrt(x):
    # ρ(x) = sqrt(ReLU(x)) − sqrt(ReLU(−x))
    return torch.sqrt(torch.relu(x) + 1e-8) - torch.sqrt(torch.relu(-x) + 1e-8)

class FlexibleAttention(nn.Module):
    """
    Tile over queries (i) and keys (j), keep only small (qi×kj) chunks:
      e_ij = act(ρ(Wq x_i + Wk x_j))
      use online softmax to avoid a separate max pass
      y_i  = Σ_j softmax(scores_ij) * (Wy x_j + We e_ij)
    """
    def __init__(self, hidden_dim, attn_dim, act="relu", q_chunk=64, k_chunk=128):
        super().__init__()
        self.Wq   = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.Wk   = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.Wphi = nn.Linear(attn_dim, 1, bias=False)
        self.Wy   = nn.Linear(hidden_dim, hidden_dim)
        self.We   = nn.Linear(attn_dim,  hidden_dim)

        self.act = act
        self.q_chunk = int(q_chunk)
        self.k_chunk = int(k_chunk)

    def _act(self, z):
        z = _rho_signed_sqrt(z)
        return torch.relu(z) if self.act == "relu" else torch.tanh(z)

    def forward(self, X):
        # X: (B, N, C)  -- all computations happen on X.device
        B, N, C = X.shape
        q  = self.Wq(X)   # (B,N,d)
        k  = self.Wk(X)   # (B,N,d)
        Vy = self.Wy(X)   # (B,N,C)

        out   = X.new_zeros(B, N, C)
        # online softmax accumulators per i
        for i0 in range(0, N, self.q_chunk):
            i1 = min(N, i0 + self.q_chunk)
            q_blk = q[:, i0:i1]                         # (B, qi, d)

            out_blk   = X.new_zeros(B, i1 - i0, C)
            denom_blk = X.new_zeros(B, i1 - i0)
            m_blk     = X.new_full((B, i1 - i0), -float('inf'))

            for j0 in range(0, N, self.k_chunk):
                j1 = min(N, j0 + self.k_chunk)
                k_blk  = k[:, j0:j1]                   # (B, kj, d)
                Vy_blk = Vy[:, j0:j1]                  # (B, kj, C)

                e = self._act(q_blk.unsqueeze(2) + k_blk.unsqueeze(1))  # (B, qi, kj, d)
                scores = self.Wphi(e).squeeze(-1)                        # (B, qi, kj)

                # online softmax update
                m_new = torch.maximum(m_blk, scores.max(dim=-1).values)     # (B, qi)
                rescale = torch.exp(m_blk - m_new)
                out_blk   *= rescale.unsqueeze(-1)
                denom_blk *= rescale

                exp_scores = torch.exp(scores - m_new.unsqueeze(-1))        # (B, qi, kj)

                part1 = torch.einsum('bij,bjc->bic', exp_scores, Vy_blk)    # (B, qi, C)
                Ve    = self.We(e)                                          # (B, qi, kj, C)
                part2 = (exp_scores.unsqueeze(-1) * Ve).sum(dim=2)          # (B, qi, C)

                out_blk   = out_blk + part1 + part2
                denom_blk = denom_blk + exp_scores.sum(dim=-1)
                m_blk     = m_new

                del e, scores, m_new, rescale, exp_scores, part1, Ve, part2, k_blk, Vy_blk

            out[:, i0:i1, :] = out_blk / (denom_blk.unsqueeze(-1) + 1e-9)
            del q_blk, out_blk, denom_blk, m_blk

        return out

# --------------------------------------------
# Graph Info (Squeeze-and-Excitation)
# --------------------------------------------
class GraphInfoSqueezeExcite(nn.Module):
    def __init__(self, hidden_dim, reduction=4):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // reduction)
        self.lin2 = nn.Linear(hidden_dim // reduction, hidden_dim)
        self.mlp  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X_bar):
        B, N, C = X_bar.shape
        y0 = X_bar.sum(dim=1)                  # (B,C)
        y1 = torch.relu(self.lin1(y0))         # (B,C//r)
        y2 = torch.sigmoid(self.lin2(y1))      # (B,C)
        X_scaled = X_bar * y2.unsqueeze(1)     # (B,N,C)
        return self.mlp(X_scaled)              # (B,N,C)

# --------------------------------------------
# Full model
# --------------------------------------------
class VertexHybridModel(nn.Module):
    """
    TIGT backbone (shared MPNN for A & A_c) + tiled GRIT attention + SE info,
    per-node regression to 3D displacement.
    """
    def __init__(self, in_dim, hidden_dim, attn_dim, L=3, reduction=4, out_dim=3,
                 share_weights=True, attn_act="relu",
                 row_chunk=512, q_chunk=64, k_chunk=128):
        super().__init__()
        self.topo = TopoPositionalEmbedding(in_dim, hidden_dim, row_chunk=row_chunk)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'dual': DualMPNNLayer(hidden_dim, share_weights=share_weights, row_chunk=row_chunk),
                'attn': FlexibleAttention(hidden_dim, attn_dim, act=attn_act, q_chunk=q_chunk, k_chunk=k_chunk),
                'info': GraphInfoSqueezeExcite(hidden_dim, reduction)
            })
            for _ in range(L)
        ])
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, X, A, Ac):
        X_prev = self.topo(X, A, Ac)            # (B,N,H)
        for layer in self.layers:
            X_A, X_AC = layer['dual'](X_prev, A, Ac)
            X_att     = layer['attn'](X_prev)
            X_bar     = X_A + X_AC + X_att
            X_prev    = layer['info'](X_bar)
        return self.node_head(X_prev)           # (B,N,3)
