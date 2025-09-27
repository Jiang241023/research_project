# model/vertex_based_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- per-graph mean over nodes ---
def segment_mean(X, batch_idx, num_groups):
    sums = X.new_zeros(num_groups, X.size(1))
    sums.index_add_(0, batch_idx, X)
    counts = torch.bincount(batch_idx, minlength=num_groups).clamp_min(1).unsqueeze(1)
    return sums / counts

class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_out),
            nn.Dropout(drop),
        )
    def forward(self, x): return self.net(x)

class LocalMPNN(nn.Module):
    """
    Edge-based (O(E)) message passing that works on concatenated batches.
    X: (sumN, F), edge_index: (2, sumE) with edges src<-dst
    """
    def __init__(self, dim, eps=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(float(eps)))
        self.lin = nn.Linear(dim, dim)
        self.msg_lin = nn.Linear(dim, dim)

    def forward(self, X, edge_index):
        src, dst = edge_index  # (E,), (E,)
        msg = self.msg_lin(X)[dst]               # (E, F)
        agg = X.new_zeros(X.size(0), X.size(1))  # (sumN, F)
        agg.index_add_(0, src, msg)
        out = (1.0 + self.eps) * X + agg
        out = self.lin(out)
        return F.relu(out)

class EdgeFlexibleAttention(nn.Module):
    """
    GRIT-style flexible attention, restricted to edges (O(E)).
    """
    def __init__(self, dim, attn_dim):
        super().__init__()
        self.Wq   = nn.Linear(dim, attn_dim, bias=False)
        self.Wk   = nn.Linear(dim, attn_dim, bias=False)
        self.Wphi = nn.Linear(attn_dim, 1, bias=False)
        self.Wy   = nn.Linear(dim, dim)
        self.We   = nn.Linear(attn_dim, dim)

    def forward(self, X, edge_index):
        N, Fdim = X.shape
        src, dst = edge_index
        q = self.Wq(X)[src]               # (E, d)
        k = self.Wk(X)[dst]               # (E, d)
        e = torch.tanh(q + k)             # (E, d)
        scores = self.Wphi(e).squeeze(-1) # (E,)

        scores = scores - scores.max()
        exp_scores = torch.exp(scores)
        denom = X.new_zeros(N)
        denom.index_add_(0, src, exp_scores)
        alpha = exp_scores / (denom[src] + 1e-9)

        vy = self.Wy(X)[dst]              # (E, F)
        ve = self.We(e)                   # (E, F)
        msg = alpha.unsqueeze(-1) * (vy + ve)  # (E, F)

        out = X.new_zeros(N, Fdim)
        out.index_add_(0, src, msg)
        return out

class GPSFlexBlock(nn.Module):
    """
    PreNorm(Local) + PreNorm(Attention) + PreNorm(FFN) with a per-graph global token.
    Works with batched graphs via batch_idx.
    """
    def __init__(self, dim, attn_dim, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.local = LocalMPNN(dim)
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.attn  = EdgeFlexibleAttention(dim, attn_dim)
        self.drop2 = nn.Dropout(drop)

        self.norm3 = nn.LayerNorm(dim)
        self.ffn   = MLP(dim, hid, dim, drop=drop)

        self.g_token = nn.Parameter(torch.zeros(1, dim))
        nn.init.xavier_uniform_(self.g_token)
        self.Wg_in  = nn.Linear(dim, dim)
        self.Wg_out = nn.Linear(dim, dim)

    def forward(self, X, edge_index, batch_idx):
        B = int(batch_idx.max()) + 1 if batch_idx.numel() > 0 else 1

        # per-graph global context injection
        g = self.g_token.expand(B, -1) + segment_mean(X, batch_idx, B)  # (B, D)
        X = X + self.Wg_in(g)[batch_idx]

        Y = self.local(self.norm1(X), edge_index)
        X = X + self.drop1(Y)

        Y = self.attn(self.norm2(X), edge_index)
        X = X + self.drop2(Y)

        Y = self.ffn(self.norm3(X))
        X = X + Y

        # (optional) update, not used further
        _ = self.Wg_out(g + segment_mean(X, batch_idx, B))
        return X

class GPSFlexHybrid(nn.Module):
    """
    GPS-style + GRIT flexible attention (edge-based), batched.
    Predicts node-level 3D displacement: (sumN, 3)
    """
    def __init__(self, in_dim, model_dim=64, attn_dim=32, depth=4, mlp_ratio=2.0, drop=0.0, out_dim=3):
        super().__init__()
        self.embed = nn.Linear(in_dim, model_dim)
        self.blocks = nn.ModuleList([
            GPSFlexBlock(model_dim, attn_dim, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim//2), nn.ReLU(),
            nn.Linear(model_dim//2, out_dim)
        )

    def forward(self, X, edge_index, batch_idx):
        X = self.embed(X)                         # (sumN, D)
        for blk in self.blocks:
            X = blk(X, edge_index, batch_idx)     # (sumN, D)
        X = self.norm_out(X)
        return self.head(X)                       # (sumN, 3)
