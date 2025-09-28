import torch
import torch.nn as nn

class TopoPositionalEmbedding(nn.Module):
    """
    TIGT-style topological embedding:
      X_h = Linear(X)                 # (B,N,H)
      h_A = GIN(X_h, A)               # (B,N,H)
      h_AC = GIN(X_h, Ac)             # (B,N,H)
      X0  = X_h + sum_{p in {A,AC}} tanh(theta_p ∘ h_p)
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.mpn = GINConv(hidden_dim, hidden_dim)  # shared GIN for A and Ac
        self.theta = nn.Parameter(torch.empty(hidden_dim, 2))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, X, A, Ac):
        # X: (B,N,Fin), A,Ac: (B,N,N)
        X_h = self.embed(X)                     # (B,N,H)
        h_A  = self.mpn(X_h, A)                 # (B,N,H)
        h_AC = self.mpn(X_h, Ac)                # (B,N,H)
        h_stack = torch.stack([h_A, h_AC], dim=-1)    # (B,N,H,2)
        h_weighted = torch.tanh(h_stack * self.theta) # (B,N,H,2) ∘ (H,2) → broadcast
        X0 = X_h + h_weighted.sum(dim=-1)            # (B,N,H)
        return X0


class GINConv(nn.Module):
    """
    GIN update without external libs:
      Z = (1+eps) * X + A @ X
      Y = Linear(Z)
      return ReLU(Y)
    """
    def __init__(self, in_dim, out_dim, eps=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor([eps], dtype=torch.float32))
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, X, A):
        B, N, C = X.shape
        neigh_sum = torch.bmm(A, X)             # (B,N,C)
        out = (1.0 + self.eps) * X + neigh_sum  # (B,N,C)
        out = self.linear(out)                   # (B,N,out_dim)
        return torch.relu(out)


class DualMPNNLayer(nn.Module):
    """
    Two parallel GINs: one on adjacency A, one on clique Ac.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.mpn_a  = GINConv(hidden_dim, hidden_dim)
        self.mpn_ac = GINConv(hidden_dim, hidden_dim)

    def forward(self, X_prev, A, Ac):
        X_A  = self.mpn_a(X_prev, A)   # (B,N,H)
        X_AC = self.mpn_ac(X_prev, Ac) # (B,N,H)
        return X_A, X_AC


class FlexibleAttention(nn.Module):
    """
    GRIT-style flexible attention (global, O(N^2) per graph in a batch):
      e_ij = tanh(Wo x_i + W1 x_j)
      score_ij = w_phi^T e_ij; alpha_i = softmax_j(score_ij)
      val_ij = Wy x_j + We e_ij
      x'_i = sum_j alpha_ij * val_ij
    """
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.Wo   = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W1   = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.Wphi = nn.Linear(attn_dim, 1, bias=False)
        self.Wy   = nn.Linear(hidden_dim, hidden_dim)
        self.We   = nn.Linear(attn_dim, hidden_dim)

    def forward(self, X):
        B, N, C = X.shape
        h_o = self.Wo(X)  # (B,N,d)
        h_1 = self.W1(X)  # (B,N,d)

        e_ij = torch.tanh(h_o.unsqueeze(2) + h_1.unsqueeze(1))  # (B,N,N,d)
        scores = self.Wphi(e_ij).squeeze(-1)                    # (B,N,N)
        alpha  = torch.softmax(scores, dim=-1)                  # (B,N,N)

        X_y  = self.Wy(X)                 # (B,N,C)
        eval = self.We(e_ij)              # (B,N,N,C)
        part1 = torch.bmm(alpha, X_y)     # (B,N,C)
        part2 = (alpha.unsqueeze(-1) * eval).sum(dim=2)  # (B,N,C)
        return part1 + part2              # (B,N,C)


class GraphInfoSqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation over channels with graph-level squeeze:
      y0 = sum_nodes X
      y1 = ReLU(Linear1(y0))
      y2 = Sigmoid(Linear2(y1))
      X_out = MLP( X * y2[:,None,:] )
    """
    def __init__(self, hidden_dim, reduction=4):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // reduction)
        self.lin2 = nn.Linear(hidden_dim // reduction, hidden_dim)
        self.mlp  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X_bar):
        B, N, C = X_bar.shape
        y0 = X_bar.sum(dim=1)                 # (B,C)
        y1 = torch.relu(self.lin1(y0))        # (B,C//r)
        y2 = torch.sigmoid(self.lin2(y1))     # (B,C)
        X_scaled = X_bar * y2.unsqueeze(1)    # (B,N,C)
        return self.mlp(X_scaled)             # (B,N,C)


class VertexHybridModel(nn.Module):
    """
    TIGT backbone + GRIT-style flexible attention + SE graph info,
    with a **per-node** regression head → (B, N, 3).
    """
    def __init__(self, in_dim, hidden_dim, attn_dim, L=3, reduction=4, out_dim=3):
        super().__init__()
        self.topo = TopoPositionalEmbedding(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'dual': DualMPNNLayer(hidden_dim),
                'attn': FlexibleAttention(hidden_dim, attn_dim),
                'info': GraphInfoSqueezeExcite(hidden_dim, reduction)
            })
            for _ in range(L)
        ])
        # per-node regression head
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)   # 3 displacement components
        )

    def forward(self, X, A, Ac):
        # X: (B,N,Fin), A,Ac: (B,N,N)
        X_prev = self.topo(X, A, Ac)   # (B,N,H)
        for layer in self.layers:
            X_A, X_AC = layer['dual'](X_prev, A, Ac)  # (B,N,H)
            X_att     = layer['attn'](X_prev)         # (B,N,H)
            X_bar     = X_A + X_AC + X_att            # (B,N,H)
            X_prev    = layer['info'](X_bar)          # (B,N,H)
        return self.node_head(X_prev)                  # (B,N,3)
