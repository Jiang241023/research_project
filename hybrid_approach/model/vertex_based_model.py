import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Graph MPNN Block using GIN ---
class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index  # edge_index shape: (2, E)
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        out = self.mlp((1 + self.eps) * x + agg)
        return out

# --- 2. Dual-Path MPNN Positional Encoder (Eq 1-4) ---
class DualMPNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mpnna = GINLayer(in_dim, hidden_dim)
        self.mpnna_c = GINLayer(in_dim, hidden_dim)
        self.theta_pe = nn.Parameter(torch.randn(1, hidden_dim * 2, 2))

    def forward(self, x, A_edges, Ac_edges, num_nodes):
        h_A = self.mpnna(x, A_edges, num_nodes)     # Eq (1)
        h_Ac = self.mpnna_c(x, Ac_edges, num_nodes) # Eq (2)
        h = torch.cat([h_A, h_Ac], dim=-1)          # Eq (3)

        pe = torch.tanh(h * self.theta_pe)          # Eq (4)
        pe_sum = pe.sum(dim=-1)
        x_0 = x + pe_sum
        return x_0

# --- 3. Flexible Attention Block (Eq 8-10) ---
class FlexibleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.W_Ev = nn.Linear(dim, dim)
        self.W_A = nn.Linear(dim, 1)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        n = x.size(0)

        Q_exp = Q.unsqueeze(1).expand(-1, n, -1)
        K_exp = K.unsqueeze(0).expand(n, -1, -1)
        edge_feat = torch.relu(Q_exp + K_exp)
        e_ij = self.W_A(edge_feat).squeeze(-1)
        alpha = F.softmax(e_ij, dim=1)

        EV = self.W_Ev(edge_feat)
        out = torch.bmm(alpha.unsqueeze(1), V.unsqueeze(0).expand(n, -1, -1) + EV)
        return out.squeeze(1)

# --- 4. Graph Information Layer (SE-like) ---
class GraphInfoLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, hidden_dim)

    def forward(self, x):
        y_g0 = x.sum(dim=0, keepdim=True)
        y_g1 = F.relu(self.lin1(y_g0))
        y_g2 = torch.sigmoid(self.lin2(y_g1))
        x_g = x * y_g2
        return x_g

# --- 5. Final MLP Head ---
class OutputMLP(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# --- 6. Full Vertex-based Hybrid Model ---
class VertexHybridModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.encoder = DualMPNNEncoder(in_dim, hidden_dim)
        self.gin_A = GINLayer(hidden_dim, hidden_dim)
        self.gin_Ac = GINLayer(hidden_dim, hidden_dim)
        self.attn = FlexibleAttention(hidden_dim)
        self.graph_info = GraphInfoLayer(hidden_dim)
        self.out_head = OutputMLP(hidden_dim, out_dim)

    def forward(self, x, A_edges, Ac_edges, num_nodes):
        # Ensure correct edge format (2, E)
        if A_edges.shape[0] != 2:
            A_edges = A_edges.t()
        if Ac_edges.shape[0] != 2:
            Ac_edges = Ac_edges.t()

        x0 = self.encoder(x, A_edges, Ac_edges, num_nodes)
        hA = self.gin_A(x0, A_edges, num_nodes)
        hAc = self.gin_Ac(x0, Ac_edges, num_nodes)
        hAttn = self.attn(x0)
        x_l = hA + hAc + hAttn
        x_g = self.graph_info(x_l)
        return self.out_head(x_g)