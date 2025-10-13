# encoder/pair_bias_builders.py
import torch
from torch import nn
from collections import deque

INF = 1_000_000

def bfs_spd(num_nodes, edge_index):
    # Returns SPD matrix (N,N) with 0 on diag, INF for unreachable.
    N = num_nodes
    spd = torch.full((N, N), INF, dtype=torch.long)
    adj = [[] for _ in range(N)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)
    for s in range(N):
        dist = [INF]*N
        q = deque([s]); dist[s] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == INF:
                    dist[v] = dist[u] + 1
                    q.append(v)
        spd[s] = torch.tensor(dist, dtype=torch.long)
    spd.fill_diagonal_(0)
    return spd

class PairBiasBuilder(nn.Module):
    """
    Builds additive attention bias BxNxN from:
      - SPD buckets (learnable embedding per distance up to max_dist)
      - Optional edge-type path encoding (bag along one shortest path, simple)
    """
    def __init__(self, max_dist: int = 32, num_edge_types: int = 0, edge_emb_dim: int = 32):
        super().__init__()
        self.max_dist = max_dist
        self.spatial_pos_emb = nn.Embedding(max_dist + 2, 1)  # +2 for {self, far}
        self.use_edge_types = num_edge_types > 0
        if self.use_edge_types:
            self.edge_type_emb = nn.Embedding(num_edge_types, 1)
            self.edge_proj = nn.Linear(1, 1)

    def forward(self, batch_list, pad_to: int, device=None):
        """
        batch_list: list[Data] with fields:
          - num_nodes, edge_index, (optional) spatial_pos (N,N long),
          - (optional) edge_type (E, long, in [0, num_edge_types))
        Returns: attn_bias [B, pad_to, pad_to], attn_mask [B, pad_to] (1=real, 0=pad)
        """
        B = len(batch_list)
        Nmax = pad_to
        attn_bias = torch.zeros(B, Nmax, Nmax, dtype=torch.float32, device=device)
        attn_mask = torch.zeros(B, Nmax, dtype=torch.bool, device=device)

        for b, data in enumerate(batch_list):
            N = int(data.num_nodes)
            attn_mask[b, :N] = True

            # SPD: use provided, else compute by BFS
            if hasattr(data, 'spatial_pos'):
                spd = data.spatial_pos.clone()
            else:
                spd = bfs_spd(N, data.edge_index.to('cpu')).to(torch.long)
            spd = spd.clamp_max(self.max_dist + 1)  # far bucket == max_dist+1
            spd[:, :].masked_fill_(spd < 0, self.max_dist + 1)
            # offsets: 0(self), 1..max_dist, far→max_dist+1
            bias_spd = self.spatial_pos_emb(spd)  # (N,N,1)
            bias_spd = bias_spd.squeeze(-1)

            bias = bias_spd  # start with SPD bias

            # Optional: edge type bag for 1-hop neighbors only (cheap & effective)
            if self.use_edge_types and hasattr(data, 'edge_type'):
                src, dst = data.edge_index
                et = data.edge_type.to(dtype=torch.long, device=bias.device)
                et = et.clamp_min(0)
                # accumulate per (u,v)
                bias_edge = torch.zeros_like(bias)
                bias_edge[src, dst] += self.edge_type_emb(et).squeeze(-1)
                bias += self.edge_proj(bias_edge.unsqueeze(-1)).squeeze(-1)

            attn_bias[b, :N, :N] = bias

        # Mask padding rows/cols to large negative so softmax ignores them
        pad_mask = ~attn_mask
        attn_bias.masked_fill_(pad_mask[:, None, :], float('-inf'))
        attn_bias.masked_fill_(pad_mask[:, :, None], float('-inf'))
        # Keep diag finite
        for b in range(B):
            if attn_mask[b].any():
                n = int(attn_mask[b].sum())
                attn_bias[b, :n, :n].fill_diagonal_(0.0)
        return attn_bias, attn_mask
