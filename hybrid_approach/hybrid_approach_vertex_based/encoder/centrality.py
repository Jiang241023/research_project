# encoder/centrality.py
import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_node_encoder

@register_node_encoder('GraphormerCentrality')
class GraphormerCentrality(nn.Module):
    """
    Graphormer-style centrality: in-degree + out-degree embeddings summed into x.
    Degrees are computed WITHOUT self-loops by default and clamped to max_deg-1.
    """
    def __init__(self, emb_dim: int, max_deg: int = 256, count_self_loops: bool = False):
        super().__init__()
        self.max_deg = max_deg
        self.count_self_loops = count_self_loops
        self.in_emb  = nn.Embedding(max_deg, emb_dim)
        self.out_emb = nn.Embedding(max_deg, emb_dim)

    def _deg(self, edge_index, num_nodes, which: str):
        col = 1 if which == 'in' else 0
        ei = edge_index
        if not self.count_self_loops:
            ei, _ = pyg.utils.remove_self_loops(ei)
        d = pyg.utils.degree(ei[col], num_nodes=num_nodes, dtype=torch.long)
        return torch.clamp(d, max=self.max_deg - 1)

    def forward(self, batch):
        in_deg  = self._deg(batch.edge_index, batch.num_nodes, 'in').to(batch.x.device)
        out_deg = self._deg(batch.edge_index, batch.num_nodes, 'out').to(batch.x.device)
        batch.x = batch.x + self.in_emb(in_deg) + self.out_emb(out_deg)
        return batch
