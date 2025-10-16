import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_edge_encoder

@register_edge_encoder('EdgeTokenInit')
class EdgeTokenInit(nn.Module):
    """
    Initialize per-edge tokens: batch.edge_x ∈ R^{E×emb_dim}
    Keeps batch.edge_attr for other uses if you want.
    """
    def __init__(self, emb_dim: int, edge_in_dim: int = 31, add_deg_centrality: bool = True):
        super().__init__()
        self.add_deg = add_deg_centrality
        in_dim = edge_in_dim + (2 if self.add_deg else 0)
        self.proj = nn.Linear(in_dim, emb_dim)

    def forward(self, batch):
        assert hasattr(batch, 'edge_attr'), "EdgeTokenInit needs batch.edge_attr"
        ea = batch.edge_attr.float()  # (E, edge_in_dim)
        parts = [ea]
        if self.add_deg:
            src, dst = batch.edge_index
            deg_out = pyg.utils.degree(src, num_nodes=batch.num_nodes, dtype=torch.float)
            deg_in  = pyg.utils.degree(dst, num_nodes=batch.num_nodes, dtype=torch.float)
            parts.append(deg_out[src].unsqueeze(-1))
            parts.append(deg_in [dst].unsqueeze(-1))
        z = torch.cat(parts, dim=-1)          # (E, in_dim)
        batch.edge_x = self.proj(z)           # (E, emb_dim)
        return batch
