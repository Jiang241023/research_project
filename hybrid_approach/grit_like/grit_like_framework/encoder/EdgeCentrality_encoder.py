import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_edge_encoder

@register_edge_encoder('EdgeCentrality')
class EdgeCentralityEncoder(nn.Module):
    """
    Graphormer-style *edge* centrality:
      centrality(e = u->v) = Emb_out[deg_out(u)] + Emb_in[deg_in(v)]
    """
    def __init__(self, emb_dim: int, max_deg: int = 256, use_raw: bool = True, edge_in_dim: int = 31):
        super().__init__()
        self.max_deg = int(max_deg)
        self.use_raw = bool(use_raw)

        # Degree embedding tables, analogous to Graphormer node centrality tables
        self.out_emb = nn.Embedding(self.max_deg, emb_dim)  # for deg_out(u)
        self.in_emb  = nn.Embedding(self.max_deg, emb_dim)  # for deg_in(v)

        # Optional raw edge feature projection
        if self.use_raw:
            self.edge_proj = nn.Linear(edge_in_dim, emb_dim)

    @torch.no_grad()
    def _node_degrees(self, edge_index, num_nodes):
        # Row 0 = sources, Row 1 = destinations (PyG convention)
        src, dst = edge_index
        deg_out = pyg.utils.degree(src, num_nodes=num_nodes, dtype=torch.long)  # out-degree per node
        deg_in  = pyg.utils.degree(dst, num_nodes=num_nodes, dtype=torch.long)  # in-degree  per node
        # Clamp for safe nn.Embedding lookup
        deg_out.clamp_(max=self.max_deg - 1)
        deg_in.clamp_(max=self.max_deg - 1)
        return deg_out, deg_in

    def forward(self, batch):
        device = batch.edge_index.device
        src, dst = batch.edge_index

        #  Node-level degrees (length N), then index per edge
        deg_out_nodes, deg_in_nodes = self._node_degrees(batch.edge_index, batch.num_nodes)
        du = deg_out_nodes[src].to(device)  # (E,)
        dv = deg_in_nodes[dst].to(device)   # (E,)

        #  Edge centrality embedding = sum of two degree embeddings
        centrality = self.out_emb(du) + self.in_emb(dv)  # (E, emb_dim)

        base = self.edge_proj(batch.edge_attr)        # (E, emb_dim)
        batch.edge_attr = base + centrality           # (E, emb_dim)

        return batch