import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


# It adds an embedding of each node’s in-degree and out-degree to the (projected) raw features
@register_node_encoder('DegreeCentrality')
class GraphormerCentralityEncoder(nn.Module):
    """
    Graphormer-style centrality = in-degree emb + out-degree emb, added to x.
    - Projects x from cfg.share.dim_in -> emb_dim (so shapes match).
    - Computes degrees ; clamps to [0, max_deg-1].
    - Uses batch.in_deg/out_deg if present, else computes on the fly.
    """
    def __init__(self, emb_dim, max_deg = 256):
        super().__init__()
        self.max_deg = int(max_deg)

        # 1) Project raw x -> emb_dim (e.g., 34 -> 64)
        in_dim = 34  
        self.proj = nn.Linear(in_dim, emb_dim)

        # 2) Degree embeddings (in/out)
        self.in_emb  = nn.Embedding(self.max_deg, emb_dim)
        self.out_emb = nn.Embedding(self.max_deg, emb_dim)

    def _degree(self, edge_index, num_nodes, which):
        # For in-degree, use the destination row (edge_index[1]). For out-degree, use the source row (edge_index[0]). row 0 = source (where an edge starts). row 1 destination  
        if which == 'in':
            col = 1
        else:
            col = 0
        ei = edge_index
        # print(f"the shape ei:{ei.shape}")
        # for i in range(2):
        #     print(f"ei[{i}]: {ei[i]}")
        # Counts how many times each node index appears in that row
        deg = pyg.utils.degree(ei[col], num_nodes=num_nodes, dtype=torch.long)

        return deg.clamp_(max=self.max_deg - 1)

    def forward(self, batch):
        device = batch.x.device

        # project x to emb_dim first so addition is valid
        x = self.proj(batch.x)

        if hasattr(batch, 'in_deg') and hasattr(batch, 'out_deg'):
            in_deg  = batch.in_deg.to(torch.long).clamp_(max=self.max_deg - 1).to(device)
            out_deg = batch.out_deg.to(torch.long).clamp_(max=self.max_deg - 1).to(device)
        else:
            in_deg  = self._degree(batch.edge_index, batch.num_nodes, 'in').to(device)
            out_deg = self._degree(batch.edge_index, batch.num_nodes, 'out').to(device)

        x = x + self.in_emb(in_deg) + self.out_emb(out_deg)
        batch.x = x
        #print(f"the shape of batch:{batch}")
        return batch
