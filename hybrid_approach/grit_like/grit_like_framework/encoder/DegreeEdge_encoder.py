import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_edge_encoder

@register_edge_encoder('DegreeEdge')
class DegreeAwareEdgeEncoder(nn.Module):
    """
    Concatenate raw edge_attr (31-D) with degree stats of endpoints,
    then project to emb_dim. 
    """
    def __init__(self, emb_dim):
        super().__init__()

        in_dim = 3

        self.proj = nn.Linear(in_dim, emb_dim) # (3, 64)

    def forward(self, batch):
        source_node_indices, destination_node_indices = batch.edge_index
        deg_in  = pyg.utils.degree(destination_node_indices, num_nodes=batch.num_nodes, dtype=torch.float)
        deg_out = pyg.utils.degree(source_node_indices, num_nodes=batch.num_nodes, dtype=torch.float)
        du = deg_out[source_node_indices]
        dv = deg_in[destination_node_indices]
        parts = [du, dv, du + dv]
        deg_feat = torch.stack(parts, dim=-1)  # (E, 3)
        #print(f"the shape of deg_feat: {deg_feat.shape}")

        batch.edge_attr = self.proj(deg_feat)
        return batch
