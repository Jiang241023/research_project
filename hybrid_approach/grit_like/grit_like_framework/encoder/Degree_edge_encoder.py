import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_edge_encoder

@register_edge_encoder('DegreeEdge')
class DegreeAwareEdgeEncoder(nn.Module):
    """
    Concatenate raw edge_attr (31-D) with degree stats of endpoints,
    then project to emb_dim. No Lazy modules -> no warm-up needed.
    """
    def __init__(self, emb_dim: int, use_products: bool = True, concat_raw: bool = True):
        super().__init__()
        self.use_products = use_products
        self.concat_raw = concat_raw

        deg_feat_dim = 4 if use_products else 3
        edge_attr_dim = 31 if concat_raw else 0
        in_dim = edge_attr_dim + deg_feat_dim

        self.proj = nn.Linear(in_dim, emb_dim)

    def forward(self, batch):
        src, dst = batch.edge_index
        deg_in  = pyg.utils.degree(dst, num_nodes=batch.num_nodes, dtype=torch.float)
        deg_out = pyg.utils.degree(src, num_nodes=batch.num_nodes, dtype=torch.float)
        du = deg_out[src]; dv = deg_in[dst]
        parts = [du, dv, du + dv]
        if self.use_products:
            parts.append(du * dv)
        deg_feat = torch.stack(parts, dim=-1)  # (E, 3/4)

        if self.concat_raw:
            if not hasattr(batch, 'edge_attr'):
                raise RuntimeError("DegreeEdge expects batch.edge_attr (E,31) when concat_raw=True.")
            feat = torch.cat([batch.edge_attr, deg_feat], dim=-1)
        else:
            feat = deg_feat

        batch.edge_attr = self.proj(feat)
        return batch
