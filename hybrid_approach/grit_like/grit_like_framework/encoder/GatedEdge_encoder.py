import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder

@register_edge_encoder('GatedEdge')
class GatedEdgeEncoder(nn.Module):
    """
    Projects edge_attr to emb_dim and modulates it with a gate:
        gate = sigmoid( MLP([x_u || x_v || raw_edge_attr]) )
    Assumes node features batch.x are already projected to cfg.gnn.dim_inner.
    """
    def __init__(self, emb_dim: int, gate_hidden,
                 init_gate_bias = 2):
        super().__init__()
        self.emb_dim = emb_dim
        self.edge_in_dim = 31              # raw edge feature dim (31 my your dataset)
        self.node_dim = int(cfg.gnn.dim_inner)      # node feature dim after node encoder

        # 1) Project raw edge features -> model edge embedding
        self.edge_proj = nn.Linear(self.edge_in_dim, self.emb_dim)

        # 2) Gate MLP over [x_u, x_v, raw_edge_attr]
        gate_in_dim = 2 * self.node_dim + self.edge_in_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1)
        )
        # Start with gate ~ sigmoid(init_gate_bias) ≈ 0.88 if bias=2.0 (mostly open)
        nn.init.constant_(self.gate_mlp[-1].bias, init_gate_bias)

    def forward(self, batch):
        if not hasattr(batch, 'x'):
            raise RuntimeError("GatedEdge requires batch.x (node features).")
        if not hasattr(batch, 'edge_attr'):
            raise RuntimeError("GatedEdge requires batch.edge_attr (raw edge features).")

        # Sanity check raw edge feature dim
        if batch.edge_attr.size(-1) != self.edge_in_dim:
            raise RuntimeError(
                f"GatedEdge expected edge_attr dim {self.edge_in_dim}, "
                f"got {batch.edge_attr.size(-1)}. "
                f"Set edge_in_dim accordingly."
            )

        src, dst = batch.edge_index
        xu, xv = batch.x[src], batch.x[dst]          # (E, node_dim)

        # Base edge embedding
        base = self.edge_proj(batch.edge_attr)       # (E, emb_dim)

        # Gate conditioned on endpoints + raw edge features
        gate_in = torch.cat([xu, xv, batch.edge_attr], dim=-1)  # (E, 2*node_dim + edge_in_dim)
        gate = torch.sigmoid(self.gate_mlp(gate_in))            # (E, 1)

        # Apply gate (broadcast across emb_dim)
        batch.edge_attr = base * gate
        return batch
