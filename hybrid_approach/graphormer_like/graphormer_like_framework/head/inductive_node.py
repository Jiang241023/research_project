import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head


# @register_head('inductive_node')
# class GNNInductiveNodeHead(nn.Module):
#     """
#     GNN prediction head for inductive node prediction tasks.

#     Args:
#         dim_in (int): Input dimension
#         dim_out (int): Output dimension.
#     """

#     def __init__(self, dim_in, dim_out=3):
#         super(GNNInductiveNodeHead, self).__init__()
#         self.layer_post_mp = MLP(
#             new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
#                              has_act=False, has_bias=True, cfg=cfg))

@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.
    """

    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        # ALWAYS use cfg.gnn.dim_out to avoid mismatches passed from the network
        target_dim_out = getattr(cfg.gnn, 'dim_out', None)
        if target_dim_out is None:
            # last fallback: use arg or default 3
            target_dim_out = 3 if dim_out is None else dim_out

        self.layer_post_mp = MLP(
            new_layer_config(dim_in, target_dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))


    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        #print(f"the shape of pred (from inductive node): {pred.shape}")
        return pred, label
