import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode
import torch_geometric.graphgym.register as register


class FeatureEncoder(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            self.node_encoder_bn = BatchNorm1dNode(
                new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False, has_bias=False, cfg=cfg)
            ) if cfg.dataset.node_encoder_bn else nn.Identity()
            self.dim_in = cfg.gnn.dim_inner

        if cfg.dataset.edge_encoder:
            cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner) if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            self.edge_encoder_bn = BatchNorm1dNode(
                new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False, has_bias=False, cfg=cfg)
            ) if cfg.dataset.edge_encoder_bn else nn.Identity()

    def forward(self, batch):
        for m in self.children():
            batch = m(batch)
        return batch


@register_network("GraphormerEdgeTransformer")
class GraphormerEdgeTransformer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, "The inner and hidden dims must match."

        TransformerLayer = register.layer_dict.get(cfg.gt.get('layer_type', 'GraphormerEdge'))

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(TransformerLayer(
                in_dim=cfg.gt.dim_hidden,
                out_dim=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                act=cfg.gnn.act,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=True,
                # extra kwargs tolerated by the layer's **kwargs
                norm_e=cfg.gt.attn.get("norm_e", True),
                O_e=cfg.gt.attn.get("O_e", True),
                cfg=cfg.gt,
            ))
        self.layers = nn.Sequential(*layers)

        # IMPORTANT: head must output 3 channels for (dx, dy, dz)
        GNNHead = register.head_dict[cfg.gnn.head]  # 'node' for node-level labels
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for m in self.children():
            batch = m(batch)
        return batch
