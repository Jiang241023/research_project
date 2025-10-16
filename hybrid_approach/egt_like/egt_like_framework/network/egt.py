import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config, BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

class FeatureEncoder(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False, has_bias=False, cfg=cfg))
            self.dim_in = cfg.gnn.dim_inner

        if cfg.dataset.edge_encoder:
            cfg.gnn.dim_edge = cfg.gnn.dim_inner  # keep equal
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False, has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

@register_network('EdgeAugmentedGraphTransformer')
class EdgeAugmentedGraphTransformer(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in == cfg.gt.dim_hidden, "Hidden dims must match"

        # Build egt_layer stack
        Layer = register.layer_dict['egt_layer']
        self.layers = nn.ModuleList([
            Layer(in_dim=cfg.gt.dim_hidden,
                  out_dim=cfg.gt.dim_hidden,
                  num_heads=cfg.gt.n_heads,
                  dropout=cfg.gt.dropout,
                  attn_dropout=cfg.gt.attn_dropout,
                  layer_norm=cfg.gt.layer_norm,
                  batch_norm=cfg.gt.batch_norm,
                  residual=True,
                  cfg=cfg.gt)
            for _ in range(cfg.gt.layers)
        ])

        # Head (expects/produces (pred,true))
        GNNHead = register.head_dict[cfg.gnn.head]
        self.head = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)  # GraphGym GNNPreMP works with batch

        for lyr in self.layers:
            batch = lyr(batch)

        # Head should return (pred, true) for your train.py
        return self.head(batch)
