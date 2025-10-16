import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('LinearNode')
class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['node-regression']:
            self.in_dim = 34
            self.encoder = torch.nn.Linear(self.in_dim, emb_dim)
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")       

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch
