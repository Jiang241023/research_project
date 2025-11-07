import numpy as np
import torch.nn.functional as F
from functools import partial
from .rrwp import add_full_rrwp
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.
    Supported PE statistics to precompute, selected by `pe_types`:

    'RRWP': Relative Random Walk Probabilities PE (Ours, for GRIT)
    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ['RRWP']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.

    if 'RRWP' in pe_types:
        param = cfg.posenc_RRWP
        transform = partial(add_full_rrwp,
                            walk_length=param.ksteps,
                            attr_name_abs="rrwp",
                            attr_name_rel="rrwp",
                            add_identity=True,
                            spd=False, # by default False
                            )
        data = transform(data)

    return data


class ComputePosencStat(BaseTransform):
    def __init__(self, pe_types, is_undirected, cfg):
        super().__init__()
        self.pe_types = pe_types
        self.is_undirected = is_undirected
        self.cfg = cfg

    def forward(self, data: Data) -> Data:
        # BaseTransform.__call__ will invoke this
        return compute_posenc_stats(
            data,
            pe_types=self.pe_types,
            is_undirected=self.is_undirected,
            cfg=self.cfg,
        )