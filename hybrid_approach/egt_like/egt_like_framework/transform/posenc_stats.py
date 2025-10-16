import math
import torch
import torch.nn.functional as F
from torch_geometric.utils import (to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ['RWSE', 'RFF', 'PPRAnchors']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.

    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing

    if 'RFF' in pe_types:
        out_dim = getattr(cfg.posenc_RFF, 'out_dim', 32)
        sigma   = getattr(cfg.posenc_RFF, 'sigma', 50.0)
        data = add_rff_pe(data, out_dim=out_dim, sigma=sigma, attr_name='pestat_RFF')

    if 'PPRAnchors' in pe_types:
            p = cfg.posenc_PPRAnchors
            data = add_ppr_anchors_pe(data, k=p.k, alpha=p.alpha, iters=p.iters, coord_attr=getattr(p, 'coord_attr', 'pos')) 
    return data


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing

def add_rff_pe(data, out_dim=32, sigma=50.0, attr_name='pestat_RFF'):
    """
    Node-wise RFF of coordinates: pe_i = cos(x_i W + b), W~N(0,1/sigma^2)
    Writes data.pestat_RFF: (N, out_dim)
    """
    X = getattr(data, 'pos', None)
    if X is None:
        X = getattr(data, 'node_coords', None)
    if X is None:
        return data
    X = X.float()
    device = X.device
    D_in = X.size(1)
    W = torch.randn(D_in, out_dim, device=device) / sigma
    b = 2*math.pi*torch.rand(out_dim, device=device)
    Z = X @ W + b
    pe = torch.cos(Z) * math.sqrt(2.0 / out_dim)
    setattr(data, attr_name, pe)
    return data

# --- smooth diffusion from K anchors via APPNP power iterations (sparse @ edge_index) ---
def add_ppr_anchors_pe(data, k=16, alpha=0.15, iters=10, coord_attr='pos', attr_name='pestat_PPR'):
    import torch
    from torch_geometric.utils import degree
    N = data.num_nodes; device = data.edge_index.device
    src, dst = data.edge_index
    deg_out = degree(src, num_nodes=N).clamp_min(1.0)  # out-degree per node

    # choose anchors (FPS or random)
    coords = getattr(data, coord_attr, None) or getattr(data, 'node_coords', None)
    if coords is not None:
        xyz = coords.float()
        sel = [int(torch.randint(0, N, (1,)).item())]
        d2  = torch.full((N,), float('inf'), device=xyz.device)
        for _ in range(1, k):
            d2 = torch.minimum(d2, ((xyz - xyz[sel[-1]])**2).sum(-1))
            sel.append(int(torch.argmax(d2).item()))
        anchors = torch.tensor(sel, device=device)
    else:
        anchors = torch.randperm(N, device=device)[:k]

    pe = torch.zeros(N, k, device=device)
    # Pv = average of neighbor values: Pv[u] = (1/deg_out[u]) * sum_{u->j} v[j]
    for j, a in enumerate(anchors.tolist()):
        x = torch.zeros(N, device=device); x[a] = 1.0
        v = x.clone()
        for _ in range(iters):
            # sum over neighbors j for each u: sum v[j] for edges (u->j)
            msg = torch.zeros(N, device=device).index_add_(0, src, v[dst])
            v = (1 - alpha) * (msg / deg_out) + alpha * x
        pe[:, j] = v
    setattr(data, attr_name, pe)
    return data


class ComputePosencStat(BaseTransform):
    def __init__(self, pe_types, is_undirected, cfg):
        self.pe_types = pe_types
        self.is_undirected = is_undirected
        self.cfg = cfg

    def __call__(self, data: Data) -> Data:
        data = compute_posenc_stats(data, pe_types=self.pe_types,
                                    is_undirected=self.is_undirected,
                                    cfg=self.cfg
                                    )
        return data