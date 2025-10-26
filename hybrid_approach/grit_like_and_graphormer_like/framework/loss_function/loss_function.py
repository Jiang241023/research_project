import torch
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_loss
from torch_geometric.graphgym.config import cfg

def laplacian_energy(pred, edge_index, normalized, norm_mode, edge_weight, node_batch):
    row, col = edge_index
    N = pred.size(0)
    E = edge_index.size(1)

    if edge_weight is None:
        w = pred.new_ones(E)
    else:
        w = edge_weight

    if normalized:
        # exact L_sym: use g = D^{-1/2} f
        deg = pred.new_zeros(N)
        deg.scatter_add_(0, row, w)  # treat as undirected degree
        deg.scatter_add_(0, col, w)
        inv_sqrt_deg = deg.clamp_min(1e-12).rsqrt()
        g = pred * inv_sqrt_deg.unsqueeze(-1)
        diff = g[row] - g[col]
        e_contrib = w * (diff * diff).sum(-1)
    else:
        diff = pred[row] - pred[col]
        e_contrib = w * (diff * diff).sum(-1)

    energy = e_contrib.sum()

    # Size normalization
    if node_batch is None:
        if norm_mode == 'per_edge':
            energy = energy / max(E, 1)
        elif norm_mode == 'per_node':
            energy = energy / max(N, 1)
        return energy

    # Optional: per-graph normalization if you have multi-graph batches
    # Map edges to graphs via their source nodes
    e2g = node_batch[row]
    num_graphs = int(node_batch.max().item()) + 1
    # Sum per graph
    per_g = pred.new_zeros(num_graphs).index_add(0, e2g, e_contrib)
    if norm_mode == 'per_edge':
        edges_per_g = torch.bincount(e2g, minlength=num_graphs).clamp_min(1)
        energy = (per_g / edges_per_g).mean()
    elif norm_mode == 'per_node':
        nodes_per_g = torch.bincount(node_batch, minlength=num_graphs).clamp_min(1)
        energy = (per_g / nodes_per_g).mean()
    else:
        energy = per_g.sum()
    return energy


@register_loss('mse_laplacian')
def mse_laplacian_loss(pred, true, batch=None):
    lam = float(getattr(cfg.model, 'laplace_lambda', 1e-3))
    use_norm = bool(getattr(cfg.model, 'laplace_normalized', True))
    norm_mode = getattr(cfg.model, 'laplace_norm_mode', 'per_edge')
    on_resid = bool(getattr(cfg.model, 'laplace_on_residuals', False))

    mse = F.mse_loss(pred, true, reduction='mean')

    if batch is None or not hasattr(batch, 'edge_index') or batch.edge_index is None:
        total = mse
        return total, pred  # <-- return tuple

    signal = pred - true if on_resid else pred
    lap = laplacian_energy(signal, batch.edge_index, normalized=use_norm, norm_mode=norm_mode)
    total = mse + lam * lap
    return total, pred