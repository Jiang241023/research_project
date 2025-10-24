import torch
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_loss
from torch_geometric.graphgym.config import cfg

def _laplacian_energy(pred, edge_index, *, normalized: bool, norm_mode: str):
    """
    Dirichlet energy on a graph: sum_{(i,j) in E} ||pred_i - pred_j||^2
    pred: [N, D], edge_index: [2, E] (can be directed or unique undirected)
    normalized: use 1/sqrt(d_i d_j) scaling
    norm_mode: 'per_edge' | 'per_node' | 'none'  (stabilizes scale across graph sizes)
    """
    row, col = edge_index
    N = pred.size(0)
    E = edge_index.size(1)

    # Optional normalized Laplacian weighting
    if normalized:
        deg = torch.zeros(N, device=pred.device, dtype=pred.dtype)
        # treat edges as undirected for degree counting
        deg.scatter_add_(0, row, torch.ones(E, device=pred.device, dtype=pred.dtype))
        deg.scatter_add_(0, col, torch.ones(E, device=pred.device, dtype=pred.dtype))
        scale = (deg[row].clamp_min(1e-12) * deg[col].clamp_min(1e-12)).rsqrt()
    else:
        scale = 1.0

    diff = pred[row] - pred[col]             # [E, D]
    e_contrib = (diff * diff).sum(-1)        # [E]
    energy = (e_contrib * scale).sum()       # scalar

    if norm_mode == 'per_edge':
        energy = energy / max(E, 1)
    elif norm_mode == 'per_node':
        energy = energy / max(N, 1)
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
    lap = _laplacian_energy(signal, batch.edge_index, normalized=use_norm, norm_mode=norm_mode)
    total = mse + lam * lap
    return total, pred