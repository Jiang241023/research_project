import torch
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_loss
from torch_geometric.graphgym.config import cfg

def laplacian_energy(pred, edge_index, norm_mode, node_batch, edge_weight = None, normalized=True):

    # Unpack edge endpoints and basic sizes
    source_index, target_index = edge_index
    num_nodes = pred.size(0)
    num_edges = edge_index.size(1)

    # Prepare edge weights 
    if edge_weight is None:
        edge_weight_vector = pred.new_ones(num_edges)
    else:
        edge_weight_vector = edge_weight

    if normalized:
        # Symmetric normalized Laplacian: energy = ∑(edge_weight_vector || D^(-1/2)f_i - D^(-1/2)f_j ||^2), D^(-1/2) = inv_sqrt_deg and f_i = pred[i]
        deg = pred.new_zeros(num_nodes)
        # scatter_add_ accumulates weights into the degree bins indexed by node ids. (di​=∑k:ik​=i​wk​+∑k:jk​=i​wk)
        deg.scatter_add_(0, source_index, edge_weight_vector)  # treat as undirected degree
        deg.scatter_add_(0, target_index, edge_weight_vector)
        inv_sqrt_deg = deg.clamp_min(1e-12).rsqrt() #inv_sqrt_deg ​= 1 /  sqrt(max(deg_i​, 10e−12))

        # Scale predictions by D^(-1/2): g_i = f_i / sqrt(deg_i)
        degree_scaled_pred = pred * inv_sqrt_deg.unsqueeze(-1) 

        # Per-edge differences on the scaled field
        pairwise_diff = degree_scaled_pred[source_index] - degree_scaled_pred[target_index]

        # Per-edge contributions: w_(ij) * ||g_i - g_j||^2
        edge_contributions = edge_weight_vector * (pairwise_diff * pairwise_diff).sum(dim=-1)
    else:
        pairwise_diff = pred[source_index] - pred[target_index]
        edge_contributions = edge_weight_vector * (pairwise_diff * pairwise_diff).sum(dim=-1)

    energy = edge_contributions.sum()

    # Size normalization
    if node_batch is None:
        if norm_mode == 'per_edge':
            energy = energy / max(num_edges, 1)
        elif norm_mode == 'per_node':
            energy = energy / max(num_nodes, 1)
        return energy

    # Map edges to graphs via their source nodes
    e2g = node_batch[source_index]
    num_graphs = int(node_batch.max().item()) + 1

    # Sum edge contributions per graph: energy_g = sum_{edges in g} c_e
    ernergy_per_graph = pred.new_zeros(num_graphs).index_add(0, e2g, edge_contributions)
    if norm_mode == 'per_edge':
        edges_per_graph = torch.bincount(e2g, minlength=num_graphs).clamp_min(1)
        energy = (ernergy_per_graph / edges_per_graph).mean()
    elif norm_mode == 'per_node':
        nodes_per_graph = torch.bincount(node_batch, minlength=num_graphs).clamp_min(1)
        energy = (ernergy_per_graph / nodes_per_graph).mean()
    else:
        energy = ernergy_per_graph.sum()
    return energy

@register_loss('mse_combined_with_laplacian')
def mse_laplacian_loss(pred, true, batch):

    # Reads hyper-params from YAML
    lam = float(getattr(cfg.model, 'laplace_lambda', 1e-3))
    use_norm = bool(getattr(cfg.model, 'laplace_normalized', True))
    norm_mode = getattr(cfg.model, 'laplace_norm_mode', 'per_edge')
    on_residuals = bool(getattr(cfg.model, 'laplace_on_residuals', False))
    only_mse = bool(getattr(cfg.model, 'only_mse', False))

    # Standard mean-squared error over all nodes and features.
    mse = F.mse_loss(pred, true, reduction='mean')

    if only_mse:
        # print(f"mse:{mse}")
        # print(f"pred:{pred}")
        return mse, pred

    if batch is None or not hasattr(batch, 'edge_index') or batch.edge_index is None:
        print("batch is None")
        return mse, pred
    
    # Residual smoothing discourages high-frequency errors without washing out the target structure.
    if on_residuals:
        signal = pred - true
        # print(f"signal (pred - true):{signal}")
    else:
        signal = pred
        #print(f"signal (pred):{signal}")

    lap = laplacian_energy(signal, batch.edge_index, node_batch=getattr(batch, 'batch', None), normalized=use_norm, norm_mode=norm_mode)
    loss = mse + lam * lap
    # print(f"loss: {loss}")
    # print(f"pred: {pred}")
    return loss, pred