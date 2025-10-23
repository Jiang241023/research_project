import os
import numpy as np
import torch

# Treat every original edge as a token/node, and connect two edge-tokens if their original edges share a node
def build_edge_edge_index(edge_index, num_edges, num_nodes):
    """Line-graph connectivity (edge→edge) if two edges share a node."""
    device = edge_index.device
    E = num_edges
    src, dst = edge_index

    nodes = torch.cat([src, dst], dim=0)
    edge_ids = torch.cat([torch.arange(E, device=device),
                          torch.arange(E, device=device)], 0)

    order = torch.argsort(nodes)
    nodes_sorted = nodes[order]
    edge_ids_sorted = edge_ids[order]
    _, counts = torch.unique_consecutive(nodes_sorted, return_counts=True)

    e_row, e_col = [], []
    start = 0
    for c in counts.tolist():
        if c > 1:
            group = edge_ids_sorted[start:start + c]
            g1 = group.repeat_interleave(c)
            g2 = group.repeat(c)
            mask = g1 != g2
            e_row.append(g1[mask])
            e_col.append(g2[mask])
        start += c

    if not e_row:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    e_row = torch.cat(e_row, dim=0)
    e_col = torch.cat(e_col, dim=0)

    key  = e_row * E + e_col
    uniq = torch.unique(key)
    e_row = (uniq // E).long()
    e_col = (uniq %  E).long()
    return torch.stack([e_row, e_col], dim=0)

# --- helpers to compute + save in .npy ---
def normalize_edge_index_shape(edge_index) -> torch.Tensor:
    """
    Ensure edge_index is (2, E). Accepts (2, E) or (E, 2).
    """
    if edge_index.dim() != 2:
        raise ValueError(f"edge_index must be 2D, got shape {tuple(edge_index.shape)}")
    if edge_index.shape[0] == 2:        # already (2, E)
        return edge_index
    if edge_index.shape[1] == 2:        # (E, 2) -> transpose
        return edge_index.t().contiguous()
    raise ValueError(f"edge_index must be (2,E) or (E,2), got {tuple(edge_index.shape)}")

def compute_edge_edge_index(edge_index,
                            num_nodes) -> torch.Tensor:
    """
    Compute line-graph edges with your build_edge_edge_index() and return (2, M) LongTensor.
    """
    ei = normalize_edge_index_shape(edge_index)
    E = int(ei.shape[1])
    if num_nodes is None:
        num_nodes = int(ei.max().item()) + 1 if E > 0 else 0
    return build_edge_edge_index(ei, num_edges=E, num_nodes=num_nodes)

def save_edge_edge_index_npy(e_edge_index,
                             out_path,
                             as_rows_M2) -> None:
    """
    Save e_edge_index to .npy. If as_rows_M2=True, saves as (M, 2) (your usual convention).
    Otherwise saves as (2, M).
    """
    e_edge_index = e_edge_index.detach().cpu()
    arr = e_edge_index.t().numpy().astype(np.int64) if as_rows_M2 else e_edge_index.numpy().astype(np.int64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, arr)

def compute_and_save_edge_edge_index_npy(edge_index,
                                         out_path,
                                         num_nodes,
                                         as_rows_M2):
    """
    One-shot: compute e_edge_index and save as .npy. Returns the saved numpy array.
    """
    e_edge_index = compute_edge_edge_index(edge_index, num_nodes=num_nodes)
    e2 = e_edge_index.t().contiguous().detach().cpu().numpy().astype(np.int64) if as_rows_M2 \
         else e_edge_index.detach().cpu().numpy().astype(np.int64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, e2)
    return e2

saved = compute_and_save_edge_edge_index_npy(
    edge_index=edge_index,
    out_path="/path/to/123_edge_edge_index.npy",  # <- file name; pick your ID/pattern
    as_rows_M2=True
)