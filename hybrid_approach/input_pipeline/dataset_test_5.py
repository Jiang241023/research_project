import numpy as np
import h5py
import os, sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from DDACSDataset import DDACSDataset
from utils.utils_DDACS import (
    extract_mesh,
    extract_element_thickness,
    extract_point_springback,
)
from model.vertex_based_model_3 import VertexFlexibleTransformer  # alias to VertexFlexibleTransformer


# ------------------------------
# Repro & device
# ------------------------------
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Feature builders
# ------------------------------
def load_element_features(h5_path, component="blank", timestep=3, op_form=10):
    """Return per-element features: strain(12) + thickness(1) → (m, 13)."""
    with h5py.File(h5_path, "r") as f:
        comp = f[f"OP{op_form}"][component]
        strain_t = comp["element_shell_strain"][timestep]  # (m,2,6)
    strain_features = strain_t.reshape(strain_t.shape[0], -1).astype(np.float32)  # (m,12)
    thickness = extract_element_thickness(h5_path, timestep=timestep, operation=op_form).astype(np.float32)  # (m,)
    return np.concatenate([strain_features, thickness[:, None]], axis=1)  # (m,13)


def triangles_to_edges(tris, num_nodes):
    """
    Build undirected edge_index from triangles.
    tris: (T,3) int64
    returns edge_index: (2,E) torch.long and degree (n,) float32 normalized
    """
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    e_undirected = np.vstack([
        np.stack([a, b],  axis=0), np.stack([b, a],  axis=0),
        np.stack([b, c],  axis=0), np.stack([c, b],  axis=0),
        np.stack([c, a],  axis=0), np.stack([a, c],  axis=0),
    ])  # (6T, 2)

    # Remove duplicates
    e_unique = np.unique(e_undirected.T, axis=0).T  # (2, E)

    # Add self-loops for stability
    self_loops = np.arange(num_nodes, dtype=np.int64)
    self_loops = np.stack([self_loops, self_loops], axis=0)
    edge_index = np.concatenate([e_unique, self_loops], axis=1)

    # Degree from edges (undirected)
    deg = np.bincount(edge_index[0], minlength=num_nodes).astype(np.float32)

    # Normalize degree
    deg_norm = (deg - deg.mean()) / (deg.std() + 1e-8)

    return torch.from_numpy(edge_index.astype(np.int64)), torch.from_numpy(deg_norm)


def element_to_node_features(num_nodes, triangles, elem_features):
    """
    Average element features to incident vertices.
    triangles: (T,3)
    elem_features: (T or 2*T, F)
    returns node_features: (n,F)
    """
    F = elem_features.shape[1]
    node_feature_sums = np.zeros((num_nodes, F), dtype=np.float32)
    node_counts = np.zeros(num_nodes, dtype=np.int32)

    for t in range(triangles.shape[0]):
        tri = triangles[t]  # (3,)
        feat = elem_features[t]  # (F,)
        for v in tri:
            node_feature_sums[v] += feat
            node_counts[v] += 1

    node_counts = np.maximum(node_counts, 1)
    return node_feature_sums / node_counts[:, None]


def load_displacement_op10(h5_path):
    _, disp = extract_point_springback(h5_path, operation=10)  # (n,3)
    return disp.astype(np.float32)


def build_vertex_features(h5_path, component="blank", op_form=10, timestep=3):
    """
    Returns:
      X: (n, 31+4)  → base 31 (= 13*? after duplication + projection) + [deg, x,y,z]
      Y: (n, 3)
      edge_index: (2,E)
    """
    # Per-element features (m,13)
    elem_feats = load_element_features(h5_path, component, timestep, op_form)

    # Mesh
    node_coords, triangles = extract_mesh(h5_path, operation=op_form, component=component)
    node_coords = node_coords.astype(np.float32)                 # (n,3)
    triangles = triangles.astype(np.int64)                       # (T,3)
    n = node_coords.shape[0]

    # Map elements(OP10 quad→tri) to triangles: duplicate along axis 0
    # (Assumes 2 triangles per quad, consistent with your earlier setup)
    T = triangles.shape[0]
    if elem_feats.shape[0] * 2 == T:
        elem_feats_tri = np.repeat(elem_feats, 2, axis=0)
    elif elem_feats.shape[0] == T:
        elem_feats_tri = elem_feats
    else:
        # Fallback: pad/trim to T
        reps = int(np.ceil(T / elem_feats.shape[0]))
        elem_feats_tri = np.repeat(elem_feats, reps, axis=0)[:T]

    # Project element features to nodes (average per incident node)
    node_base = element_to_node_features(n, triangles, elem_feats_tri)  # (n,13)

    # To hit the 31 "base" channels you were using previously, expand with simple
    # polynomial features (safe & fast) – or keep 13 directly if you prefer.
    # Here we build [x, x^2, log1p(|x|)] per channel then truncate to 31.
    x = node_base
    poly = np.concatenate([x, x**2, np.log1p(np.abs(x))], axis=1)  # (n, 39)
    base_31 = poly[:, :31].astype(np.float32)                      # (n,31)

    # Edges and degree
    edge_index, deg_norm = triangles_to_edges(triangles, n)        # (2,E), (n,)

    # Final vertex features = base_31 + structural [deg_norm, coords(3)]
    S = np.concatenate([deg_norm[:, None].numpy(), node_coords], axis=1).astype(np.float32)  # (n,4)
    X = np.concatenate([base_31, S], axis=1).astype(np.float32)    # (n, 35)

    # Targets
    Y = load_displacement_op10(h5_path)                            # (n,3)
    return torch.from_numpy(X), torch.from_numpy(Y), edge_index


# ------------------------------
# Dataset wrapper
# ------------------------------
class FormingDisplacementDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        _, _, h5_path = self.base_dataset[idx]
        X, Y, edge_index = build_vertex_features(h5_path)
        return X, Y, edge_index


# ------------------------------
# Training / Evaluation
# ------------------------------
def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    running = 0.0
    for X, Y, edge_index in tqdm(loader, desc="Train", leave=False):
        # batch_size=1 → squeeze
        X = X[0].to(device)
        Y = Y[0].to(device)
        edge_index = edge_index[0].to(device)

        opt.zero_grad(set_to_none=True)
        Y_hat = model(X, edge_index)       # (n,3)
        loss = loss_fn(Y_hat, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item()
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, loss_fn, tag="Eval"):
    model.eval()
    running = 0.0
    for X, Y, edge_index in tqdm(loader, desc=tag, leave=False):
        X = X[0].to(device)
        Y = Y[0].to(device)
        edge_index = edge_index[0].to(device)
        Y_hat = model(X, edge_index)
        loss = loss_fn(Y_hat, Y)
        running += loss.item()
    return running / max(1, len(loader))



# Data
data_dir = Path("/mnt/data/darus/")
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

full = FormingDisplacementDataset(dataset)

n = len(full)
n_train = int(0.7 * n)
n_test  = int(0.2 * n)
n_eval  = n - n_train - n_test
train_set, test_set, eval_set = random_split(full, [n_train, n_test, n_eval])
print(len(train_set), len(test_set), len(eval_set))

# IMPORTANT: graphs are big (≈11k verts), use batch_size=1
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, num_workers=0)
eval_loader  = DataLoader(eval_set,  batch_size=1, shuffle=False, num_workers=0)

# Model
model = VertexFlexibleTransformer(  # alias → VertexFlexibleTransformer
    in_dim_base=13,
    struct_dim=4,
    hidden_dim=128,
    num_layers=3,
    heads=4,
    out_dim=3,
    dropout=0.1,
).to(device)

# Optim & loss
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# Train
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    tr = train_one_epoch(model, train_loader, opt, loss_fn)
    te = evaluate(model, test_loader, loss_fn, tag="Test")
    print(f"Epoch {epoch:02d} | train {tr:.5f} | test {te:.5f}")

# Final eval
ev = evaluate(model, eval_loader, loss_fn, tag="Eval")
print(f"Final eval MSE: {ev:.5f}")

