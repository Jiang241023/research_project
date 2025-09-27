import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from DDACSDataset import DDACSDataset
from utils.utils_DDACS import extract_point_cloud, extract_mesh, display_structure, extract_element_thickness, extract_point_springback
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from model.vertex_based_model import GPSFlexHybrid


# set random seed
torch.manual_seed(0)

# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

# Get concatenated_strainstressthickness_features
def load_element_features(h5_path, component="blank", timestep=3, op_form=10):
    with h5py.File(h5_path, "r") as f:
        comp = f[f"OP{op_form}"][component]
        #stress_t = comp["element_shell_stress"][timestep]  # (m,3,6)
        strain_t = comp["element_shell_strain"][timestep]  # (m,2,6)

    # concatenated_strainstress_features = np.concatenate([stress_t.reshape(stress_t.shape[0], -1).astype(np.float32),  # 18
    #                                                     strain_t.reshape(strain_t.shape[0], -1).astype(np.float32),  # 12
    #                                                     ], axis=1)  # (m,30)
    strain_features = strain_t.reshape(strain_t.shape[0], -1).astype(np.float32)  # (m, 12)

    thickness = extract_element_thickness(h5_path, timestep=timestep, operation=op_form).astype(np.float32)  
    concatenated_features = np.concatenate([strain_features, thickness[:, None]], axis=1)  
    return concatenated_features  # (m,31)

# Get quad mesh
def load_quad_mesh(h5_path, component="blank", op_form=10):
    node_coords, triangles = extract_mesh(h5_path, operation=op_form, component=component)
    #print(f"the shape of triangles:\n {triangles.shape}") # (22050, 3)
    #print(f"first ten elements of triangles:\n {triangles[:10]}")
    #print(f"the dtype of triangles:\n {triangles.dtype}")
    return node_coords.astype(np.float32), triangles.astype(np.int64)

# Project element features to node features
def element_to_node_features(num_nodes, triangles, elem_features):
    node_feature_sums = np.zeros((num_nodes, elem_features.shape[1]), dtype=np.float32)
    #print(f"the shape of node_feature_sums:\n {node_feature_sums.shape}") #(11236, 31)
    node_counts = np.zeros(num_nodes, dtype=np.int32)
    #print(f"the shape of node_counts:\n {node_counts.shape}")  #(11236,)

    # Loop over each element
    for triangle_index in range(len(triangles)):
        triangle_feature = elem_features[triangle_index]    # (31,)
        triangle_nodes = triangles[triangle_index]           #  (3,)
        #print(f"the shape of triangle_nodes:\n {triangle_nodes.shape}")

        # Give this triangle's feature to each of its 3 nodes
        for node_index in triangle_nodes:
            # Accumulate the feature into the node's total
            node_feature_sums[node_index] += triangle_feature 
            #print(f"the shape of node_feature_sums:\n {node_feature_sums.shape}")

            # Keep track of how many triangles this node belongs to
            node_counts[node_index] += 1
            #print(f"node_counts:\n {node_counts}")
    #print(f"node_feature_sums: {node_counts[:10]}")
    #print(f"the dtype of node_feature_sums: {node_feature_sums.dtype}")
    #print(f"node_counts: {node_counts[:10]}")

    average_node_features = node_feature_sums / np.maximum(node_counts[:, None], 1)  # (11236, 31), the shape of node_counts[:, None] becomes (n, 1)
    #print(f"average_features: {average_node_features[:2]}")
    #print(f"the shape of average_features: {average_node_features.shape}")

    return average_node_features

def load_displacement_op10(h5_path):
    _, displacement_vectors = extract_point_springback(h5_path, operation=10)  # OP10
    #print(f"the shape of displacement_vectors:\n {displacement_vectors.shape}")
    return displacement_vectors.astype(np.float32)  #  (11236, 3)

def prepare_sample(h5_path, component="blank", op_form=10, timestep=3):
    # Load per-element strain/stress/thickness features (11025, 31)
    concatenated_features = load_element_features(h5_path, component, timestep, op_form)  # (11025, 31)

    # Repeat to match number of triangle elements (22050)
    # similar to thickness_per_triangle = np.repeat(thickness, 2)[:len(triangles)]
    repeated_elem_feats = np.repeat(concatenated_features, 2, axis=0)  # (22050, 31), axis=0 along the rows axis.
    #print(f"the shape of repeated_elem_feats:\n {repeated_elem_feats.shape}") 

    # Load mesh triangles
    node_coords, triangles = load_quad_mesh(h5_path, component, op_form)  # triangles: (22050, 3)

    # # Safety version (keep if needed later)
    # valid_mask = np.all(triangles < raw_displacement.shape[0], axis=1)
    # triangles = triangles[valid_mask]
    # repeated_elem_feats = repeated_elem_feats[valid_mask]

    # Load displacement 
    raw_displacement = load_displacement_op10(h5_path)  # (11236, 3)
    num_nodes = raw_displacement.shape[0] # num_nodes: 11236
    #print(f"num_nodes: {num_nodes}")
    average_node_features = element_to_node_features(num_nodes, triangles, repeated_elem_feats) # (11236, 31)
    #print(f"the shape of node_feats:\n {node_feats.shape}")
    node_displacement = raw_displacement[:num_nodes] # (11236, 3)
    #print(f"the shape of node_displacement:\n {node_displacement.shape}")
    #print(f"one of node_displacement:\n {node_displacement[:1]}")
    return average_node_features, node_displacement

def edge_index_from_triangles(triangles: np.ndarray) -> torch.LongTensor:
    """Undirected edges from triangles → edge_index (2, E)."""
    edges = set()
    for tri in triangles:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edges.update([(i, j), (j, i), (j, k), (k, j), (k, i), (i, k)])
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()  # (2, E)
    return ei

def subgraph_first_k(x, y, triangles, k):
    N = x.shape[0]; k = min(int(k), int(N))
    keep = np.arange(k)
    mask = np.all(triangles < k, axis=1)
    return x[:k], y[:k], triangles[mask]

class FormingDisplacementDatasetGPS(Dataset):
    def __init__(self, base_dataset, max_nodes=None):
        self.base = base_dataset
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _, _, h5_path = self.base[idx]
        x, y = prepare_sample(h5_path)                  # x:(N, F), y:(N, 3)
        _, triangles = load_quad_mesh(h5_path)          # (T, 3)
        if self.max_nodes is not None:
            x, y, triangles = subgraph_first_k(x, y, triangles, self.max_nodes)
        edge_index = edge_index_from_triangles(triangles)  # (2, E)
        return {
            "x": torch.tensor(x, dtype=torch.float32),         # (N, F)
            "edge_index": edge_index,                          # (2, E) long
            "y": torch.tensor(y, dtype=torch.float32),         # (N, 3)
        }

def collate_graphs(batch):
    """
    batch: list of dicts {x:(Ni,F), edge_index:(2,Ei), y:(Ni,3)}
    returns: X:(sumN,F), EI:(2,sumE), Y:(sumN,3), batch_idx:(sumN,)
    """
    Xs, Ys, EIs, B = [], [], [], []
    offset = 0
    for i, item in enumerate(batch):
        x = item["x"]              # (Ni,F)
        y = item["y"]              # (Ni,3)
        ei = item["edge_index"]    # (2,Ei) long

        Xs.append(x)
        Ys.append(y)
        EIs.append(ei + offset)    # offset node ids
        B.append(torch.full((x.size(0),), i, dtype=torch.long))
        offset += x.size(0)

    X = torch.cat(Xs, dim=0)               # (sumN,F)
    Y = torch.cat(Ys, dim=0)               # (sumN,3)
    EI = torch.cat(EIs, dim=1).contiguous()# (2,sumE)
    batch_idx = torch.cat(B, dim=0)        # (sumN,)
    return {"x": X, "edge_index": EI, "y": Y, "batch_idx": batch_idx}

# ---- build dataset/loaders ----
full_dataset = FormingDisplacementDatasetGPS(dataset, max_nodes=1024)  # start with 512–2048 for safety
train_frac, test_frac, eval_frac = 0.7, 0.2, 0.1
train_dataset, test_dataset, eval_dataset = random_split(full_dataset, [train_frac, test_frac, eval_frac])
print(len(train_dataset), len(test_dataset), len(eval_dataset))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  collate_fn=collate_graphs)
eval_loader  = DataLoader(eval_dataset,  batch_size=8, shuffle=False,  collate_fn=collate_graphs)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False,  collate_fn=collate_graphs)

# ---- model, loss, opt ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_dim = 13
model = GPSFlexHybrid(in_dim=in_dim, model_dim=64, attn_dim=32, depth=4, mlp_ratio=2.0, drop=0.0, out_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch")
    run, totN = 0.0, 0
    for batch in pbar:
        X  = batch["x"].to(device)              # (sumN,F)
        EI = batch["edge_index"].to(device)     # (2,sumE)
        Y  = batch["y"].to(device)              # (sumN,3)
        B  = batch["batch_idx"].to(device)      # (sumN,)

        optimizer.zero_grad()
        Y_hat = model(X, EI, B)                 # (sumN,3)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()

        run += loss.item() * X.size(0)
        totN += X.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    print(f"Epoch {epoch}: Train Loss = {run/max(1,totN):.6f}")

    model.eval()
    vr, vN = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Epoch {epoch} [Eval]", unit="batch"):
            X  = batch["x"].to(device)
            EI = batch["edge_index"].to(device)
            Y  = batch["y"].to(device)
            B  = batch["batch_idx"].to(device)
            Y_hat = model(X, EI, B)
            vr += criterion(Y_hat, Y).item() * X.size(0)
            vN += X.size(0)
    print(f"Epoch {epoch}: Val Loss = {vr/max(1,vN):.6f}")