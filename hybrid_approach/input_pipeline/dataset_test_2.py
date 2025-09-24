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
        stress_t = comp["element_shell_stress"][timestep]  # (m,3,6)
        strain_t = comp["element_shell_strain"][timestep]  # (m,2,6)
    concatenated_strainstress_features = np.concatenate([
        stress_t.reshape(stress_t.shape[0], -1).astype(np.float32),  # 18
        strain_t.reshape(strain_t.shape[0], -1).astype(np.float32),  # 12
    ], axis=1)  # (m,30)

    thickness = extract_element_thickness(h5_path, timestep=timestep, operation=op_form).astype(np.float32)  # (m,)
    concatenated_strainstressthickness_features = np.concatenate([concatenated_strainstress_features, thickness[:, None]], axis=1)  # (m,31)
    return concatenated_strainstressthickness_features  # (m,31)

# Get quad mesh
def load_quad_mesh(h5_path, component="blank", op_form=10):
    node_coords, elements = extract_mesh(h5_path, operation=op_form, component=component)
    return node_coords.astype(np.float32), elements.astype(np.int64)


# Project element features to node features
def element_to_node_features(num_nodes, elements, elem_features):
    node_feature_sums = np.zeros((num_nodes, elem_features.shape[1]), dtype=np.float32)
    node_counts = np.zeros(num_nodes, dtype=np.int32)

    for elem_idx, node_ids in enumerate(elements):
        for node_id in node_ids:
            node_feature_sums[node_id] += elem_features[elem_idx]
            node_counts[node_id] += 1

    return node_feature_sums / np.maximum(node_counts[:, None], 1)  # (n, 31)

def load_displacement_op10(h5_path):
    _, displacement_vectors = extract_point_springback(h5_path, operation=10)  # OP10
    return displacement_vectors.astype(np.float32)  # (n, 3)

def prepare_sample(h5_path, component="blank", op_form=10, timestep=3):
    # Load element-level features: shape (m_data, 31)
    elem_feats = load_element_features(h5_path, component, timestep, op_form)

    # Load mesh: node coords and all mesh elements: shape (m_mesh, 4)
    node_coords, elements = load_quad_mesh(h5_path, component, op_form)

    # Truncate elements to match number of strain/stress entries
    if elements.shape[0] > elem_feats.shape[0]:
        elements = elements[:elem_feats.shape[0]]
    elif elements.shape[0] < elem_feats.shape[0]:
        elem_feats = elem_feats[:elements.shape[0]]

    # Load displacement
    raw_displacement = load_displacement_op10(h5_path)
    max_valid_node = raw_displacement.shape[0] - 1

    # Filter elements that are fully within valid node range
    valid_mask = np.all(elements <= max_valid_node, axis=1)
    elements = elements[valid_mask]
    elem_feats = elem_feats[valid_mask]

    num_nodes = raw_displacement.shape[0]

    node_feats = element_to_node_features(num_nodes, elements, elem_feats)
    node_displacement = raw_displacement[:num_nodes]

    assert node_feats.shape[0] == node_displacement.shape[0], \
        f"Shape mismatch: node_feats {node_feats.shape}, disp {node_displacement.shape}"

    return node_feats, node_displacement


class NodeMLP(nn.Module):
    def __init__(self, in_dim=31, hidden_dim=64, out_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):  # x: (n, 31)
        return self.mlp(x)  # (n, 3)

class FormingDisplacementDataset(Dataset):
    def __init__(self, base_dataset):  # base_dataset = DDACSDataset(...)
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        _, _, h5_path = self.base_dataset[idx]
        x, y = prepare_sample(h5_path)  # from earlier
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
# Wrap full dataset with your custom Dataset class
full_dataset = FormingDisplacementDataset(dataset)

# Then split using random_split
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NodeMLP(in_dim=31, hidden_dim=128, out_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        x, y = batch  # each is (1, n, d)
        x = x.squeeze(0).to(device)  # (n, 31)
        y = y.squeeze(0).to(device)  # (n, 3)

        optimizer.zero_grad()
        pred = model(x)  # (n, 3)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            x, y = batch
            x = x.squeeze(0).to(device)
            y = y.squeeze(0).to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item()
    print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
