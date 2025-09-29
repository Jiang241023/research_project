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
def load_quad_mesh(h5_path, component="blank", op_form=10, timestep = 3):
    node_coords, triangles = extract_mesh(h5_path, operation=op_form, component=component, timestep = timestep)
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
    node_coords, triangles = load_quad_mesh(h5_path, component, op_form, timestep)  # triangles: (22050, 3)

    # # Safety version (keep if needed later)
    # valid_mask = np.all(triangles < raw_displacement.shape[0], axis=1)
    # triangles = triangles[valid_mask]
    # repeated_elem_feats = repeated_elem_feats[valid_mask]

    # Load displacement 
    raw_displacement = load_displacement_op10(h5_path)  # (11236, 3)
    num_nodes = raw_displacement.shape[0] # num_nodes: 11236
    #print(f"num_nodes: {num_nodes}")
    
    average_node_features = element_to_node_features(num_nodes, triangles, repeated_elem_feats) # (11236, 31)
    new_concatenated_features = np.concatenate([node_coords, average_node_features], axis=1)
    #print(f"the shape of node_feats:\n {node_feats.shape}")
    node_displacement = raw_displacement[:num_nodes] # (11236, 3)
    #print(f"the shape of node_displacement:\n {node_displacement.shape}")
    #print(f"one of node_displacement:\n {node_displacement[:1]}")
    return new_concatenated_features, node_displacement

# Wrap full dataset
class FormingDisplacementDataset(Dataset):
    def __init__(self, base_dataset):  # base_dataset = DDACSDataset(...)
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        _, _, h5_path = self.base_dataset[idx]
        x, y = prepare_sample(h5_path)  # from earlier
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
# Wrap full dataset
full_dataset = FormingDisplacementDataset(dataset)

# Then split using random_split
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_frac, test_frac, eval_frac = 0.7, 0.2, 0.1

train_dataset, test_dataset, eval_dataset = random_split(full_dataset, [train_frac, test_frac, eval_frac])

new_concatenated_features, node_displacement = train_dataset[0]
print(f"new_concatenated_features of nodes: {new_concatenated_features[0]}") # x,y,z + other features 

print(f"node_displacement: {node_displacement[0]}")

print(len(train_dataset), len(test_dataset), len(eval_dataset))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)