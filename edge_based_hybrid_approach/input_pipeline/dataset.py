import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
from DDACSDataset import DDACSDataset
from utils_DDACS import extract_point_cloud, extract_mesh, display_structure, extract_element_thickness, extract_point_springback

# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

# Get a sample simulation
sim_id, metadata, h5_path = dataset[20]
print(f"Sample simulation: {sim_id}")
print(f"File: {h5_path}")
print(f"Metadata: {metadata}")

component = "blank"
timestep = 3

with h5py.File(h5_path, "r") as f:
    comp = f["OP10"][component]

    # coordinates
    coords_all = comp["node_coordinates"][...]           # (T,N,3) or (N,3)
    if coords_all.ndim == 3:
        coords_t = coords_all[timestep]                  # (N,3)
    elif coords_all.ndim == 2:
        coords_t = coords_all                            # already (N,3)
    else:
        raise ValueError(f"Unexpected coords shape: {coords_all.shape}")

    # displacements (may be absent or (T,N,3) or (N,3))
    disp_t = None
    if "node_displacement" in comp:
        disp_all = comp["node_displacement"][...]
        if disp_all.ndim == 3:
            disp_t = disp_all[timestep]                  # (N,3)
        elif disp_all.ndim == 2:
            disp_t = disp_all                            # (N,3)
        else:
            raise ValueError(f"Unexpected disp shape: {disp_all.shape}")

print("coords_t shape:", coords_t.shape)
print("disp_t   shape:", None if disp_t is None else disp_t.shape)

# build features
X_t = np.concatenate([coords_t, disp_t], axis=1) if disp_t is not None else coords_t
print("Vertex feature matrix shape:", X_t.shape)
print("First 5 rows:\n", X_t[:5])