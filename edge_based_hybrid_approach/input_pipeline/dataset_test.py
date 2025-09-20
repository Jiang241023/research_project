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

# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

class StrainStressDataset(Dataset):
    def __init__(self, base_dataset, component="blank", timestep=3, operation="OP10"):
        self.base_dataset = base_dataset
        self.component = component
        self.timestep = timestep
        self.operation = operation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sim_id, metadata, h5_path = self.base_dataset[idx]
        with h5py.File(h5_path, "r") as f:
            comp = f[self.operation][self.component]

            # inputs
            stress = comp["element_shell_stress"][self.timestep]   # (m, 3, 6)
            strain = comp["element_shell_strain"][self.timestep]   # (m, 2, 6)

            stress_avg = stress.mean(axis=1)   # (m, 6)
            strain_avg = strain.mean(axis=1)   # (m, 6)

            thickness = comp["element_shell_thickness"][self.timestep]  # (m,)

            # targets
            disp = comp["node_displacement"][self.timestep]  # (n, 3)

        # Stack features per element: [strain, stress, thickness]
        # For simplicity: concatenate strain_avg + stress_avg + thickness[:,None]
        x = np.concatenate([strain_avg, stress_avg, thickness[:,None]], axis=1)  # (m, 13)

        y = disp  # displacement per node (different size than m!)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


N = len(dataset)  # 32071
train_frac, test_frac, eval_frac = 0.7, 0.2, 0.1

train_len = int(N * train_frac)
test_len  = int(N * test_frac)
eval_len  = N - train_len - test_len  # ensure exact

torch.manual_seed(0)  # reproducible split
train_dataset, test_dataset, eval_dataset = random_split(
    StrainStressDataset(dataset),
    [train_len, test_len, eval_len]
)

print(len(train_dataset), len(test_dataset), len(eval_dataset))
for i in range(2):
    x, y = train_dataset[i]   # get i-th sample
    print(f"Sample {i}:")
    print("  x shape:", x.shape)   # (m, 13)
    print("  y shape:", y.shape)   # (n, 3)
    print("  first row of x:\n", x[0])
    print("  first row of y:\n", y[0])



train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)
eval_loader  = DataLoader(eval_dataset, batch_size=4, shuffle=False)


