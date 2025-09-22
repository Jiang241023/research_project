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

# set random seed
torch.manual_seed(0)

# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

class StrainStressThicknessDataset(Dataset):
    def __init__(self, base_dataset, component, timestep, operation, op_num):
        """
        base_dataset: DDACSDataset providing (sim_id, metadata, h5_path)
        component: e.g., "blank"
        timestep: forming timestep to sample inputs from OP10
        operation: "OP10" group name for inputs (stress/strain/thickness)
        op_num: numeric operation for springback helper (usually 10)
        """
        self.base_dataset = base_dataset
        self.component = component
        self.timestep = timestep
        self.operation = operation
        self.op_num = op_num

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sim_id, metadata, h5_path = self.base_dataset[idx]

        # --- inputs from OP10, element domain (m elements) ---
        with h5py.File(h5_path, "r") as f:
            comp = f[self.operation][self.component]

            stress_t = comp["element_shell_stress"][self.timestep]   # (m, 3, 6)
            strain_t = comp["element_shell_strain"][self.timestep]   # (m, 2, 6)

            # concatenated stress/strain features from each layer at timestep 3
            stress_feat = stress_t.reshape(stress_t.shape[0], -1)    # (m, 18)
            strain_feat = strain_t.reshape(strain_t.shape[0], -1)    # (m, 12)


        thickness = extract_element_thickness(h5_path, timestep=self.timestep, operation=self.op_num)  # (m,)

        # targets from springback OP10
        final_coords, displacement_vectors = extract_point_springback(h5_path, operation=self.op_num)  # (n,3), (n,3)

        # features per element: [strain_avg, stress_avg, thickness]
        x = np.concatenate([stress_feat, strain_feat, thickness[:, None]], axis=1).astype(np.float32)  # (m, 31)

        # target per node: displacement after springback
        y = displacement_vectors.astype(np.float32)  # (n, 3)

        return torch.from_numpy(x), torch.from_numpy(y)

train_frac, test_frac, eval_frac = 0.7, 0.2, 0.1
N = len(dataset)
train_len = int(N * train_frac)
test_len  = int(N * test_frac)
eval_len  = N - train_len - test_len  

train_dataset, test_dataset, eval_dataset = random_split(
    StrainStressThicknessDataset(dataset, component="blank", timestep=3, operation="OP10", op_num=10),
    [train_len , test_len, eval_len]
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

# get the first batch from the loader
# x_batch, y_batch = next(iter(train_loader))
# print("train batch -> x:", x_batch.shape, "y:", y_batch.shape)

# x_batch, y_batch = next(iter(test_loader))
# print("test batch -> x:", x_batch.shape, "y:", y_batch.shape)

# x_batch, y_batch = next(iter(eval_loader))
# print("eval batch -> x:", x_batch.shape, "y:", y_batch.shape)
