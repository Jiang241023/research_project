import numpy as np
import h5py
from pathlib import Path
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from DDACSDataset import DDACSDataset
from utils.utils_DDACS import extract_mesh, extract_element_thickness, extract_point_springback
from model.vertex_based_model_2 import VertexHybridModel

# ---------------- basic setup ----------------
torch.manual_seed(0)
data_dir = Path("/mnt/data/darus/")
dataset  = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")


# ---------------- feature builders ----------------
def load_element_features(h5_path, component="blank", timestep=3, op_form=10):
    with h5py.File(h5_path, "r") as f:
        comp = f[f"OP{op_form}"][component]
        strain_t = comp["element_shell_strain"][timestep]  # (m,2,6)
    strain_features = strain_t.reshape(strain_t.shape[0], -1).astype(np.float32)  # (m,12)
    thickness = extract_element_thickness(h5_path, timestep=timestep, operation=op_form).astype(np.float32)
    concatenated = np.concatenate([strain_features, thickness[:, None]], axis=1)   # (m,13)
    return concatenated


def load_quad_mesh(h5_path, component="blank", op_form=10):
    node_coords, triangles = extract_mesh(h5_path, operation=op_form, component=component)
    return node_coords.astype(np.float32), triangles.astype(np.int64)


def element_to_node_features(num_nodes, triangles, elem_features):
    node_sums   = np.zeros((num_nodes, elem_features.shape[1]), dtype=np.float32)
    node_counts = np.zeros(num_nodes, dtype=np.int32)

    for tri_idx in range(len(triangles)):
        feat = elem_features[tri_idx]     # (F=13,)
        i, j, k = triangles[tri_idx]
        for n in (i, j, k):
            node_sums[n] += feat
            node_counts[n] += 1

    return node_sums / np.maximum(node_counts[:, None], 1)


def load_displacement_op10(h5_path):
    _, disp = extract_point_springback(h5_path, operation=10)  # (N,3)
    return disp.astype(np.float32)


def prepare_sample(h5_path, component="blank", op_form=10, timestep=3):
    elem_feats = load_element_features(h5_path, component, timestep, op_form)  # (m,13)
    repeated   = np.repeat(elem_feats, 2, axis=0)                              # triangles doubled
    _, triangles = load_quad_mesh(h5_path, component, op_form)                 # (T,3)
    raw_disp   = load_displacement_op10(h5_path)                               # (N,3)
    num_nodes  = raw_disp.shape[0]
    node_feats = element_to_node_features(num_nodes, triangles, repeated)      # (N,13)
    node_disp  = raw_disp[:num_nodes]                                          # (N,3)
    return node_feats, node_disp, triangles


# --------------- adjacency from triangles (dense) ---------------
def subgraph_first_k(x, y, triangles, k):
    N = x.shape[0]
    k = min(int(k), int(N))
    mask = np.all(triangles < k, axis=1)
    return x[:k], y[:k], triangles[mask], k


def dense_adj_from_triangles(triangles: np.ndarray, num_nodes: int) -> torch.Tensor:
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if triangles.size == 0:
        return A
    i = torch.from_numpy(triangles[:, 0])
    j = torch.from_numpy(triangles[:, 1])
    k = torch.from_numpy(triangles[:, 2])
    r = torch.cat([i, j, j, k, k, i], dim=0)
    c = torch.cat([j, i, k, j, i, k], dim=0)
    A[r, c] = 1.0
    return A


# ---------------- dataset ----------------
class FormingDisplacementDataset(Dataset):
    def __init__(self, base_dataset, max_nodes=512):
        self.base = base_dataset
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _, _, h5_path = self.base[idx]
        x, y, triangles = prepare_sample(h5_path)         # x:(N,13), y:(N,3), triangles:(T,3)
        _, tri_global   = load_quad_mesh(h5_path)
        # we subgraph on the first K nodes (indices are already contiguous from 0)
        x, y, tri_sub, k = subgraph_first_k(x, y, tri_global, self.max_nodes)
        A  = dense_adj_from_triangles(tri_sub, k)         # (k,k)
        Ac = A.clone()                                    # reuse A as "clique"
        return {
            'x': torch.tensor(x, dtype=torch.float32),    # (k,13)
            'adj': A,                                     # (k,k)
            'clique': Ac,                                 # (k,k)
            'y': torch.tensor(y, dtype=torch.float32)     # (k,3)
        }


# ---------------- splits / loaders ----------------
full_dataset = FormingDisplacementDataset(dataset)

n = len(full_dataset)
train_len = int(0.7 * n)
test_len  = int(0.2 * n)
eval_len  = n - train_len - test_len

train_dataset, test_dataset, eval_dataset = random_split(
    full_dataset, [train_len, test_len, eval_len],
    generator=torch.Generator().manual_seed(42)
)

print(len(train_dataset), len(test_dataset), len(eval_dataset))

# fixed-size graphs ⇒ default collate works (B,N,*) and (B,N,N)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=2, pin_memory=True)
eval_loader  = DataLoader(eval_dataset,  batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=2, pin_memory=True)


# ---------------- model / train ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feat_dim = 13
model = VertexHybridModel(in_dim=feat_dim, hidden_dim=64, attn_dim=32, L=3, reduction=4, out_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 1
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")
    for batch in pbar:
        X  = batch['x'].to(device)        # (B,N,13)
        A  = batch['adj'].to(device)      # (B,N,N)
        Ac = batch['clique'].to(device)   # (B,N,N)
        Y  = batch['y'].to(device)        # (B,N,3)

        optimizer.zero_grad()
        Y_hat = model(X, A, Ac)           # (B,N,3)
        loss  = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        total_loss   += loss.item() * bs
        total_graphs += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    print(f"Epoch {epoch}: Train Loss = {total_loss / max(1, total_graphs):.6f}")

    # ---- eval ----
    model.eval()
    val_sum = 0.0
    val_graphs = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Epoch {epoch} [Eval]", unit="batch"):
            X  = batch['x'].to(device)
            A  = batch['adj'].to(device)
            Ac = batch['clique'].to(device)
            Y  = batch['y'].to(device)
            Y_hat = model(X, A, Ac)
            bs = X.size(0)
            val_sum     += criterion(Y_hat, Y).item() * bs
            val_graphs  += bs
    print(f"Epoch {epoch}: Val Loss = {val_sum / max(1, val_graphs):.6f}")
