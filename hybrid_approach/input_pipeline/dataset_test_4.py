import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from collections import defaultdict
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from DDACSDataset import DDACSDataset
from utils.utils_DDACS import extract_mesh, extract_element_thickness, extract_point_springback
from model.vertex_based_model_2 import VertexHybridModel
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# set random seed
torch.manual_seed(0)

# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset  = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")


# Get concatenated features
def load_element_features(h5_path, component="blank", timestep=3, op_form=10):
    with h5py.File(h5_path, "r") as f:
        comp = f[f"OP{op_form}"][component]
        strain_t = comp["element_shell_strain"][timestep]  # (m,2,6)
        #stress_t = comp["element_shell_stress"][timestep]  # (m,3,6)
    # concatenated_strainstress_features = np.concatenate([stress_t.reshape(stress_t.shape[0], -1).astype(np.float32),  # 18
    #                                                     strain_t.reshape(strain_t.shape[0], -1).astype(np.float32),  # 12
    #                                                     ], axis=1)  # (m,30)
    strain_features = strain_t.reshape(strain_t.shape[0], -1).astype(np.float32)  # (m,12)
    thickness = extract_element_thickness(h5_path, timestep=timestep, operation=op_form).astype(np.float32)
    concatenated = np.concatenate([strain_features, thickness[:, None]], axis=1)   # (m,13)
    return concatenated

# Get quad mesh
def load_quad_mesh(h5_path, component="blank", op_form=10):
    node_coords, triangles = extract_mesh(h5_path, operation=op_form, component=component)
    return node_coords.astype(np.float32), triangles.astype(np.int64)

# Project element features to node features
def element_to_node_features(num_nodes, triangles, elem_features):
    node_sums = np.zeros((num_nodes, elem_features.shape[1]), dtype=np.float32) # (11236, 13)
    #print(f"the shape of the node_sums:\n {node_sums.shape}")
    node_counts = np.zeros(num_nodes, dtype=np.int32) # (11236,)
    #print(f"the shape of the node_counts:\n {node_counts.shape}")
    #print(f"the shape of the triangle feature:\n {triangles.shape}")

    for tri_idx in range(len(triangles)):
        feat = elem_features[tri_idx]     # (F=13,)
        i, j, k = triangles[tri_idx]
        # print(f"k: {k}")
        for n in (i, j, k):
            node_sums[n] += feat
            node_counts[n] += 1
    #print(f"one element of node_sums: {node_sums[:1]}")
    return node_sums / np.maximum(node_counts[:, None], 1)

# Get the displacement at time step 3
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

# ---------------- subgraph + adjacencies ----------------
def subgraph_first_k(node_feats, node_disp, triangles, k):
    """Keep the first k nodes and all faces fully inside [0..k-1]."""
    N = node_feats.shape[0]
    k = min(int(k), int(N))
    mask = np.all(triangles < k, axis=1)
    return node_feats[:k], node_disp[:k], triangles[mask], k


def dense_adj_from_triangles(triangles, num_nodes):
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    if triangles.size == 0:
        return A
    i = torch.from_numpy(triangles[:, 0])
    j = torch.from_numpy(triangles[:, 1])
    k = torch.from_numpy(triangles[:, 2])
    r = torch.cat([i, j, j, k, k, i], dim=0)
    c = torch.cat([j, i, k, j, i, k], dim=0)
    A[r, c] = True
    return A


def _add_clique_(Ac: torch.Tensor, verts):
    """
    In-place: add a complete subgraph on 'verts' (list/tuple of ints).
    """
    m = len(verts)
    for a in range(m):
        va = int(verts[a])
        for b in range(a + 1, m):
            vb = int(verts[b])
            Ac[va, vb] = 1.0
            Ac[vb, va] = 1.0


def clique_adjacency_from_cycles(triangles: np.ndarray, num_nodes: int, add_quads: bool = True) -> torch.Tensor:
    """
    Clique adjacency A_c as in the paper:
      - add K3 for every triangle
      - (bounded version) detect adjacent-triangle pairs (4-cycles) and add K4 on the 4 vertices

    triangles: (T,3) int64 numpy
    returns:   (num_nodes, num_nodes) float32 torch tensor
    """
    Ac = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if triangles.size == 0:
        return Ac

    tri = triangles.astype(np.int64)

    # ---- 1) K3 for every triangle (each triangle is already a clique) ----
    for a, b, c in tri:
        _add_clique_(Ac, (a, b, c))

    if not add_quads:
        return Ac

    # ---- 2) K4 for every pair of adjacent triangles (bounded cycle of length 4) ----
    # Build map: undirected edge -> list of incident triangle indices
    edge2tris = defaultdict(list)
    for t_idx, (a, b, c) in enumerate(tri):
        for u, v in ((a, b), (b, c), (c, a)):
            if v < u:  # store edges as (min,max)
                u, v = v, u
            edge2tris[(u, v)].append(t_idx)

    # For each shared edge, combine incident triangles;
    # if their union has 4 distinct vertices, add a K4 on those vertices
    for _, incident in edge2tris.items():
        if len(incident) < 2:
            continue
        # could be more than 2 in non-manifold areas; consider all pairs
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                t1 = tri[incident[i]]
                t2 = tri[incident[j]]
                verts = set([int(t1[0]), int(t1[1]), int(t1[2]),
                             int(t2[0]), int(t2[1]), int(t2[2])])
                if len(verts) == 4:          # a proper 4-cycle
                    _add_clique_(Ac, list(verts))

    # zero diagonal just in case
    Ac.fill_diagonal_(0.0)
    return Ac


# ---------------- dataset ----------------
class FormingDisplacementDataset(Dataset):
    def __init__(self, base_dataset, max_nodes=11236):
        self.base = base_dataset
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _, _, h5_path = self.base[idx]
        node_feats, node_disp, triangles = prepare_sample(h5_path)   # (N,13), (N,3), (T,3)
        _, tri_global = load_quad_mesh(h5_path)

        x_sub, y_sub, tri_sub, k = subgraph_first_k(node_feats, node_disp, tri_global, self.max_nodes)

        # Standard adjacency (edges from faces)
        A  = dense_adj_from_triangles(tri_sub, k)              # (k,k)
        # Clique adjacency per the paper (K3 + bounded K4s)
        Ac = clique_adjacency_from_cycles(tri_sub, k, add_quads=True)

        return {
            'x': torch.tensor(x_sub, dtype=torch.float32),     # (k,13)
            'adj': A,                                          # (k,k)
            'clique': Ac,                                      # (k,k)
            'y': torch.tensor(y_sub, dtype=torch.float32)      # (k,3)
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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=2, pin_memory=True)
eval_loader  = DataLoader(eval_dataset,  batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, num_workers=2, pin_memory=True)


# ---------------- model / train ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feat_dim = 13
model = VertexHybridModel(in_dim=feat_dim, hidden_dim=16, attn_dim=16, L=3, reduction=4, out_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 1
use_amp = (device.type == "cuda")
scaler  = torch.amp.GradScaler(enabled=use_amp)

def amp_ctx():
    # returns a context manager each time
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")

    for batch in pbar:
        X = batch['x'].to(device, non_blocking=True)   # (B,N,13)
        Y = batch['y'].to(device, non_blocking=True)   # (B,N,3)
        A  = batch['adj']      # keep on CPU if your model handles it
        Ac = batch['clique']   # keep on CPU if your model handles it

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            Y_hat = model(X, A, Ac)
            loss  = criterion(Y_hat, Y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = X.size(0)
        total_loss   += loss.item() * bs
        total_graphs += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch}: Train Loss = {total_loss / max(1, total_graphs):.6f}")

    # ---------- EVAL ----------
    model.eval()
    val_sum = 0.0
    val_graphs = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Epoch {epoch} [Eval]", unit="batch"):
            X = batch['x'].to(device, non_blocking=True)
            Y = batch['y'].to(device, non_blocking=True)
            A  = batch['adj']     # CPU
            Ac = batch['clique']  # CPU

            with amp_ctx():
                Y_hat = model(X, A, Ac)
                loss  = criterion(Y_hat, Y)

            bs = X.size(0)
            val_sum    += loss.item() * bs
            val_graphs += bs

    print(f"Epoch {epoch}: Val Loss = {val_sum / max(1, val_graphs):.6f}")