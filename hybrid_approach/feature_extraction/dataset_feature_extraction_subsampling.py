import numpy as np
import h5py
from pathlib import Path
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from DDACSDataset import DDACSDataset
from utils_DDACS import  extract_mesh, extract_element_thickness, extract_point_springback, extract_point_cloud
import torch
from tqdm import tqdm
import time
from pathlib import Path

# Get concatenated_strainstressthickness_features
def load_element_features(h5_path, component="blank", timestep=3, op_form=10):
    with h5py.File(h5_path, "r") as f:
        comp = f[f"OP{op_form}"][component]
        stress_t = comp["element_shell_stress"][timestep]  # (m,3,6)
        strain_t = comp["element_shell_strain"][timestep]  # (m,2,6)
        thickness_t = comp["element_shell_thickness"][timestep]  # (m,1)
        
    concatenated_features = np.concatenate([stress_t.reshape(stress_t.shape[0], -1).astype(np.float32),  # 18
                                            strain_t.reshape(strain_t.shape[0], -1).astype(np.float32),  # 12
                                            thickness_t.reshape(thickness_t.shape[0], -1).astype(np.float32) # 1
                                            ], axis=1)  # (m,31)
    #print(f"the shape of concatenated_features:\n {concatenated_features.shape}")

    #strain_features = strain_t.reshape(strain_t.shape[0], -1).astype(np.float32)  # (m, 12)
    #thickness = extract_element_thickness(h5_path, timestep=timestep, operation=op_form).astype(np.float32)  
    #concatenated_features = np.concatenate([concatenated_strainstress_features, thickness[:, None]], axis=1)  

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

# Get displacement
def load_displacement_op10(h5_path, operation = 10):
    _, displacement_vectors = extract_point_springback(h5_path, operation)  # OP10
    #print(f"the shape of displacement_vectors:\n {displacement_vectors.shape}")
    return displacement_vectors.astype(np.float32)  #  (11236, 3)

# Append edge into definite triangle
def add_edge(edges_dict, node_u, node_v, tri_idx):
    """Register an undirected edge (min, max) and track which triangle(s) it belongs to.
        node_v, node_u: the two original node indices that form an edge of a triangle
        node_a, node_b the canonicalized (sorted) version of (u, v).
    """
    if node_u < node_v:
        node_a, node_b = (node_u , node_v)
    else:
        node_a, node_b = (node_v, node_u)

    if (node_a, node_b) not in edges_dict:
        edges_dict[(node_a, node_b)] = []

    # The edge between nodes node_a and node_b is used by triangles(tri_idx)
    #print(f"triangle index: {tri_idx}")
    edges_dict[(node_a, node_b)].append(tri_idx)
    #print(f"edges_dict: {edges_dict}\n")

# Build edges
def build_edges_from_triangles(triangles):
    """
    Build unique undirected edges and record incident triangles per edge.
    Returns:
      edge_index: (E, 2) int64
      edge2tris: a list-of-lists that records, for each edge, which triangle(s) in the mesh contain that edge.
    """
    edges_dict = {}
    # Each triangle contributes 3 edges
    #print(f"the shape of triangles: {triangles.shape}") # (22050, 3)
    for tri_idx, (i, j, k) in enumerate(triangles):
        add_edge(edges_dict, i, j, tri_idx)
        add_edge(edges_dict, j, k, tri_idx)
        add_edge(edges_dict, k, i, tri_idx)

    # Finalize edge list
    keys = list(edges_dict.keys())
    edge_list = np.array(keys, dtype=np.int64)
    #print(f"edge list: {edge_list}")
    #print(f"the shape of edge list: {edge_list.shape}")
    edge2tris = [edges_dict[k] for k in keys]        # length E
    #print(f"edge2tris: {edge2tris[0]}")

    return edge_list, edge2tris

# Compute edge features
def compute_edge_features(edge_index, edge2tris, repeated_elem_feats):
    """
    Edge features = mean over incident triangle features (stress 18 + strain 12 + thickness 1 = 31).

    Args:
      edge_index:          (E, 2) int64 (not used here but kept for clarity if needed later)
      edge2tris:           list of lists of triangle indices per edge
      repeated_elem_feats: (T, 31) float32 per-triangle features

    Returns:
      edge_features: (E, 31) float32
    """
    Edge_nums = edge_index.shape[0]
    Feat_length = repeated_elem_feats.shape[1]  # 31
    edge_features = np.zeros((Edge_nums, Feat_length ), dtype=np.float32)

    for edge in range(Edge_nums):
        #print(f"edge:{edge}")
        tri_ids = edge2tris[edge]
        #print(f"triangle index: {tri_ids}")
        tri_ids_array = np.asarray(tri_ids, dtype=np.int64)
        #print(f"triangle index array: {tri_ids_array}")
        edge_features[edge] = repeated_elem_feats[tri_ids_array].mean(axis=0)
        #print(f"edge_features: {edge_features}")
    #print(f"triangle index: {tri_ids}")
    #print(f"triangle index array: {tri_ids_array}")
    #print(f"edge_features: {edge_features}")
    #print(f"the shape of edge_features:\n {edge_features.shape}")
    return edge_features

# Farthest point sampling
def farthest_point_sampling(node_coords: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    FPS over Euclidean coords. points: (N,3), returns k unique indices in [0,N).
    k: how many node indices I want to keep (500/1000/2000)
    """
    N = node_coords.shape[0]
    if k >= N:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(N))

    # Pick one random starting index.
    selected = np.empty(k, dtype=np.int64) # Make a length-k NumPy array of integers without initializing its contents
    selected[0] = first

    # Initialize the “nearest distance to the selected set”
    d = np.full(N, np.inf, dtype=np.float32)

    # Prime distances using the first center
    diff = node_coords - node_coords[first]
    d = np.minimum(d, np.linalg.norm(diff, axis=1)) # dist[i] = sqrt( (dx)^2 + (dy)^2 + (dz)^2 )
    for t in range(1, k):
        next = int(np.argmax(d)) # farthest from current selected set
        selected[t] = next
        diff = node_coords - node_coords[next]
        d = np.minimum(d, np.linalg.norm(diff, axis=1)) # After the update, d[i] remains the distance to the nearest selected node (now among all chosen so far).
    return selected # Return the k indices

def random_sampling(N: int, k: int, seed: int = 0) -> np.ndarray:
    if k >= N:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=k, replace=False).astype(np.int64)

# Subsample pipeline (recompute edges/features from submesh)
def subsample_graph(
    *, 
    k: int,
    sampling: str,
    seed: int,
    triangles: np.ndarray,                 # (T,3) on original node ids
    repeated_elem_feats: np.ndarray,       # (T,31)
    X: np.ndarray,                         # (N,34)
    Y: np.ndarray,                         # (N,3)
    node_coords: np.ndarray,                    # (N,3)
    node_index: np.ndarray                 # (N,)
):
    N = node_coords.shape[0]
    # 1) choose nodes
    if sampling.lower() == "fps":
        keep = farthest_point_sampling(node_coords, k, seed=seed)
    elif sampling.lower() == "random":
        keep = random_sampling(N, k, seed=seed)
    else:
        raise ValueError(f"Unknown sampling '{sampling}', use 'fps' or 'random'.")

    keep = np.asarray(keep, dtype=np.int64)
    k_eff = keep.shape[0]

    # remap node ids to 0..k_eff-1
    old2new = np.full(N, -1, dtype=np.int64)
    old2new[keep] = np.arange(k_eff, dtype=np.int64)

    # keep only triangles whose 3 nodes are all kept
    tri_keep_mask = np.all(np.isin(triangles, keep), axis=1)
    triangles_sub_old = triangles[tri_keep_mask]                 # indices in old space
    triangles_sub = old2new[triangles_sub_old]                   # (T',3) in new 0..k_eff-1

    # build edges from sub-triangles and recompute edge features
    edge_index_sub, edge2tris_sub = build_edges_from_triangles(triangles_sub)
    rep_feats_sub = repeated_elem_feats[tri_keep_mask]           # align with triangles_sub
    edge_features_sub = compute_edge_features(edge_index_sub, edge2tris_sub, rep_feats_sub)
    edge_index_2_sub = np.arange(edge_index_sub.shape[0], dtype=np.int64)

    # subset node-level arrays
    X_sub       = X[keep]
    Y_sub       = Y[keep]
    coords_sub  = node_coords[keep]
    node_idx_sub= node_index[keep]

    return (X_sub.astype(np.float32),
            Y_sub.astype(np.float32),
            edge_index_sub.astype(np.int64),
            edge_features_sub.astype(np.float32),
            coords_sub.astype(np.float32),
            node_idx_sub.astype(np.int64),
            edge_index_2_sub.astype(np.int64),
            keep)


def prepare_sample(h5_path, component="blank", op_form=10, timestep=3,
                   subsample_to: int | None = None, sampling: str = "fps", seed: int = 0):
    # 1) per-element → per-triangle features
    concatenated_features = load_element_features(h5_path, component, timestep, op_form)   # (m,31)
    repeated_elem_feats = np.repeat(concatenated_features, 2, axis=0)                      # (T=2m,31)

    # 2) mesh triangles and node coords
    node_coords, triangles = load_quad_mesh(h5_path, component, op_form, timestep)         # triangles: (T,3), coords: (N,3)

    # 3) node-level displacement (OP10)
    raw_displacement = load_displacement_op10(h5_path)                                     # (N,3)
    num_nodes = raw_displacement.shape[0]
    node_index = np.arange(num_nodes, dtype=np.int64)

    # 4) node features: avg triangle feats → nodes, then concat with coords
    average_node_features = element_to_node_features(num_nodes, triangles, repeated_elem_feats)  # (N,31)
    new_concatenated_features = np.concatenate([node_coords, average_node_features], axis=1)     # (N,34)
    node_displacement = raw_displacement[:num_nodes].astype(np.float32)                           # (N,3)

    # ---- If NOT subsampling: original full graph ----
    if subsample_to is None:
        edge_index, edge2tris = build_edges_from_triangles(triangles)
        edge_features = compute_edge_features(edge_index, edge2tris, repeated_elem_feats)
        edge_index_2 = np.arange(edge_index.shape[0], dtype=np.int64)
        return (new_concatenated_features.astype(np.float32),
                node_displacement.astype(np.float32),
                edge_index.astype(np.int64),
                edge_features.astype(np.float32),
                node_coords.astype(np.float32),
                node_index.astype(np.int64),
                edge_index_2.astype(np.int64))

    # ---- Subsampled graph (recompute edges/features from submesh) ----
    return subsample_graph(
        k=int(subsample_to),
        sampling=sampling,
        seed=seed,
        triangles=triangles,
        repeated_elem_feats=repeated_elem_feats,
        X=new_concatenated_features,
        Y=node_displacement,
        node_coords=node_coords,
        node_index=node_index
    )

def features_per_sample(ddacs, out_dir: Path, action="save_npy",
                        subsample_sizes=(500, 1000, 2000),
                        sampling="fps", seed=0, also_save_full=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    if action == "save_npy":
        n = len(ddacs)
        t0 = time.perf_counter()
        for i in tqdm(range(n), desc="Saving features (NPY per array)"):
            sample_id, _, h5_path = ddacs[i]

            # Optionally save the full graph
            if also_save_full:
                (X, Y, EI, EF, node_coords, node_idx, EI_2) = prepare_sample(h5_path)
                np.save(out_dir / f"{sample_id}_new_concatenated_features.npy", X)
                np.save(out_dir / f"{sample_id}_node_displacement.npy", Y)
                np.save(out_dir / f"{sample_id}_edge_index.npy", EI)
                np.save(out_dir / f"{sample_id}_edge_features.npy", EF)
                np.save(out_dir / f"{sample_id}_node_coords.npy", node_coords)
                np.save(out_dir / f"{sample_id}_node_index.npy", node_idx)
                np.save(out_dir / f"{sample_id}_edge_index_2.npy", EI_2)

            # Save each subsampled variant
            for k in subsample_sizes:
                sub_dir = out_dir / f"N{k}"
                sub_dir.mkdir(parents=True, exist_ok=True)

                (Xk, Yk, EIk, EFk, Ck, Nk, EI2k, keep_idx) = prepare_sample(
                    h5_path, subsample_to=k, sampling=sampling, seed=seed
                )

                np.save(sub_dir / f"{sample_id}_new_concatenated_features.npy", Xk)
                np.save(sub_dir / f"{sample_id}_node_displacement.npy",Yk)
                np.save(sub_dir / f"{sample_id}_edge_index.npy",EIk)
                np.save(sub_dir / f"{sample_id}_edge_features.npy",EFk)
                np.save(sub_dir / f"{sample_id}_node_coords.npy",Ck)
                np.save(sub_dir / f"{sample_id}_node_index.npy",Nk)
                np.save(sub_dir / f"{sample_id}_edge_index_2.npy",EI2k)
                # (Optional) save the mapping from new→old ids for audit
                np.save(sub_dir / f"{sample_id}_kept_old_node_ids.npy", keep_idx.astype(np.int64))

        total_time = time.perf_counter() - t0
        print("\n=== Save summary ===")
        print(f"root dir: {out_dir}")
        print(f"variants: {[f'N{k}' for k in subsample_sizes]} (sampling='{sampling}')")
        print(f"total time: {total_time:.2f} s")

    elif action == "check_npy":
        # adjust for a subsampled dir if you want to inspect a specific K
        sample_to_check = "16039"
        sub_dir = out_dir / "N500"  # change to N1000/N2000 or remove for full
        X  = np.load(sub_dir / f"{sample_to_check}_new_concatenated_features.npy")
        Y  = np.load(sub_dir / f"{sample_to_check}_node_displacement.npy")
        EI = np.load(sub_dir / f"{sample_to_check}_edge_index.npy")
        EF = np.load(sub_dir / f"{sample_to_check}_edge_features.npy")
        node_coords = np.load(sub_dir / f"{sample_to_check}_node_coords.npy")
        node_index  = np.load(sub_dir / f"{sample_to_check}_node_index.npy")
        EI_2 = np.load(sub_dir / f"{sample_to_check}_edge_index_2.npy")
        print("-----------------------------------")
        print(f"For sample {sample_to_check} (dir {sub_dir}):")
        print("-----------------------------------")
        print(f"N={X.shape[0]}, E={EI.shape[0]}")
        print(f"new_concatenated_features[0]:\n{X[0]}")
        print(f"node_displacement[0]:\n{Y[0]}")
        print(f"edge_index[0]: {EI[0]}")
        print(f"edge_features[0]:\n{EF[0]}")
        print(f"node_coords[0]:\n{node_coords[0]}")
        print(f"node_index[0]:\n{node_index[0]}")
        print(f"edge_index_2[0]: {EI_2[0]}")
    else:
        print("action must be one of: 'save_npy', 'check_npy'")

# set random seed
torch.manual_seed(0)

# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

# Add the save path
OUT_DIR = Path("/mnt/data/jiang/subsample")
OUT_DIR.mkdir(parents=True, exist_ok=True)

features_per_sample(
    dataset,
    OUT_DIR,
    action="save_npy",
    subsample_sizes=(500, 1000, 2000),
    sampling="fps",          # or "random"
    seed=0,
    also_save_full=False     # set True if the original full graphs are needed.
)