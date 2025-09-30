import numpy as np
import h5py
from pathlib import Path
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from input_pipeline.DDACSDataset import DDACSDataset
from utils.utils_DDACS import  extract_mesh, extract_element_thickness, extract_point_springback
import torch
from tqdm import tqdm
import time
from pathlib import Path

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

    concatenated_strainstress_features = np.concatenate([stress_t.reshape(stress_t.shape[0], -1).astype(np.float32),  # 18
                                                        strain_t.reshape(strain_t.shape[0], -1).astype(np.float32),  # 12
                                                        ], axis=1)  # (m,30)
    #strain_features = strain_t.reshape(strain_t.shape[0], -1).astype(np.float32)  # (m, 12)

    thickness = extract_element_thickness(h5_path, timestep=timestep, operation=op_form).astype(np.float32)  
    concatenated_features = np.concatenate([concatenated_strainstress_features, thickness[:, None]], axis=1)  
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

def _add_edge(edges_dict, u, v, tri_idx):
    """Register an undirected edge (min, max) and track which triangle(s) it belongs to."""
    a, b = (u, v) if u < v else (v, u)
    if (a, b) not in edges_dict:
        edges_dict[(a, b)] = []
    edges_dict[(a, b)].append(tri_idx)

def build_edges_from_triangles(triangles, num_nodes):
    """
    Build unique undirected edges and record incident triangles per edge.
    Returns:
      edge_index: (E, 2) int64
      edge2tris:  list of lists; edge2tris[e] = [triangle indices sharing this edge]
      degrees:    (num_nodes,) int32 node degree in the undirected edge graph
    """
    edges_dict = {}
    # Each triangle contributes 3 edges
    for t_idx, (i, j, k) in enumerate(triangles):
        _add_edge(edges_dict, i, j, t_idx)
        _add_edge(edges_dict, j, k, t_idx)
        _add_edge(edges_dict, k, i, t_idx)

    # Finalize edge list
    edge_list = np.array(list(edges_dict.keys()), dtype=np.int64)  # (E, 2)
    edge2tris = [edges_dict[k] for k in edges_dict.keys()]         # length E

    # Degrees: count edges incident to each node
    degrees = np.zeros(num_nodes, dtype=np.int32)
    np.add.at(degrees, edge_list[:, 0], 1)
    np.add.at(degrees, edge_list[:, 1], 1)

    return edge_list, edge2tris, degrees

# Compute edge features
def compute_edge_features(node_coords, edge_index, edge2tris, degrees, repeated_elem_feats):
    """
    Compute edge features.

    Args:
      node_coords:         (N, 3) float32
      edge_index:          (E, 2) int64 with u < v
      edge2tris:           list of lists of triangle indices per edge
      degrees:             (N,) int32
      repeated_elem_feats: (T, 31) float32 per-triangle features (already repeated from quads)

    Returns:
      edge_features: (E, 35) float32
    """
    E = edge_index.shape[0]
    edge_features = np.zeros((E, 35), dtype=np.float32)

    # Precompute for speed
    coords = node_coords.astype(np.float32)
    deg = degrees.astype(np.float32)

    for e in range(E):
        u, v = edge_index[e]
        # 1) geometric: length
        vec = coords[v] - coords[u]                      # (3,)
        length = np.linalg.norm(vec)                     # scalar

        # 2) degrees of endpoints
        deg_u = deg[u]
        deg_v = deg[v]

        # 3) shared triangles count (1 or 2 typically)
        tri_ids = edge2tris[e]
        shared = float(len(tri_ids))

        # 4) incident-triangle feature mean (31)
        if len(tri_ids) > 0:
            tri_feat_mean = repeated_elem_feats[np.array(tri_ids, dtype=np.int64)].mean(axis=0)
        else:
            tri_feat_mean = np.zeros((repeated_elem_feats.shape[1],), dtype=np.float32)

        # pack: [length, deg_u, deg_v, shared, tri_feat_mean(31)] → 35 dims
        edge_features[e, 0]  = length
        edge_features[e, 1]  = deg_u
        edge_features[e, 2]  = deg_v
        edge_features[e, 3]  = shared
        edge_features[e, 4:] = tri_feat_mean

    return edge_features


def prepare_sample(h5_path, component="blank", op_form=10, timestep=3):
    # 1) per-element features (m=11025) → per-triangle features (2m=22050)
    concatenated_features = load_element_features(h5_path, component, timestep, op_form)  # (11025, 31)
    repeated_elem_feats = np.repeat(concatenated_features, 2, axis=0)  # (22050, 31)

    # 2) mesh triangles and node coords
    node_coords, triangles = load_quad_mesh(h5_path, component, op_form, timestep)  # triangles: (22050, 3), coords: (N,3)

    # 3) node-level displacement (OP10)
    raw_displacement = load_displacement_op10(h5_path)  # (N, 3)
    num_nodes = raw_displacement.shape[0]

    # 4) node features: average triangle features to nodes then concat with coords
    average_node_features = element_to_node_features(num_nodes, triangles, repeated_elem_feats) # (N, 31)
    new_concatenated_features = np.concatenate([node_coords, average_node_features], axis=1)    # (N, 34)
    node_displacement = raw_displacement[:num_nodes].astype(np.float32)                         # (N, 3)

    # 5) EDGE INDEX + EDGE FEATURES
    edge_index, edge2tris, degrees = build_edges_from_triangles(triangles, num_nodes)           # (E,2)
    edge_features = compute_edge_features(node_coords, edge_index, edge2tris, degrees, repeated_elem_feats)  # (E,35)

    return new_concatenated_features.astype(np.float32), node_displacement, edge_index.astype(np.int64), edge_features.astype(np.float32)


# Add the save path
OUT_DIR = Path("/home/RUS_CIP/st186731/research_project/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save or check samples
def features_h5_per_sample(ddacs, out_dir, save_or_check):
    if save_or_check == "save":
        n = len(ddacs)
        t0 = time.perf_counter()

        for i in tqdm(range(n), desc="Saving features (HDF5 per sample)"):
            sample_id, _, h5_path = ddacs[i]
            out_file = out_dir / f"{sample_id}.h5"

            X, Y, edge_index, edge_features = prepare_sample(h5_path)
            X = X.astype(np.float32)                 # (N,34)
            Y = Y.astype(np.float32)                 # (N,3)
            EI = edge_index.astype(np.int64)         # (E,2)
            EF = edge_features.astype(np.float32)    # (E,35)

            with h5py.File(out_file, "w") as f:
                f.create_dataset("new_concatenated_features", data=X)
                f.create_dataset("node_displacement", data=Y)
                f.create_dataset("edge_index", data=EI)
                f.create_dataset("edge_features", data=EF)
                f.attrs["sample_id"] = int(sample_id)

        total_time = time.perf_counter() - t0
        print("\n=== Save summary ===")
        print(f"dir: {out_dir}")
        print(f"total time: {total_time:.2f} s ")

    elif save_or_check == "check":
        out_file = out_dir / f"16039.h5"
        with h5py.File(out_file, "r") as f:
            X  = f["new_concatenated_features"][:]   # (N, 34)
            Y  = f["node_displacement"][:]           # (N, 3)
            EI = f["edge_index"][:]                  # (E, 2)
            EF = f["edge_features"][:]               # (E, 35)
            sid = f.attrs["sample_id"]
            print(f"For sample {sid}:\n"
                  f"  X[0] = {X[0]}\n"
                  f"  Y[0] = {Y[0]}\n"
                  f"  EI[0] = {EI[0]}\n"
                  f"  EF[0] = {EF[0]}")
    else:
        print(r"U should choose save or check")

# Run it
features_h5_per_sample(dataset, OUT_DIR, save_or_check = "check")