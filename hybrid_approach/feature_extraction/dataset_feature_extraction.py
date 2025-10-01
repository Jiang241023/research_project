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
    

def prepare_sample(h5_path, component="blank", op_form=10, timestep=3):
    # 1) per-element features (m=11025) → per-triangle features (2m=22050)
    concatenated_features = load_element_features(h5_path, component, timestep, op_form)  # (11025, 31)
    repeated_elem_feats = np.repeat(concatenated_features, 2, axis=0)                     # (22050, 31)

    # 2) mesh triangles and node coords
    node_coords, triangles = load_quad_mesh(h5_path, component, op_form, timestep)        # triangles: (22050, 3), coords: (N,3)

    # 3) node-level displacement (OP10)
    raw_displacement = load_displacement_op10(h5_path)                                     # (N, 3)
    num_nodes = raw_displacement.shape[0]

    # 4) node features: avg triangle features → nodes, then concat with coords
    average_node_features = element_to_node_features(num_nodes, triangles, repeated_elem_feats)  # (N, 31)
    new_concatenated_features = np.concatenate([node_coords, average_node_features], axis=1)     # (N, 34)
    node_displacement = raw_displacement[:num_nodes].astype(np.float32)                          # (N, 3)

    # 5) EDGE INDEX + EDGE FEATURES (no degrees needed)
    edge_index, edge2tris = build_edges_from_triangles(triangles)                                  # (E,2), list-of-lists
    edge_features = compute_edge_features(edge_index, edge2tris, repeated_elem_feats)              # (E,31)

    return (new_concatenated_features.astype(np.float32),
            node_displacement,
            edge_index.astype(np.int64),
            edge_features.astype(np.float32))

# set random seed
torch.manual_seed(0)

# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

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
            EF = edge_features.astype(np.float32)    # (E,31)

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
            EF = f["edge_features"][:]               # (E, 31)
            sid = f.attrs["sample_id"]
            print(f"For sample {sid}:\n"
                  f"  new_concatenated_features[0] = {X[0]}\n"
                  f"  node_displacement[0] = {Y[0]}\n"
                  f"  edge_index[0] = {EI[0]}\n"
                  f"  edge_features[0] = {EF[0]}")
    else:
        print(r"U should choose save or check")

# Run it
features_h5_per_sample(dataset, OUT_DIR, save_or_check = "check")