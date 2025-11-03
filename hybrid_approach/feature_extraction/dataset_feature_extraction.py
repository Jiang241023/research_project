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
def load_element_features(h5_path, component="blank", timestep=-1, op_form=20):
    with h5py.File(h5_path, "r") as f:
        comp = f[f"OP{op_form}"][component]
        stress_t = comp["element_shell_stress"][timestep]  # (m,3,6)
        #print(f"the shape of the stress:{stress_t.shape}")
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
def load_quad_mesh(h5_path, component="blank", op_form=20, timestep = -1):
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
def load_displacement_op10(h5_path, operation = 20):
    _, displacement_vectors = extract_point_springback(h5_path, operation)  
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

def build_edge_edge_index(edge_index, num_nodes = None):
    """
    Build line-graph connectivity over edges: connect two *edges* if they share a node.
    Outputs a directed list (both directions) with dtype int64 and shape (M, 2).

    Args:
        edge_index : (E, 2) int64, undirected unique edges (i<j is fine).
        num_nodes  : optional; inferred from edge_index if None.

    Returns:
        edge_edge_index : (M, 2) int64 — pairs (e_i -> e_j) whenever edges e_i, e_j
                          share at least one endpoint. No self-loops. Duplicates removed.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    # For each node, collect incident edge IDs
    incident = [[] for _ in range(num_nodes)]
    for e_id, (u, v) in enumerate(edge_index):
        incident[u].append(e_id)
        incident[v].append(e_id)

    src, dst = [], []
    # For each node's incident edge list, fully connect the edges (both directions)
    for edges_at_node in incident:
        d = len(edges_at_node)
        if d < 2:
            continue
        # all unordered pairs, then add both directions
        for i in range(d - 1):
            ei = edges_at_node[i]
            for j in range(i + 1, d):
                ej = edges_at_node[j]
                if ei != ej:
                    src.append(ei); dst.append(ej)
                    src.append(ej); dst.append(ei)

    if not src:
        return np.empty((0, 2), dtype=np.int64)

    e2e = np.column_stack([np.asarray(src, dtype=np.int64),
                           np.asarray(dst, dtype=np.int64)])
    # Deduplicate (may occur if the same pair appears via two different nodes)
    e2e = np.unique(e2e, axis=0)
    return e2e

def prepare_sample(h5_path, component="blank", op_form=20, timestep=1):
    # 1) per-element features (m=11025) → per-triangle features (2m=22050)
    concatenated_features = load_element_features(h5_path, component, timestep, op_form)  # (11025, 31)
    print(f"the shape of concatenated_features:{concatenated_features.shape}")
    repeated_elem_feats = np.repeat(concatenated_features, 2, axis=0)                     # (22050, 31)

    # 2) mesh triangles and node coords
    node_coords, triangles = load_quad_mesh(h5_path, component, op_form, timestep)        # triangles: (22050, 3), coords: (N,3)
    print(f"the shape of node_coords:{node_coords.shape}")
    print(f"the shape of triangles:{triangles.shape}")

    # 3) node-level displacement (OP10)
    raw_displacement = load_displacement_op10(h5_path)                                     # (N, 3)
    print(f"the shape of raw_displacement:{raw_displacement.shape}")
    num_nodes = raw_displacement.shape[0]
    print(f"the num_nodes:{num_nodes}")
    node_index = np.arange(num_nodes, dtype=np.int64) 

    # 4) node features: avg triangle features → nodes, then concat with coords
    average_node_features = element_to_node_features(num_nodes, triangles, repeated_elem_feats)  # (N, 31)
    new_concatenated_features = np.concatenate([node_coords, average_node_features], axis=1)     # (N, 34)
    node_displacement = raw_displacement[:num_nodes].astype(np.float32)                          # (N, 3)

    # 5) EDGE INDEX + EDGE FEATURES (no degrees needed)
    edge_index, edge2tris = build_edges_from_triangles(triangles)                                  # (E,2), list-of-lists
    edge_features = compute_edge_features(edge_index, edge2tris, repeated_elem_feats)              # (E,31)
    edge_index_2 = np.arange(edge_index.shape[0], dtype=np.int64)          # (E,)
    edge_edge_index = build_edge_edge_index(edge_index)                         # (M,2)

    return (new_concatenated_features.astype(np.float32),
            node_displacement,
            edge_index.astype(np.int64),
            edge_features.astype(np.float32),
            node_coords,
            node_index,
            edge_index_2,
            edge_edge_index)

def npz_path(out_dir, sample_id):
    return out_dir / f"{sample_id}.npz"

def features_per_sample(ddacs, out_dir, action="save_npz"):
    out_dir.mkdir(parents=True, exist_ok=True)

    if action == "save_npz":
        exclude_ids = ["19044", "116133"]
        n = len(ddacs)
        t0 = time.perf_counter()
        for i in tqdm(range(n), desc="Saving features (NPZ bundle per sample)"):
            sample_id, _, h5_path = ddacs[i]  

            # skip unwanted ids
            if str(sample_id) in exclude_ids:         
                continue
              
            (X, Y, EI, EF,POS, NODE_IDX, EI_2, EEI) = prepare_sample(h5_path)

            # One NPZ per sample (keys match your previous filenames)
            np.savez(
            npz_path(out_dir, sample_id),
            new_concatenated_features=X,
            node_displacement=Y,
            edge_index=EI,
            edge_features=EF,
            node_coords=POS,
            node_index=NODE_IDX,
            edge_index_2=EI_2,
            edge_edge_index=EEI,
            )
        total_time = time.perf_counter() - t0
        print("\n=== Save summary ===")
        print(f"dir: {out_dir}")
        print("format: NPZ (one bundle per sample)")
        print(f"total time: {total_time:.2f} s")


    elif action == "check_npz":
        sample_to_check = "16039" # change to any ID you saved
        npz_file = npz_path(out_dir, sample_to_check)
        if not npz_file.exists():
            raise FileNotFoundError(f"{npz_file} not found; run with action='save_npz' first.")
        with np.load(npz_file) as z:
            X = z["new_concatenated_features"]
            Y = z["node_displacement"]
            EI = z["edge_index"]
            EF = z["edge_features"]
            POS = z["node_coords"]
            NODE_IDX = z["node_index"]
            EI_2 = z["edge_index_2"]
            EEI = z["edge_edge_index"]
            print("-----------------------------------")
            print(f"For sample {sample_to_check} (NPZ bundle):")
            print("-----------------------------------")
            print(f"new_concatenated_features shape: {X.shape}; new_concatenated_features[0]:\n{X[0]}")
            print(f"node_displacement shape: {Y.shape}; node_displacement[0]:\n{Y[0]}")
            print(f"edge_index shape: {EI.shape}; edge_index[0]: {EI[0]}")
            print(f"edge_features shape: {EF.shape}; edge_features[0]:\n{EF[0]}")
            print(f"node_coords shape: {POS.shape}; node_coords[0]:\n{POS[0]}")
            print(f"node_index shape: {NODE_IDX.shape}; node_index[0]: {NODE_IDX[0]}")
            print(f"edge_index_2 shape: {EI_2.shape}; edge_index_2[0]: {EI_2[0]}")
            print(f"edge_edge_index shape: {EEI.shape}; edge_edge_index[0]: {EEI[0]}")
    else:
        print("action must be one of: 'save_npz', 'check_npz'")

if __name__ == '__main__':
    # Setup data directory
    data_dir = Path("/mnt/data/darus/")

    # Load dataset
    dataset = DDACSDataset(data_dir, "h5")
    print(f"Loaded {len(dataset)} simulations")

    # Add the save path
    OUT_DIR = Path("/mnt/data/jiang/op20")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Run it
    features_per_sample(dataset, OUT_DIR, action="save_npz")