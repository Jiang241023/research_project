import os, sys
import numpy as np
import h5py
import torch
from pathlib import Path
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import csv

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from DDACSDataset import DDACSDataset
from utils.utils_DDACS import (
    extract_mesh,
    extract_element_thickness,
    extract_point_springback,
)

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
torch.manual_seed(0)

# ---------------------------------------------------------------------
# Data root & base dataset
# ---------------------------------------------------------------------
data_dir = Path("/mnt/data/darus/")
dataset = DDACSDataset(data_dir, "h5")
N_TOTAL = len(dataset)
print(f"Loaded {N_TOTAL} simulations")

# ---------------------------------------------------------------------
# Timing helpers (no shadowing)
# ---------------------------------------------------------------------
import time as _time

def now():
    return _time.perf_counter()

class SectionTimer:
    def __init__(self):
        self.t = defaultdict(float)
    def add(self, key, dt):
        self.t[key] += dt
    def to_ordered(self):
        # Stable, nice order for CSV
        keys = [
            # IO first
            "io:read_strain",
            "io:extract_thickness",
            "io:extract_mesh",
            "io:extract_springback",
            # compute next
            "compute:reshape_strain",
            "compute:concat_elem_feats",
            "compute:repeat_elem_feats",
            "compute:astype_mesh",
            "compute:astype_disp",
            "compute:project_elem_to_nodes",
            "compute:concat_coords_features",
            # totals (per sub-function)
            "total:load_element_features",
            "total:load_quad_mesh",
            "total:load_displacement",
            # end-to-end optional
            "total:prepare_sample",
        ]
        od = OrderedDict()
        for k in keys:
            if k in self.t:
                od[k] = self.t[k]
        # include any extra, unexpected keys
        for k in sorted(self.t.keys()):
            if k not in od:
                od[k] = self.t[k]
        return od

# ---------------------------------------------------------------------
# Feature loaders (timed variants)
# ---------------------------------------------------------------------
def load_element_features_timed(h5_path, component="blank", timestep=3, op_form=10, t:SectionTimer=None):
    t0 = now()
    with h5py.File(h5_path, "r") as f:
        comp = f[f"OP{op_form}"][component]
        t1 = now()
        strain_t = comp["element_shell_strain"][timestep]  # (m, 2, 6)
        t2 = now()
    if t: t.add("io:read_strain", t2 - t1)

    t3 = now()
    strain_features = strain_t.reshape(strain_t.shape[0], -1).astype(np.float32)  # (m, 12)
    t4 = now()
    if t: t.add("compute:reshape_strain", t4 - t3)

    t5 = now()
    thickness = extract_element_thickness(h5_path, timestep=timestep, operation=op_form).astype(np.float32)
    t6 = now()
    if t: t.add("io:extract_thickness", t6 - t5)

    t7 = now()
    concatenated = np.concatenate([strain_features, thickness[:, None]], axis=1)  # (m, 13)
    t8 = now()
    if t: t.add("compute:concat_elem_feats", t8 - t7)
    if t: t.add("total:load_element_features", t8 - t0)
    return concatenated

def load_quad_mesh_timed(h5_path, component="blank", op_form=10, timestep=3, t:SectionTimer=None):
    t0 = now()
    node_coords, triangles = extract_mesh(h5_path, operation=op_form, component=component, timestep=timestep)
    t1 = now()
    if t: t.add("io:extract_mesh", t1 - t0)

    t2 = now()
    node_coords = node_coords.astype(np.float32)
    triangles = triangles.astype(np.int64)
    t3 = now()
    if t: t.add("compute:astype_mesh", t3 - t2)
    if t: t.add("total:load_quad_mesh", t3 - t0)
    return node_coords, triangles

def load_displacement_op10_timed(h5_path, t:SectionTimer=None):
    t0 = now()
    _, disp = extract_point_springback(h5_path, operation=10)
    t1 = now()
    if t: t.add("io:extract_springback", t1 - t0)

    t2 = now()
    disp = disp.astype(np.float32)
    t3 = now()
    if t: t.add("compute:astype_disp", t3 - t2)
    if t: t.add("total:load_displacement", t3 - t1 + (t1 - t0))  # equal to (t3 - t0)
    return disp

# ---------------------------------------------------------------------
# Element→node projection (timed)
# ---------------------------------------------------------------------
def element_to_node_features_timed(num_nodes, triangles, elem_features, t:SectionTimer=None):
    t0 = now()
    F = elem_features.shape[1]
    node_feature_sums = np.zeros((num_nodes, F), dtype=np.float32)
    node_counts = np.zeros(num_nodes, dtype=np.int32)
    for tri_idx in range(len(triangles)):
        feat = elem_features[tri_idx]        # (F,)
        tri_nodes = triangles[tri_idx]       # (3,)
        for n in tri_nodes:
            node_feature_sums[n] += feat
            node_counts[n] += 1
    avg = node_feature_sums / np.maximum(node_counts[:, None], 1)
    t1 = now()
    if t: t.add("compute:project_elem_to_nodes", t1 - t0)
    return avg

# ---------------------------------------------------------------------
# Timed sample preparation
# ---------------------------------------------------------------------
def prepare_sample_timed(h5_path, component="blank", op_form=10, timestep=3):
    T = SectionTimer()
    t_start = now()

    elem_feats = load_element_features_timed(h5_path, component, timestep, op_form, T)  # (m, 13)

    t0 = now()
    repeated_elem_feats = np.repeat(elem_feats, 2, axis=0)                               # (~2m, 13)
    t1 = now()
    T.add("compute:repeat_elem_feats", t1 - t0)

    node_coords, triangles = load_quad_mesh_timed(h5_path, component, op_form, timestep, T)
    raw_disp = load_displacement_op10_timed(h5_path, T)
    N = raw_disp.shape[0]

    avg_node_feats = element_to_node_features_timed(N, triangles, repeated_elem_feats, T)

    t2 = now()
    x = np.concatenate([node_coords.astype(np.float32), avg_node_feats], axis=1)  # [N, 16]
    y = raw_disp[:N]                                                              # [N, 3]
    t3 = now()
    T.add("compute:concat_coords_features", t3 - t2)

    T.add("total:prepare_sample", now() - t_start)
    return x, y, T

# ---------------------------------------------------------------------
# Iterate WHOLE dataset, save CSV, print aggregates
# ---------------------------------------------------------------------
def run_full_benchmark(ddacs: DDACSDataset, csv_path="timings_full_dataset.csv", warmup=3):
    """
    ddacs: the base DDACSDataset (iterable, yields (sim_id, meta, h5_path))
    warmup: number of initial samples to ignore in aggregation (to reduce cache/first-open effects)
    """
    # Prepare CSV header
    header = [
        "index", "sim_id", "num_nodes", "num_elems_tri", "X_shape0", "X_shape1",
        # section keys will be appended dynamically
    ]
    # We'll create a union of keys encountered to keep columns consistent
    union_keys = OrderedDict()

    # First pass to collect keys AND write lines progressively
    rows = []
    errors = 0

    for i in tqdm(range(len(ddacs)), desc="Timing all sims"):
        try:
            sim_id, _, h5_path = ddacs[i]
            # Run timed prep
            x, y, T = prepare_sample_timed(h5_path)
            N = x.shape[0]
            # triangles count equals ~2*m; but we don't have m here without re-opening; approximate via projection inputs:
            # We'll infer triangle count from degree averages later if needed. For now omit or put -1.
            row = {
                "index": i,
                "sim_id": sim_id,
                "num_nodes": int(N),
                "num_elems_tri": -1,
                "X_shape0": int(x.shape[0]),
                "X_shape1": int(x.shape[1]),
            }
            od = T.to_ordered()
            for k, v in od.items():
                row[k] = v
                union_keys[k] = True
            rows.append(row)
        except Exception as e:
            errors += 1
            # Keep going; log minimal info
            rows.append({
                "index": i, "sim_id": None, "num_nodes": -1, "num_elems_tri": -1,
                "X_shape0": -1, "X_shape1": -1, "error": str(e)
            })
            union_keys["error"] = True
            continue

    # Compose final header with sections
    header_extended = header + list(union_keys.keys())

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_extended)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregates (skip warmup and errors)
    agg = defaultdict(float)
    count_ok = 0
    for r in rows[warmup:]:
        if "error" in r and r["error"]:
            continue
        for k in union_keys.keys():
            if k.startswith("io:") or k.startswith("compute:") or k.startswith("total:"):
                v = r.get(k, 0.0)
                if isinstance(v, (int, float)):
                    agg[k] += float(v)
        count_ok += 1

    print(f"\nFinished. CSV saved to: {csv_path}")
    print(f"Successful samples: {count_ok}/{len(ddacs)}; errors: {errors}")

    if count_ok == 0:
        return

    avg = {k: agg[k] / count_ok for k in agg.keys()}
    io_total_avg = sum(v for k, v in avg.items() if k.startswith("io:"))
    compute_total_avg = sum(v for k, v in avg.items() if k.startswith("compute:"))
    end_to_end_avg = avg.get("total:prepare_sample", io_total_avg + compute_total_avg)

    print("\n=== Averages over full dataset (excluding warmup & errors) ===")
    print(f"io_total_avg:       {io_total_avg*1000:8.2f} ms  ({io_total_avg/(end_to_end_avg+1e-12):6.2%} of total)")
    print(f"compute_total_avg:  {compute_total_avg*1000:8.2f} ms  ({compute_total_avg/(end_to_end_avg+1e-12):6.2%} of total)")
    print(f"end_to_end_avg:     {end_to_end_avg*1000:8.2f} ms")

    print("\n--- Top section averages (ms) ---")
    for k in sorted(avg.keys()):
        print(f"{k:35s} {avg[k]*1000:8.2f}")

# ---------------------------------------------------------------------
# Run full benchmark
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: quick warm cache
    try:
        for j in range(3):
            _, _, p = dataset[j]
            _ = prepare_sample_timed(p)
    except Exception:
        pass

    run_full_benchmark(dataset, csv_path="timings_full_dataset.csv", warmup=3)
