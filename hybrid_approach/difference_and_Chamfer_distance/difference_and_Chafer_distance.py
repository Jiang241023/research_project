import os
import re
import numpy as np
from pathlib import Path
from DDACSDataset import DDACSDataset
from utils_DDACS import extract_mesh, extract_point_springback
from scipy.spatial import cKDTree as KDTree


# Config 
OPERATION   = 10
TIMESTEP    = 2
pred_dir    = "/home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression-fullsamples-10epochs-alpha0.8-beta0.2-grit_likewithlap/grit_like"
data_dir    = Path("/mnt/ac142464/data/darus/")

# save a one-row CSV with totals
WRITE_CSV   = True
save_dir    = Path("/home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/difference_and_Chamfer_distance_output")
save_dir.mkdir(parents=True, exist_ok=True)
totals_csv_path = save_dir / "dataset_totals_grit_like_fullsamples_10e_alpha0.8_beta0.2_withlap.csv"

#  Utils 
def find_h5_by_id(dataset, sid):
    """Return (sim_id, metadata, h5_path) for the given string/int sample id."""
    sid = str(sid)
    for i in range(len(dataset)):
        sim_id, meta, h5_path = dataset[i]
        if str(sim_id) == sid:
            return sim_id, meta, h5_path
    raise FileNotFoundError(f"Sample id {sid} not found in dataset")

def nearest_neighbor_distances(query_points, reference_points):
    """Per-point NN distances from query_points to reference_points (L2).
       KDTree:"k-dimensional tree.” It's a space-partitioning data structure that indexes points in ℝᵈ so you can answer nearest-neighbor queries much faster than a brute-force O(M·N) scan.
    """
    tree = KDTree(reference_points)
    distances, _ = tree.query(query_points, k=1, workers=-1)
    return distances.astype(np.float64)

# Streaming pooled stats (mean/std/max) over arbitrary many arrays
class RunningStats:
    """Streaming mean/std/max over arbitrarily many 1D/ND arrays."""
    def __init__(self):
        self.total_count = 0
        self.running_sum = 0.0
        self.running_sum_of_squares = 0.0
        self.running_max = -np.inf

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).ravel() # .ravel() returns a 1-D view of the array when possible; otherwise a copy.
        if values.size == 0:
            return
        self.total_count += values.size
        self.running_sum += float(values.sum())
        self.running_sum_of_squares += float((values ** 2).sum())
        current_batch_max = float(values.max())
        if current_batch_max > self.running_max:
            self.running_max = current_batch_max

    @property
    def mean(self):
        count = self.total_count
        if count <= 0:
            return float("nan")
        return self.running_sum / count

    @property
    def std(self):
        if self.total_count == 0:
            return float("nan")
        current_mean = self.mean
        current_variance = max(0.0, self.running_sum_of_squares / self.total_count - current_mean ** 2)
        return current_variance ** 0.5

    @property
    def max(self):
        return self.running_max

# scan prediction files
def scan_prediction_files(pred_dir):
    """Return list of (sample_id, full_path) for *_pred_node_displacement.npy files."""
    pattern = re.compile(r"^(\d+)_pred_node_displacement\.npy$")
    out = []
    for name in os.listdir(pred_dir):
        match = pattern.match(name)
        if match:
            out.append((match.group(1), os.path.join(pred_dir, name)))
    out.sort(key=lambda t: int(t[0])) # Sorts the list numerically by sample_id. t[0] is the sample_id string; int(t[0]) ensures 2, 10, 100 order
    return out

# Main 
if __name__ == "__main__":
    # Load dataset index
    dataset = DDACSDataset(data_dir, "h5")
    pairs = scan_prediction_files(pred_dir)
    if not pairs:
        raise SystemExit(f"No *_pred_node_displacement.npy files in: {pred_dir}")

    # Pooled stats across ALL nodes in ALL samples
    gt_stats   = RunningStats()
    pred_stats = RunningStats()
    diff_stats = RunningStats()
    cham_fwd   = RunningStats()
    cham_bwd   = RunningStats()
    processed   = 0
    skipped     = 0

    for k, (sid, path) in enumerate(pairs, 1):
        try:
            # Locate H5
            sim_id, metadata, h5_path = find_h5_by_id(dataset, sid)

            # Mesh + ground truth
            node_coords, triangles = extract_mesh(
                h5_path, operation=OPERATION, component='blank', timestep=TIMESTEP
            )
            final_coords_gt, disp_gt = extract_point_springback(h5_path, operation=OPERATION)

            # Prediction
            disp_pred = np.load(path)
            if disp_pred.shape != disp_gt.shape:
                raise ValueError(f"shape mismatch for id={sid}: pred {disp_pred.shape} vs gt {disp_gt.shape}")

            # Magnitudes and differences
            mag_gt   = np.linalg.norm(disp_gt,   axis=1)
            mag_pred = np.linalg.norm(disp_pred, axis=1)
            diff_mag = np.linalg.norm(disp_pred - disp_gt, axis=1)

            # Chamfer on final positions
            final_coords_pred = node_coords + disp_pred
            
            # nearest-neighbor distances from GT final coords to Pred final coords and nearest-neighbor distances from Pred final coords to GT final coords
            distances_fwd = nearest_neighbor_distances(final_coords_gt,  final_coords_pred)  # GT→Pred
            distances_bwd = nearest_neighbor_distances(final_coords_pred, final_coords_gt)  # Pred→GT

            # Update pooled stats
            gt_stats.update(mag_gt)
            pred_stats.update(mag_pred)
            diff_stats.update(diff_mag)
            cham_fwd.update(distances_fwd)
            cham_bwd.update(distances_bwd)

            processed   += 1

            # Optional progress
            if (k % 10 == 0) or (k == len(pairs)):
                print(f"[{k}/{len(pairs)}] processed")
        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping id={sid}: {e}")

    # Dataset-level (pooled) totals
    chamfer_symmetric_mean = cham_fwd.mean + cham_bwd.mean

    print("\n--- Springback stats (TOTAL across all nodes) ---")
    print(f"Ground_Truth : mean={gt_stats.mean:.4f},  max={gt_stats.max:.4f},  std={gt_stats.std:.4f}")
    print(f"Prediction   : mean={pred_stats.mean:.4f}, max={pred_stats.max:.4f}, std={pred_stats.std:.4f}")
    print(f"Difference(L2 per-node) : mean={diff_stats.mean:.4f},  max={diff_stats.max:.4f},  std={diff_stats.std:.4f}")

    print("\n--- Chamfer stats (L2, TOTAL) ---")
    print(f"Forward (GT→Pred): mean={cham_fwd.mean:.6f}, max={cham_fwd.max:.6f}, std={cham_fwd.std:.6f}")
    print(f"Backward(Pred→GT): mean={cham_bwd.mean:.6f}, max={cham_bwd.max:.6f}, std={cham_bwd.std:.6f}")
    print(f"Symmetric        : {chamfer_symmetric_mean:.6f}")
    print(f"\nProcessed {processed}/{len(pairs)} files")

    if WRITE_CSV:
        headers = [
            "gt_mean","gt_max","gt_std",
            "pred_mean","pred_max","pred_std",
            "diff_mean","diff_max","diff_std",
            "chamfer_distance_fwd_mean","chamfer_distance_fwd_max","chamfer_distance_fwd_std",
            "chamfer_distance_bwd_mean","chamfer_distance_bwd_max","chamfer_distance_bwd_std",
            "chamfer_distance_symmetric","num_files","skipped"
        ]
        values = [
            gt_stats.mean, gt_stats.max, gt_stats.std,
            pred_stats.mean, pred_stats.max, pred_stats.std,
            diff_stats.mean, diff_stats.max, diff_stats.std,
            cham_fwd.mean, cham_fwd.max, cham_fwd.std,
            cham_bwd.mean, cham_bwd.max, cham_bwd.std,
            chamfer_symmetric_mean, processed, skipped
        ]
        with open(totals_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            f.write(",".join(f"{v:.10f}" if isinstance(v, float) else str(v) for v in values) + "\n")
        print(f"[OK] Wrote totals CSV → {totals_csv_path}")

# Example runs:
#python /home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/difference_and_Chamfer_distance/difference_and_CHamfer_distance.py