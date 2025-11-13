import os
import re
import numpy as np
from pathlib import Path
from DDACSDataset import DDACSDataset
from utils_DDACS import extract_mesh, extract_point_springback
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt

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
       KDTree:"k-dimensional tree.” 
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

# Scan prediction files
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

# For storing all samples
def summarize(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(values.mean()), float(values.max()), float(values.std()))

# Main 
if __name__ == "__main__":
    # Config 
    operation   = 20    # 10 or 20
    timestep    = 0     # 2 or 0
    # pred_dir    = "/home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/grit_like"
    pred_dir    = "/home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/grit_like_op20_grit_like_fullsamples_15epoch_alpha1_beta1_withlap"
    data_dir    = Path("/mnt/data/darus/")
    experiment_name = "op20_grit_like_fullsamples_15epoch_alpha1_beta1_withlap"

    # save a one-row CSV with totals
    WRITE_CSV   = True
    WRITE_SAMPLES_CSV   = True
    MAKE_BOXPLOTS = True
    if operation == 10 and timestep == 2:
        save_dir    = Path("/home/RUS_CIP/st186731/research_project/hybrid_approach/evaluation_output/op10")
        save_dir.mkdir(parents=True, exist_ok=True)
    elif operation == 20 and timestep == 0:
        save_dir    = Path("/home/RUS_CIP/st186731/research_project/hybrid_approach/evaluation_output/op20")
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("please check the operation and timestep")
        
    totals_csv_path = save_dir / f"{experiment_name}_dataset_totals.csv"
    samples_csv_path = save_dir / f"{experiment_name}_per_sample.csv"

    # Load dataset index
    dataset = DDACSDataset(data_dir, "h5")
    pairs = scan_prediction_files(pred_dir)
    if not pairs:
        raise SystemExit(f"No *_pred_node_displacement.npy files in: {pred_dir}")

    # Pooled stats across ALL nodes in ALL samples
    gt_stats   = RunningStats()
    pred_stats = RunningStats()
    diff_stats = RunningStats()
    cham_gt_pred  = RunningStats()
    cham_pred_gt   = RunningStats()
    processed   = 0
    skipped     = 0

    sample_rows = []
    for k, (sid, path) in enumerate(pairs, 1):
        try:
            # Locate H5
            sim_id, metadata, h5_path = find_h5_by_id(dataset, sid)

            # Mesh + ground truth
            node_coords, triangles = extract_mesh(
                h5_path, operation=operation, component='blank', timestep=timestep
            )
            final_coords_gt, disp_gt = extract_point_springback(h5_path, operation=operation)

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
            distances_gt_pred = nearest_neighbor_distances(final_coords_gt,  final_coords_pred)  # GT→Pred
            distances_pred_gt = nearest_neighbor_distances(final_coords_pred, final_coords_gt)  # Pred→GT

            # Update pooled stats
            gt_stats.update(mag_gt)
            pred_stats.update(mag_pred)
            diff_stats.update(diff_mag)
            cham_gt_pred.update(distances_gt_pred)
            cham_pred_gt.update(distances_pred_gt)

            gt_mean_summarize, gt_max_summarize, gt_std_summarize = summarize(mag_gt)
            pred_mean_summarize, pred_max_summarize, pred_std_summarize = summarize(mag_pred)
            diff_mean_summarize, diff_max_summarize, diff_std_summarize = summarize(diff_mag)
            cham_gt_pred_mean_summarize, cham_gt_pred_max_summarize,cham_gt_pred_std_summarize = summarize(distances_gt_pred)
            cham_pred_gt_mean_summarize, cham_pred_gt_max_summarize, cham_pred_gt_std_summarize = summarize(distances_pred_gt)
            cham_sym = cham_gt_pred_mean_summarize + cham_pred_gt_mean_summarize  

            sample_rows.append([
                int(sid),
                gt_mean_summarize, gt_max_summarize, gt_std_summarize,
                pred_mean_summarize, pred_max_summarize, pred_std_summarize,
                diff_mean_summarize, diff_max_summarize, diff_std_summarize,
                cham_gt_pred_mean_summarize, cham_gt_pred_max_summarize,cham_gt_pred_std_summarize,
                cham_pred_gt_mean_summarize, cham_pred_gt_max_summarize, cham_pred_gt_std_summarize,
                cham_sym
                ])

            processed   += 1

            # Optional progress
            if (k % 10 == 0) or (k == len(pairs)):
                print(f"[{k}/{len(pairs)}] processed")
        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping id={sid}: {e}")

    # Dataset-level (pooled) totals
    chamfer_symmetric_mean = cham_gt_pred.mean + cham_pred_gt.mean

    print("\n--- Springback stats (TOTAL across all nodes) ---")
    print(f"Ground_Truth : mean={gt_stats.mean:.4f},  max={gt_stats.max:.4f},  std={gt_stats.std:.4f}")
    print(f"Prediction   : mean={pred_stats.mean:.4f}, max={pred_stats.max:.4f}, std={pred_stats.std:.4f}")
    print(f"Difference(L2 per-node) : mean={diff_stats.mean:.4f},  max={diff_stats.max:.4f},  std={diff_stats.std:.4f}")

    print("\n--- Chamfer stats (L2, TOTAL) ---")
    print(f"(GT→Pred): mean={cham_gt_pred.mean:.6f}, max={cham_gt_pred.max:.6f}, std={cham_gt_pred.std:.6f}")
    print(f"(Pred→GT): mean={cham_pred_gt.mean:.6f}, max={cham_pred_gt.max:.6f}, std={cham_pred_gt.std:.6f}")
    print(f"Symmetric        : {chamfer_symmetric_mean:.6f}")
    print(f"\nProcessed {processed}/{len(pairs)} files")

    if WRITE_CSV:
        headers = [
            "gt_mean","gt_max","gt_std",
            "pred_mean","pred_max","pred_std",
            "diff_mean","diff_max","diff_std",
            "chamfer_distance_gt_pred_mean","chamfer_distance_gt_pred_max","chamfer_distance_gt_pred_std",
            "chamfer_distance_pred_gt_mean","chamfer_distance_pred_gt_max","chamfer_distance_pred_gt_std",
            "chamfer_distance_symmetric","num_files","skipped"
        ]
        values = [
            gt_stats.mean, gt_stats.max, gt_stats.std,
            pred_stats.mean, pred_stats.max, pred_stats.std,
            diff_stats.mean, diff_stats.max, diff_stats.std,
            cham_gt_pred.mean, cham_gt_pred.max, cham_gt_pred.std,
            cham_pred_gt.mean, cham_pred_gt.max, cham_pred_gt.std,
            chamfer_symmetric_mean, processed, skipped
        ]
        with open(totals_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            parts = []
            for v in values:
                if isinstance(v, (float, np.floating)):
                    s = f"{v:.10f}"
                else:
                    s = str(v)
                parts.append(s)
            f.write(",".join(parts) + "\n")
        print(f"[OK] Wrote totals CSV → {totals_csv_path}")

    if WRITE_SAMPLES_CSV and len(sample_rows) > 0:
        sample_headers = [
            "sample_id",
            "gt_mean","gt_max","gt_std",
            "pred_mean","pred_max","pred_std",
            "diff_mean","diff_max","diff_std",
            "chamfer_distance_gt_pred_mean","chamfer_distance_gt_pred_max","chamfer_distance_gt_pred_std",
            "chamfer_distance_pred_gt_mean","chamfer_distance_pred_gt_max","chamfer_distance_pred_gt_std",
            "chamfer_distance_symmetric"
        ]
        with open(samples_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(sample_headers) + "\n")
            for row in sample_rows:
                out = []
                for v in row:
                    if isinstance(v, (float, np.floating)):
                        out.append(f"{v:.10f}")
                    else:
                        out.append(str(v))
                f.write(",".join(out) + "\n")
        print(f"[OK] Wrote per-sample CSV → {samples_csv_path}")
    
    if MAKE_BOXPLOTS and len(sample_rows) > 0:
        # transpose rows -> columns for easy slicing
        cols = list(zip(*sample_rows))
        # columns follow sample_headers order:
        # 0: sample_id
        gt_mean_array   = np.asarray(cols[1],  dtype=float)
        gt_max_array    = np.asarray(cols[2],  dtype=float)
        gt_std_array    = np.asarray(cols[3],  dtype=float)
        pred_mean_array = np.asarray(cols[4],  dtype=float)
        pred_max_array  = np.asarray(cols[5],  dtype=float)
        pred_std_array  = np.asarray(cols[6],  dtype=float)
        diff_mean_array = np.asarray(cols[7],  dtype=float)
        diff_max_array  = np.asarray(cols[8],  dtype=float)
        diff_std_array  = np.asarray(cols[9],  dtype=float)
        cham_gt_pred_mean_array = np.asarray(cols[10], dtype=float)
        cham_gt_pred_max_array  = np.asarray(cols[11], dtype=float)
        cham_gt_pred_std_array  = np.asarray(cols[12], dtype=float)
        cham_pred_gt_mean_array = np.asarray(cols[13], dtype=float)
        cham_pred_gt_max_array  = np.asarray(cols[14], dtype=float)
        cham_pred_gt_std_array  = np.asarray(cols[15], dtype=float)
        cham_sym_array          = np.asarray(cols[16], dtype=float)

        def make_boxplot(data_dict, title, outfile, ylabel=None, showfliers=False):
            fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
            labels = list(data_dict.keys())
            data   = [np.asarray(v, dtype=float) for v in data_dict.values()]
            ax.boxplot(
                data,
                labels=labels,
                showmeans=False,
                meanline=True,
                vert=True,
                patch_artist=False,
                showfliers=showfliers
            )
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)
            ax.set_title(title)
            if ylabel:
                ax.set_ylabel(ylabel)
            fig.tight_layout()
            out_path = save_dir / outfile
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print(f"[OK] Saved box plot → {out_path}")

        make_boxplot(
            {
                "GT mean": gt_mean_array,
                "Pred mean": pred_mean_array,
                "Diff mean": diff_mean_array
            },
            f"{experiment_name}: Springback magnitude (per-sample mean)",
            f"{experiment_name}_boxplot_magnitude_mean.png",
            ylabel="Displacement"
        )

# Example runs:
#python /home/RUS_CIP/st186731/research_project/hybrid_approach/evaluation/evaluation.py