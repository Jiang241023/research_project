import os, re  
import numpy as np
from pathlib import Path
from DDACSDataset import DDACSDataset
from utils_DDACS import extract_mesh, extract_point_springback
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt

# utils
def find_h5_by_id(dataset, sid):
    sid = str(sid)
    for i in range(len(dataset)):
        sim_id, meta, h5_path = dataset[i]
        if str(sim_id) == sid:
            return sim_id, meta, h5_path
    raise FileNotFoundError(f"Sample id {sid} not found in dataset")

# For Chamfer distance
def nearest_neighbor_distances(query_points, reference_points):
    tree = KDTree(reference_points)
    distances, _ = tree.query(query_points, k=1, workers=-1)
    return distances.astype(np.float64)

class RunningStats:
    def __init__(self):
        self.total_count = 0
        self.running_sum = 0.0
        self.running_sum_of_squares = 0.0
        self.running_max = -np.inf
    def update(self, values):
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        self.total_count += values.size
        self.running_sum += float(values.sum())
        self.running_sum_of_squares += float((values ** 2).sum())
        self.running_max = max(self.running_max, float(values.max()))

    @property
    def mean(self):
        if self.total_count <= 0:
            out = float("nan")
        else:
            out = self.running_sum / self.total_count
        return out

    @property
    def std(self):
        if self.total_count == 0:
            return float("nan")
        m = self.mean
        var = max(0.0, self.running_sum_of_squares / self.total_count - m ** 2)
        return var ** 0.5

    @property
    def max(self):
        return self.running_max

def scan_prediction_files(pred_dir):
    pat = re.compile(r"^(\d+)_pred_node_displacement\.npy$")
    out = {}
    for name in os.listdir(pred_dir):
        m = pat.match(name)
        if m:
            sid = m.group(1)
            out[sid] = os.path.join(pred_dir, name)
    return out  
def summarize(values):
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(v.mean()), float(v.max()), float(v.std()))

# main (plot per-node displacement magnitudes across all samples)
if __name__ == "__main__":
    # Config
    operation = 10  # 10 or 20
    timestep  = 2   # 2 or 0

    #  set two experiment folders
    pred_dir_1 = "/home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/grit_like-fullsamples-10epochs-alpha0.8-beta0.2-grit_likewithlap"
    pred_dir_2 = "/home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/graphormer_like-fullsamples-10epochs-alpha0.8-beta0.2-graphormer_likewithlap"

    data_dir   = Path("/mnt/data/darus/")
    experiment_name = "compare_grit_like_vs_graphormer_like_op10_1"

    MAKE_BOXPLOTS      = True
    WRITE_SAMPLES_CSV  = True
    WRITE_TOTALS_CSV   = True

    if operation == 10 and timestep == 2:
        save_dir = Path("/home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/Evaluation_output/op10")
    elif operation == 20 and timestep == 0:
        save_dir = Path("/home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/Evaluation_output/op20")
    else:
        raise ValueError("please check the operation and timestep")
    save_dir.mkdir(parents=True, exist_ok=True)

    samples_csv_path = save_dir / f"{experiment_name}_per_sample.csv"
    totals_csv_path  = save_dir / f"{experiment_name}_dataset_totals.csv"

    # Load dataset & predictions
    dataset = DDACSDataset(data_dir, "h5")
    preds1  = scan_prediction_files(pred_dir_1)   # sid -> file
    preds2  = scan_prediction_files(pred_dir_2)   # sid -> file

    # Work on the intersection of sample IDs present in both experiments
    common_ids = sorted(set(preds1.keys()).intersection(preds2.keys()), key=int)
    if not common_ids:
        raise SystemExit("No overlapping prediction files between the two experiments.")

    # Stats (pooled across all nodes of all samples)
    gt_stats      = RunningStats()
    pred1_stats   = RunningStats()
    pred2_stats   = RunningStats()
    diff1_stats   = RunningStats()
    diff2_stats   = RunningStats()

    sample_rows = []  # one row per sample

    # collect all per-node magnitudes for boxplot
    all_mag_gt      = []
    all_mag_pred1   = []
    all_mag_pred2   = []
    all_difference1 = []
    all_difference2 = []

    for k, sid in enumerate(common_ids, 1):
        try:
            sim_id, meta, h5_path = find_h5_by_id(dataset, sid)

            # Mesh + GT
            node_coords, triangles = extract_mesh(
                h5_path, operation=operation,
                component='blank', timestep=timestep
            )
            final_coords_gt, displacement_gt = extract_point_springback(
                h5_path, operation=operation
            )

            # Predictions
            displacement_pred1 = np.load(preds1[sid])
            displacement_pred2 = np.load(preds2[sid])

            if displacement_pred1.shape != displacement_gt.shape:
                raise ValueError(f"[{sid}] exp1 shape {displacement_pred1.shape} != gt {displacement_gt.shape}")
            if displacement_pred2.shape != displacement_gt.shape:
                raise ValueError(f"[{sid}] exp2 shape {displacement_pred2.shape} != gt {displacement_gt.shape}")

            # Magnitudes
            mag_gt     = np.linalg.norm(displacement_gt, axis=1)
            mag_pred1  = np.linalg.norm(displacement_pred1, axis=1)
            mag_pred2  = np.linalg.norm(displacement_pred2, axis=1)

            # Per-node absolute differences vs GT
            difference_1 = np.linalg.norm(displacement_pred1 - displacement_gt, axis=1)
            difference_2 = np.linalg.norm(displacement_pred2 - displacement_gt, axis=1)

            # Chamfer distances (symmetric) for both models
            final_coords_pred1 = node_coords + displacement_pred1
            final_coords_pred2 = node_coords + displacement_pred2

            # GT ↔ Pred1
            distances1_gt_pred = nearest_neighbor_distances(final_coords_gt,  final_coords_pred1)  # GT→Pred1
            distances1_pred_gt = nearest_neighbor_distances(final_coords_pred1, final_coords_gt)   # Pred1→GT
            chamfer_sym_pred1  = float(distances1_gt_pred.mean() + distances1_pred_gt.mean())

            # GT ↔ Pred2
            distances2_gt_pred = nearest_neighbor_distances(final_coords_gt,  final_coords_pred2)  # GT→Pred2
            distances2_pred_gt = nearest_neighbor_distances(final_coords_pred2, final_coords_gt)   # Pred2→GT
            chamfer_sym_pred2  = float(distances2_gt_pred.mean() + distances2_pred_gt.mean())

            # accumulate per-node magnitudes for global boxplots
            all_mag_gt.append(mag_gt)
            all_mag_pred1.append(mag_pred1)
            all_mag_pred2.append(mag_pred2)
            all_difference1.append(difference_1)
            all_difference2.append(difference_2)

            # streaming stats across all samples
            gt_stats.update(mag_gt)
            pred1_stats.update(mag_pred1)
            pred2_stats.update(mag_pred2)
            diff1_stats.update(difference_1)
            diff2_stats.update(difference_2)

            # Per-sample summaries (for CSV only)
            gt_mean_summarize,    gt_max_summarize,    gt_std_summarize      = summarize(mag_gt)
            pred1_mean_summarize, pred1_max_summarize, pred1_std_summarize   = summarize(mag_pred1)
            pred2_mean_summarize, pred2_max_summarize, pred2_std_summarize   = summarize(mag_pred2)

            # Store per-sample row
            sample_rows.append([
                int(sid),
                gt_mean_summarize,    gt_max_summarize,    gt_std_summarize,
                pred1_mean_summarize, pred1_max_summarize, pred1_std_summarize,
                pred2_mean_summarize, pred2_max_summarize, pred2_std_summarize,
                chamfer_sym_pred1, chamfer_sym_pred2
            ])

            if (k % 10 == 0) or (k == len(common_ids)):
                print(f"[{k}/{len(common_ids)}] processed")

        except Exception as e:
            print(f"[WARN] Skipping id={sid}: {e}")

    # CSV with per-sample numbers + store column arrays and global means
    if sample_rows:
        # per-sample CSV 
        if WRITE_SAMPLES_CSV:
            headers = [
                "sample_id",
                "gt_mean","gt_max","gt_std",
                "pred1_mean","pred1_max","pred1_std",
                "pred2_mean","pred2_max","pred2_std",
                "chamfer_symmetric_pred1","chamfer_symmetric_pred2",
            ]
            with open(samples_csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for row in sample_rows:
                    outs = []
                    for v in row:
                        outs.append(f"{v:.10f}" if isinstance(v, (float, np.floating)) else str(v))
                    f.write(",".join(outs) + "\n")
            print(f"[OK] Wrote per-sample CSV → {samples_csv_path}")

        # per-sample columns as numpy arrays
        cols = list(zip(*sample_rows))
        # 0: sample_id
        gt_mean_array    = np.asarray(cols[1], dtype=float)
        gt_max_array     = np.asarray(cols[2], dtype=float)
        gt_std_array     = np.asarray(cols[3], dtype=float)

        pred1_mean_array = np.asarray(cols[4], dtype=float)
        pred1_max_array  = np.asarray(cols[5], dtype=float)
        pred1_std_array  = np.asarray(cols[6], dtype=float)

        pred2_mean_array = np.asarray(cols[7], dtype=float)
        pred2_max_array  = np.asarray(cols[8], dtype=float)
        pred2_std_array  = np.asarray(cols[9], dtype=float)

        chamfer_sym_pred1_array = np.asarray(cols[10], dtype=float)
        chamfer_sym_pred2_array = np.asarray(cols[11], dtype=float)

        # mean Chamfer distance across all samples (per experiment)
        chamfer_sym_pred1_mean_all = float(np.nanmean(chamfer_sym_pred1_array))
        chamfer_sym_pred2_mean_all = float(np.nanmean(chamfer_sym_pred2_array))

        # global pooled stats across all samples (GT, preds, differences)
        gt_mean_all,    gt_max_all,    gt_std_all    = gt_stats.mean,    gt_stats.max,    gt_stats.std
        pred1_mean_all, pred1_max_all, pred1_std_all = pred1_stats.mean, pred1_stats.max, pred1_stats.std
        pred2_mean_all, pred2_max_all, pred2_std_all = pred2_stats.mean, pred2_stats.max, pred2_stats.std
        diff1_mean_all, diff1_max_all, diff1_std_all = diff1_stats.mean, diff1_stats.max, diff1_stats.std
        diff2_mean_all, diff2_max_all, diff2_std_all = diff2_stats.mean, diff2_stats.max, diff2_stats.std

        #  totals CSV with global mean / max / std 
        if WRITE_TOTALS_CSV:
            totals_headers = [
                "gt_mean","gt_max","gt_std",
                "pred1_mean","pred1_max","pred1_std",
                "pred2_mean","pred2_max","pred2_std",
                "diff1_mean","diff1_max","diff1_std",
                "diff2_mean","diff2_max","diff2_std",
                "chamfer_symmetric_pred1_mean","chamfer_symmetric_pred2_mean",
                "num_samples"
            ]
            totals_values = [
                gt_mean_all, gt_max_all, gt_std_all,
                pred1_mean_all, pred1_max_all, pred1_std_all,
                pred2_mean_all, pred2_max_all, pred2_std_all,
                diff1_mean_all, diff1_max_all, diff1_std_all,
                diff2_mean_all, diff2_max_all, diff2_std_all,
                chamfer_sym_pred1_mean_all, chamfer_sym_pred2_mean_all,
                len(sample_rows)
            ]
            with open(totals_csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(totals_headers) + "\n")
                parts = []
                for v in totals_values:
                    if isinstance(v, (float, np.floating)):
                        parts.append(f"{v:.10f}")
                    else:
                        parts.append(str(v))
                f.write(",".join(parts) + "\n")
            print(f"[OK] Wrote dataset totals CSV → {totals_csv_path}")

    # Boxplot: per-node distributions
    if MAKE_BOXPLOTS and all_mag_gt:
        # flatten all nodes from all samples
        mag_gt_all      = np.concatenate(all_mag_gt)
        mag_pred1_all   = np.concatenate(all_mag_pred1)
        mag_pred2_all   = np.concatenate(all_mag_pred2)
        difference1_all = np.concatenate(all_difference1)
        difference2_all = np.concatenate(all_difference2)

        def make_boxplot(data_dict, title, outfile, ylabel=None, showfliers=False):
            fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
            labels = list(data_dict.keys())
            data   = [np.asarray(v, dtype=float) for v in data_dict.values()]
            ax.boxplot(
                data,
                tick_labels=labels,
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

        # Springback magnitude (per-node)
        make_boxplot(
            {
                "Ground Truth":                 mag_gt_all,
                "Vertex-based Hybrid Approach": mag_pred1_all,
                "Edge-based Hybrid Approach":   mag_pred2_all,
            },
            f" Springback Magnitude (op{operation})",
            f"{experiment_name}_boxplot_gt_pred1_pred2_pernode.png",
            ylabel="Springback Displacement"
        )

        # Springback difference (per-node)
        make_boxplot(
            {
                "Vertex-based Hybrid Approach": difference1_all,
                "Edge-based Hybrid Approach":   difference2_all,
            },
            f"Springback Difference (op{operation})",
            f"{experiment_name}_boxplot_pred1_pred2_diff_pernode.png",
            ylabel="Springback Difference (|Prediction − Ground Truth|)"
        )

# Example run:
#python /home/RUS_CIP/st186731/research_project/hybrid_approach/Evalutation/evaluation_2.py
