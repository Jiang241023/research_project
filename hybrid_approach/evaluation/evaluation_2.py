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
        return float("nan") if self.total_count <= 0 else self.running_sum / self.total_count
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
    return out  # dict: sid -> path

def summarize(values):
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(v.mean()), float(v.max()), float(v.std()))

# main (plot per-node displacement magnitudes across all samples)
if __name__ == "__main__":
    # Config
    operation = 10 # 10 or 20
    timestep  = 2 # 2 or 0

    #  set two experiment folders
    pred_dir_1 = "/home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/grit_like-fullsamples-10epochs-alpha0.8-beta0.2-grit_likewithlap"
    pred_dir_2 = "/home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/graphormer_like-fullsamples-10epochs-alpha0.8-beta0.2-graphormer_likewithlap"

    data_dir   = Path("/mnt/data/darus/")
    experiment_name = "compare_grit_like_vs_graphormer_like_op10_1"

    MAKE_BOXPLOTS     = True
    WRITE_SAMPLES_CSV = True

    if operation == 10 and timestep == 2:
        save_dir = Path("/home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/Evaluation_output/op10")
    elif operation == 20 and timestep == 0:
        save_dir = Path("/home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/Evaluation_output/op20")
    else:
        raise ValueError("please check the operation and timestep")
    save_dir.mkdir(parents=True, exist_ok=True)

    samples_csv_path = save_dir / f"{experiment_name}_per_sample.csv"

    # Load dataset & predictions
    dataset = DDACSDataset(data_dir, "h5")
    preds1  = scan_prediction_files(pred_dir_1)   # sid -> file
    preds2  = scan_prediction_files(pred_dir_2)   # sid -> file

    # Work on the intersection of sample IDs present in both experiments
    common_ids = sorted(set(preds1.keys()).intersection(preds2.keys()), key=int)
    if not common_ids:
        raise SystemExit("No overlapping prediction files between the two experiments.")

    # Stats
    gt_stats     = RunningStats()
    pred1_stats  = RunningStats()
    pred2_stats  = RunningStats()

    sample_rows = []  # one row per sample

    # NEW: collect all per-node magnitudes for boxplot (no mean)
    all_mag_gt    = []
    all_mag_pred1 = []
    all_mag_pred2 = []
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
                raise ValueError(f"[{sid}] exp1 shape {disp_pred1.shape} != gt {displacement_gt.shape}")
            if displacement_pred2.shape != displacement_gt.shape:
                raise ValueError(f"[{sid}] exp2 shape {displacement_pred2.shape} != gt {displacement_gt.shape}")

            # Magnitudes
            mag_gt     = np.linalg.norm(displacement_gt, axis=1)
            mag_pred1  = np.linalg.norm(displacement_pred1, axis=1)
            mag_pred2  = np.linalg.norm(displacement_pred2, axis=1)

            # Per-node absolute differences vs GT
            difference_1 = np.linalg.norm(displacement_pred1 - displacement_gt, axis=1)
            difference_2 = np.linalg.norm(displacement_pred2 - displacement_gt, axis=1)

            # accumulate per-node magnitudes for global boxplot
            all_mag_gt.append(mag_gt)
            all_mag_pred1.append(mag_pred1)
            all_mag_pred2.append(mag_pred2)
            all_difference1.append(difference_1)
            all_difference2.append(difference_2)

            # stats
            gt_stats.update(mag_gt)
            pred1_stats.update(mag_pred1)
            pred2_stats.update(mag_pred2)

            # Per-sample summaries (for CSV only)
            gt_mean_summarize, gt_max_summarize, gt_std_summarize = summarize(mag_gt)
            pred1_mean_summarize, pred1_max_summarize, pred1_std_summarize = summarize(mag_pred1)
            pred2_mean_summarize, pred2_max_summarize, pred2_std_summarize = summarize(mag_pred2)

            sample_rows.append([
                int(sid),
                gt_mean_summarize, gt_max_summarize, gt_std_summarize,
                pred1_mean_summarize, pred1_max_summarize, pred1_std_summarize,
                pred2_mean_summarize, pred2_max_summarize, pred2_std_summarize
            ])

            if (k % 10 == 0) or (k == len(common_ids)):
                print(f"[{k}/{len(common_ids)}] processed")

        except Exception as e:
            print(f"[WARN] Skipping id={sid}: {e}")

    # CSV with per-sample numbers 
    if WRITE_SAMPLES_CSV and sample_rows:
        headers = [
            "sample_id",
            "gt_mean","gt_max","gt_std",
            "pred1_mean","pred1_max","pred1_std",
            "pred2_mean","pred2_max","pred2_std",
        ]
        with open(samples_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in sample_rows:
                outs = []
                for v in row:
                    outs.append(f"{v:.10f}" if isinstance(v, (float, np.floating)) else str(v))
                f.write(",".join(outs) + "\n")
        print(f"[OK] Wrote per-sample CSV → {samples_csv_path}")

    # Boxplot: per-node distributions
    if MAKE_BOXPLOTS and all_mag_gt:
        # NEW: flatten all nodes from all samples
        mag_gt_all    = np.concatenate(all_mag_gt)
        mag_pred1_all = np.concatenate(all_mag_pred1)
        mag_pred2_all = np.concatenate(all_mag_pred2)
        difference1_all = np.concatenate(all_difference1)
        difference2_all     = np.concatenate(all_difference2)


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

        # Springback magnitude
        make_boxplot(
            {
                "Ground Truth":                mag_gt_all,
                "Vertex-based Hybrid Approach": mag_pred1_all,
                "Edge-based Hybrid Approach":   mag_pred2_all,
            },
            f"{experiment_name}: Springback magnitude (per-node)",
            f"{experiment_name}_boxplot_gt_pred1_pred2_pernode.png",
            ylabel="Displacement"
        )

        # Springback difference
        make_boxplot(
            {
                "Vertex-based Hybrid Approach": difference1_all,
                "Edge-based Hybrid Approach":   difference2_all,
            },
            f"{experiment_name}: Springback difference (per-node)",
            f"{experiment_name}_boxplot_pred1_pred2_diff_pernode.png",
            ylabel="|Prediction − Ground Truth|"
        )
