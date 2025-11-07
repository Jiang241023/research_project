import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib import ticker
from matplotlib.ticker import LogFormatter, LogLocator
from pathlib import Path

from DDACSDataset import DDACSDataset
from utils_DDACS import extract_mesh, extract_point_springback

def find_h5_by_id(dataset, sid):
    """Return (sim_id, metadata, h5_path) for the given string/int sample id."""
    sid = str(sid)
    for i in range(len(dataset)):
        sim_id, meta, h5_path = dataset[i]
        if str(sim_id) == sid:
            return sim_id, meta, h5_path
    raise FileNotFoundError(f"Sample id {sid} not found in dataset")

def add_colorbar(fig, ax, mappable, label):
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.8)
    if isinstance(mappable.norm, LogNorm):
        cbar.locator   = LogLocator(base=10, subs=(1.,))
        cbar.formatter = LogFormatter(base=10, labelOnlyBase=False)
        cbar.update_ticks()
    else:
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.set_label(label, fontsize=11)
    return cbar

def scan_prediction_files(pred_dir):
    """Return list of (sample_id_str, full_path) for *_pred_node_displacement.npy files, sorted by id."""
    pattern = re.compile(r"^(\d+)_pred_node_displacement\.npy$")
    out = []
    for name in os.listdir(pred_dir):
        m = pattern.match(name)
        if m:
            out.append((m.group(1), os.path.join(pred_dir, name)))
    out.sort(key=lambda t: int(t[0]))
    return out

def plot_one_sample(sim_id, node_coords, disp_gt, disp_pred, save_path):
    """Make the 3-panel figure and save to save_path."""
    # Magnitudes
    mag_gt   = np.linalg.norm(disp_gt,   axis=1)  # [mm]
    mag_pred = np.linalg.norm(disp_pred, axis=1)  # [mm]
    difference_mag = np.linalg.norm(disp_pred - disp_gt, axis=1)

    # Coordinates to plot (GT final pose for comparison)
    coords_plot = node_coords + disp_gt

    # Color normalization
    norm_linear = Normalize(vmin=0.0, vmax=1.4)

    # LogNorm for differences: vmin > 0
    minimum_allowed_lower_bound = 1e-6
    pos_mask = difference_mag > 0
    vmin_diff = max(minimum_allowed_lower_bound,
                    difference_mag[pos_mask].min() if np.any(pos_mask) else minimum_allowed_lower_bound)
    vmax_diff = 0.3
    norm_difference = LogNorm(vmin=vmin_diff, vmax=vmax_diff)

    # Precompute colors for linear panels
    col1 = plt.cm.plasma(norm_linear(mag_gt))
    col2 = plt.cm.plasma(norm_linear(mag_pred))
    diff_for_plot = np.clip(difference_mag, vmin_diff, None)

    # Figure
    fig = plt.figure(figsize=(FIGURE_SIZE[0]*1.6, FIGURE_SIZE[1]), dpi=FIGURE_DPI)

    # Panel 1: GT magnitude
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(coords_plot[:, 0], coords_plot[:, 1], coords_plot[:, 2], c=col1, s=1, alpha=0.9)
    add_colorbar(fig, ax1, plt.cm.ScalarMappable(cmap="plasma", norm=norm_linear), "Magnitude [mm]")
    ax1.set_title(f"Original Springback\n(OP{OPERATION}, id={sim_id})")
    ax1.set_xlim(AXIS_LIMITS); ax1.set_ylim(AXIS_LIMITS); ax1.set_zlim(AXIS_LIMITS)
    ax1.view_init(VIEW_ELEVATION, VIEW_AZIMUTH)
    ax1.set_xlabel("X [mm]"); ax1.set_ylabel("Y [mm]"); ax1.set_zlabel("Z [mm]")

    # Panel 2: Pred magnitude
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(coords_plot[:, 0], coords_plot[:, 1], coords_plot[:, 2], c=col2, s=1, alpha=0.9)
    add_colorbar(fig, ax2, plt.cm.ScalarMappable(cmap="plasma", norm=norm_linear), "Magnitude [mm]")
    ax2.set_title(f"Predicted Springback\n(OP{OPERATION}, id={sim_id})")
    ax2.set_xlim(AXIS_LIMITS); ax2.set_ylim(AXIS_LIMITS); ax2.set_zlim(AXIS_LIMITS)
    ax2.view_init(VIEW_ELEVATION, VIEW_AZIMUTH)
    ax2.set_xlabel("X [mm]"); ax2.set_ylabel("Y [mm]"); ax2.set_zlabel("Z [mm]")

    # Panel 3: Difference (LOG)
    ax3 = fig.add_subplot(133, projection="3d")
    sc3 = ax3.scatter(
        coords_plot[:, 0], coords_plot[:, 1], coords_plot[:, 2],
        c=diff_for_plot, cmap="viridis", norm=norm_difference,
        s=1, alpha=0.95
    )
    cbar3 = fig.colorbar(sc3, ax=ax3, shrink=0.8)
    cbar3.set_ticks([3e-1, 1e-1, 1e-2, 1e-3])
    cbar3.set_ticklabels(['0.3', '0.1', '0.01', '0.001'])
    cbar3.set_label("Difference Magnitude [mm]", fontsize=11)

    ax3.set_title("Comparison (Log Scale)")
    ax3.set_xlim(AXIS_LIMITS); ax3.set_ylim(AXIS_LIMITS); ax3.set_zlim(AXIS_LIMITS)
    ax3.view_init(VIEW_ELEVATION, VIEW_AZIMUTH)
    ax3.set_xlabel("X [mm]"); ax3.set_ylabel("Y [mm]"); ax3.set_zlabel("Z [mm]")

    plt.suptitle(f"Springback Comparison - Sample {sim_id} - {MODEL_TAG}-fullsamples-15epochs-alpha1-beta1-grit_likewithlap", y=0.98, fontsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # important to free memory


if __name__ == "__main__":

    # Config
    OPERATION   = 10
    TIMESTEP    = 2
    # pred_dir    = "/home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/grit_like"
    # pred_dir    = "/home/RUS_CIP/st186731/research_project/RP-3875/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/graphormer_like"
    pred_dir    = "/home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/grit_like"
    data_dir    = Path("/mnt/data/darus/")

    FIGURE_SIZES = {"double_col": (7.0, 3.0)}
    FIGURE_SIZE  = FIGURE_SIZES["double_col"]
    FIGURE_DPI   = 150
    AXIS_LIMITS  = [0, 110]
    VIEW_ELEVATION = 30
    VIEW_AZIMUTH   = 45

    SAVE_DIR = Path("/home/RUS_CIP/st186731/research_project/figures/grit_like")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_TAG = Path(pred_dir).name  # e.g., "grit_like" — used in filenames

    dataset = DDACSDataset(data_dir, "h5")
    print(f"Loaded {len(dataset)} simulations")
    pairs = scan_prediction_files(pred_dir)
    if not pairs:
        raise SystemExit(f"No *_pred_node_displacement.npy files in: {pred_dir}")

    for idx, (sid_str, pred_path) in enumerate(pairs, 1):
        try:
            sim_id, metadata, h5_path = find_h5_by_id(dataset, sid_str)

            # Mesh + GT for this sample
            node_coords, triangles = extract_mesh(
                h5_path, operation=OPERATION, component='blank', timestep=TIMESTEP
            )
            final_coords_gt, disp_gt = extract_point_springback(h5_path, operation=OPERATION)

            # Prediction for this sample
            disp_pred = np.load(pred_path)
            if disp_pred.shape != disp_gt.shape:
                raise ValueError(f"shape mismatch for id={sid_str}: pred {disp_pred.shape} vs gt {disp_gt.shape}")

            # Save figure
            out_name = f"springback_id{sid_str}_{MODEL_TAG}.png"
            out_path = SAVE_DIR / out_name
            plot_one_sample(sim_id, node_coords, disp_gt, disp_pred, out_path)
            print(f"[{idx}/{len(pairs)}] saved → {out_path}")
        except Exception as e:
            print(f"[WARN] id={sid_str}: {e}")

# Example runs:
#python /home/RUS_CIP/st186731/research_project/hybrid_approach/visualization/visualization.py