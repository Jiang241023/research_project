from matplotlib.cm import ScalarMappable
import numpy as np
import h5py
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
from matplotlib.colors import Normalize



# ===================== User-adjustable settings =====================

# Data root (where your HDF5 dataset lives)
DATA_DIR = Path("/mnt/data/darus/")  # adapt if needed

# Which simulation to render
SIM_INDEX = 0  # dataset[0]

# What to display
FIELD     = "strain"   # 'stress' or 'strain'
WHICH     = "von Mises value"       # 'von Mises value' or one of: 'xx','yy','zz','xy','yz','zx'

# Operation, component, and timestep
OPERATION = 10         # 10 or 20
COMPONENT = "blank"    # for OP20 only 'blank' is valid in your setup
TIMESTEP  = -1         # use -1 for last, or e.g., 3 for explicit index

# Plot look & feel
FIGURE_SIZES = {
    "single_col": (3.5, 2.6),
    "single_col_cb": (5, 3.5),
    "single_col_tall": (3.5, 3.5),
    "double_col": (7.0, 3.0),
    "double_col_tall": (7.0, 4.5),
    "square": (3.5, 3.5),
    "poster": (10, 8),
}
FIGURE_SIZE = FIGURE_SIZES["double_col"]
FIGURE_DPI  = 150
AXIS_LIMITS = [0, 110]   # adapt to your geometry
VIEW_ELEV   = 30
VIEW_AZI    = 45

# Where to save output figures
SAVE_DIR = Path("/home/RUS_CIP/st186731/research_project/figure")

# ====================================================================

# ddacs imports (package version)
from DDACSDataset import DDACSDataset
from utils_DDACS import (
    extract_mesh,
)

# ------------------------- I/O helpers -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(filename_base: str, dpi: int = FIGURE_DPI):
    """Save current plt figure as PNG + PDF in SAVE_DIR."""
    ensure_dir(SAVE_DIR)
    png_path = SAVE_DIR / f"{filename_base}.png"
    pdf_path = SAVE_DIR / f"{filename_base}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")

# ------------------------- Data helpers -------------------------

def load_raw_stress_strain(h5_path: Path, component="blank", timestep=-1, operation=10):
    """
    Returns shell element stress/strain for a given timestep.
      stress_t: (m,3,6), strain_t: (m,2,6)
    """
    with h5py.File(h5_path, "r") as f:
        comp = f[f"OP{operation}"][component]
        stress_t = comp["element_shell_stress"][timestep]  # (m,3,6)
        strain_t = comp["element_shell_strain"][timestep]  # (m,2,6)
    return stress_t, strain_t

def avg_thickness_points(stress_t: np.ndarray, strain_t: np.ndarray):
    """
    Average through-thickness integration points.
      stress_t: (m,3,6) → (m,6)
      strain_t: (m,2,6) → (m,6)
    """
    stress6 = stress_t.mean(axis=1).astype(np.float32)
    strain6 = strain_t.mean(axis=1).astype(np.float32)
    return stress6, strain6

def von_mises_from_6(six: np.ndarray):
    """
    six ordered [xx, yy, zz, xy, yz, zx] → von Mises equivalent scalar.
    Works for both stress and strain.
    """
    sxx, syy, szz, txy, tyz, tzx = [six[..., i] for i in range(6)]
    j2 = 0.5*((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) + 3.0*(txy**2 + tyz**2 + tzx**2)
    return np.sqrt(np.maximum(j2, 0.0))

def elem_scalar(stress6: np.ndarray, strain6: np.ndarray, field="stress", which="von Mises value"):
    """
    Select a scalar per element to visualize.
      field ∈ {'stress','strain'}
      which ∈ {'von Mises value','xx','yy','zz','xy','yz','zx'}
    Returns: (m,) array
    """
    tensor = stress6 if field == "stress" else strain6
    if which == "von Mises value":
        return von_mises_from_6(tensor)
    idx = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "zx": 5}[which]
    return tensor[:, idx]

def repeat_elem_to_triangles(elem_vals: np.ndarray, triangles: np.ndarray):
    """
    Your mesh is quads split into 2 triangles → repeat per-element values to per-triangle.
    elem_vals: (m,)
    triangles: (2m, 3)
    Returns: (2m,)
    """
    m = len(elem_vals)
    repeat = len(triangles) // m
    if repeat * m != len(triangles):
        raise ValueError("Triangle count must be an integer multiple of elements.")
    return np.repeat(elem_vals, repeat)

# ------------------------- Plotting -------------------------

def plot_tri_scalar(vertices, triangles, tri_vals, title,
                    vmin=None, vmax=None,
                    fname_base=None,
                    cbar_label=" stress [MPa]",
                    cbar_on_right=True):
    faces = vertices[triangles]

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax  = fig.add_subplot(111, projection="3d")

    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(tri_vals, [1, 99])
        if vmin == vmax:
            vmin, vmax = float(tri_vals.min()), float(tri_vals.max())

    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    colors = plt.cm.viridis(norm(tri_vals))

    coll = Poly3DCollection(faces, facecolors=colors, edgecolors=colors, alpha=1.0)
    ax.add_collection3d(coll)

    sm = ScalarMappable(cmap="viridis", norm=norm)
    # orientation options: 'vertical' or 'horizontal'
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05, orientation="vertical")
    # --- set a short, clean label ---
    cbar.set_label(cbar_label, rotation=270, labelpad=15, va="center")
    if cbar_on_right:
        cbar.ax.yaxis.set_label_position('right')  # keep label on the right

    # (optional) format ticks
    # from matplotlib import ticker
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    ax.set_title(title)
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
    ax.set_xlim(AXIS_LIMITS); ax.set_ylim(AXIS_LIMITS); ax.set_zlim(AXIS_LIMITS)
    ax.view_init(VIEW_ELEV, VIEW_AZI)
    plt.tight_layout()

    if fname_base:
        savefig(fname_base, dpi=FIGURE_DPI)

    plt.show()

# ------------------------- Main routine -------------------------

def main():
    # Load dataset and pick a sample
    dataset = DDACSDataset(DATA_DIR, "h5")
    print(f"Loaded {len(dataset)} simulations")

    sim_id, metadata, h5_path = dataset[SIM_INDEX]
    print(f"Sample simulation: {sim_id}")
    print(f"File: {h5_path}")
    print(f"Metadata: {metadata}")

    # Mesh
    vertices, triangles = extract_mesh(h5_path, component=COMPONENT, timestep=TIMESTEP, operation=OPERATION)

    # Element stress/strain → (m,6) each
    stress_t, strain_t  = load_raw_stress_strain(h5_path, component=COMPONENT, timestep=TIMESTEP, operation=OPERATION)
    stress6, strain6    = avg_thickness_points(stress_t, strain_t)

    # Choose scalar per element
    elem_vals = elem_scalar(stress6, strain6, field=FIELD, which=WHICH)  # (m,)
    tri_vals  = repeat_elem_to_triangles(elem_vals, triangles)            # (2m,)

    # Titles & filenames
    unit_hint = " (units as in file)" if FIELD == "stress" else ""
    title_mesh = f"{FIELD.capitalize()} ({WHICH.upper()}) – Sim {sim_id} – Timestep={TIMESTEP} – OP{OPERATION}"
    base = f"{FIELD.capitalize()}"

    # Plot + save
    plot_tri_scalar(vertices, triangles, tri_vals, title_mesh, fname_base=f"{base}_trisurf")


if __name__ == "__main__":
    main()
