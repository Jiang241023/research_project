import numpy as np
import h5py
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
from matplotlib.colors import Normalize

from hybrid_approach.input_pipeline.DDACSDataset import DDACSDataset
from hybrid_approach.utils.utils_DDACS import (
    extract_point_cloud,
    extract_mesh,
    display_structure,
    extract_element_thickness,
    extract_point_springback,
)

data_dir = Path("/mnt/data/darus/")

dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

# Display sample simulation
sim_id, metadata, h5_path = dataset[0]
print(f"Sample simulation: {sim_id}")
print(f"File: {h5_path}")
print(f"Metadata: {metadata}")


try:
    import scienceplots

    plt.style.use(["science"])
except ImportError as exc:
    print("For proper plotting, use scienceplots")

# Standard figure sizes (width, height) in inches based on paper dimensions.
FIGURE_SIZES = {
    "single_col": (3.5, 2.6),  # Single column
    "single_col_cb": (5, 3.5),  # Single column, with colorbar
    "single_col_tall": (3.5, 3.5),  # Single column, square-ish
    "double_col": (7.0, 3.0),  # Double column, wide
    "double_col_tall": (7.0, 4.5),  # Double column, taller
    "square": (3.5, 3.5),  # Square single column
    "poster": (10, 8),  # For presentations/posters
}

# Visualization constants for consistency
FIGURE_SIZE = FIGURE_SIZES["double_col"]
FIGURE_DPI = 150
AXIS_LIMITS = [0, 110]
VIEW_ELEVATION = 30
VIEW_AZIMUTH = 45

# Component settings
COMPONENT_COLORS = {"blank": "red", "die": "blue", "punch": "green", "binder": "orange"}
COMPONENT_NAMES = {
    "blank": "Blank (Workpiece)",
    "die": "Die (Lower Tool)",
    "punch": "Punch (Upper Tool)",
    "binder": "Binder (Clamp)",
}

# Variables
THICKNESS_TIMESTEP = 3  # The last time step of the operation.
COMPONENT = "blank"  # Only blank possible if operation 20 is selected.
OPERATION = 10  # Can be 10 or 20.

fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
ax = fig.add_subplot(111, projection="3d")

# Get mesh and thickness information
vertices, triangles = extract_mesh(
    h5_path, "blank", timestep=THICKNESS_TIMESTEP, operation=OPERATION
)
print("the shape of vertices:", vertices.shape)
thickness = extract_element_thickness(
    h5_path, timestep=THICKNESS_TIMESTEP, operation=OPERATION
)
faces = vertices[triangles]

# Map element data to triangles
thickness_per_triangle = np.repeat(thickness, 2)[: len(triangles)]
print(f"Thickness range: {thickness.min():.4f} - {thickness.max():.4f} mm")

# It is recommended to try set this globally. Keeping these values stationary, will result in similar colormaps over different simulations and makes them actually comparable.
global_thickness_min = 0.8
global_thickness_max = 1.15

norm = Normalize(vmin=global_thickness_min, vmax=global_thickness_max)
colors = plt.cm.viridis(norm(thickness_per_triangle))

collection = Poly3DCollection(faces, facecolors=colors, edgecolors=colors, alpha=1)

ax.add_collection3d(collection)

sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
cbar.set_label("Thickness [mm]")

ax.set_title(
    f"Thickness Distribution - Simulation {sim_id} - Timestep {THICKNESS_TIMESTEP} - OP{OPERATION}"
)
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")

ax.set_xlim(AXIS_LIMITS)
ax.set_ylim(AXIS_LIMITS)
ax.set_zlim(AXIS_LIMITS)
ax.view_init(VIEW_ELEVATION, VIEW_AZIMUTH)

plt.tight_layout()
plt.show()