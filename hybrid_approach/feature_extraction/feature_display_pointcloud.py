from matplotlib.cm import ScalarMappable
import numpy as np
import h5py
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
from matplotlib.colors import Normalize
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
COMPONENT_COLORS = {"blank": "red"}
COMPONENT_NAMES = {
    "blank": "Blank (Workpiece)",
}

# Variables
FINAL_FORMING_TIMESTEP = (
    2  # The last time step of the forming operation where the tools and blank exist.
)

fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
ax = fig.add_subplot(111, projection="3d")

for component in COMPONENT_COLORS.keys():
    try:
        # Get point cloud information
        coords = extract_point_cloud(
            h5_path, component, timestep=FINAL_FORMING_TIMESTEP
        )
        alpha = 0.3 if component == "blank" else 0.7
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=COMPONENT_COLORS[component],
            label=COMPONENT_NAMES[component],
            s=0.5,
            alpha=alpha,
        )
        print(f"{component}: {coords.shape[0]} nodes")
    except Exception as e:
        print(f"Could not load {component}: {e}")

ax.set_title(
    f"Deep Drawing Setup - Simulation {sim_id} - Timestep {FINAL_FORMING_TIMESTEP} - OP10"
)

ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")

ax.set_xlim(AXIS_LIMITS)
ax.set_ylim(AXIS_LIMITS)
ax.set_zlim(AXIS_LIMITS)
ax.view_init(VIEW_ELEVATION, VIEW_AZIMUTH)

ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")

plt.tight_layout()
plt.show()