import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
from DDACSDataset import DDACSDataset
from edge_based_hybrid_approach.utils.utils_DDACS import extract_point_cloud, extract_mesh, display_structure, extract_element_thickness, extract_point_springbackpy
# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

# Get a sample simulation
sim_id, metadata, h5_path = dataset[0]
print(f"Sample simulation: {sim_id}")
print(f"File: {h5_path}")
print(f"Metadata: {metadata}")