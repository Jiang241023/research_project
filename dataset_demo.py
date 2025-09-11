import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ddacs import DDACSDataset
from ddacs.utils import extract_point_cloud, extract_mesh, display_structure, extract_element_thickness
# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

# Get a sample simulation
sim_id, metadata, h5_path = dataset[200]
print(f"Sample simulation: {sim_id}")
print(f"File: {h5_path}")
print(f"Metadata: {metadata}")

# Load and analyze metadata
df = dataset._metadata
print(f"Metadata structure:")
print(f"  Total simulations: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print("  > Note: The column 'ID' is not returned in dataset.")

# Show parameter meanings and ranges
print(f"\nParameters (physical values):")

# Note: Update these when you have the actual unnormalized data
desc = dataset.get_metadata_descriptions()
for key in desc:
    print(f"  > {key:4} - {desc[key]}")

print(f"\nSample simulations:")
print(df.head())