import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
import sys
sys.path.append("../edge_based_hybrid_approach")

from DDACSDataset import DDACSDataset
from utils.utils_DDACS import extract_point_cloud, extract_mesh, display_structure, extract_element_thickness, extract_point_springback

# Setup data directory
data_dir = Path("/mnt/data/darus/")

# Load dataset
dataset = DDACSDataset(data_dir, "h5")
print(f"Loaded {len(dataset)} simulations")

# Get a sample simulation
for index in range(1): # If all datasets are needed, just set range(len(dataset))
    sim_id, metadata, h5_path = dataset[index]
    print(f"Sample simulation: {sim_id}")
    print(f"File: {h5_path}")
    print(f"Metadata: {metadata}")

    # Load and analyze metadata
    #df = dataset._metadata
    #print(f"Metadata structure:")
    #print(f"  Total simulations: {len(df)}")
    #print(f"  Columns: {list(df.columns)}")
    #print("  > Note: The column 'ID' is not returned in dataset.")

    # Show parameter meanings and ranges
    #print(f"\nParameters (physical values):")

    # Note: Update these when you have the actual unnormalized data
    #desc = dataset.get_metadata_descriptions()
    #for key in desc:
    #    print(f"  > {key:4} - {desc[key]}")

    #print(f"\nSample simulations:")
    #print(df.head())

    component = "blank"
    timestep = 3

    with h5py.File(h5_path, "r") as f:
        # choose "blank"
        op10 = f["OP10"]
        comp = op10[component]

        # coordinates
        coords_all = comp["node_coordinates"][:] # [:] tells h5py to read the entire dataset from disk into a NumPy array  
        print(f"the dimension of coords_all parameter is: {coords_all.ndim}")
        coords_with_definite_timestep = coords_all

        # displacements
        disp_all = comp["node_displacement"][:] # tells h5py to read the entire dataset from disk into a NumPy array
        print(f"the dimension of disp_all parameter is: {disp_all.ndim}")    
        disp_with_definite_timestep = disp_all[timestep]

    print(f"the shape of coords with timestep {timestep}:", coords_with_definite_timestep.shape)
    print(f"the shape of disp with timestep {timestep}:", disp_with_definite_timestep.shape)

    # build features
    concatenated_representations = np.concatenate([coords_with_definite_timestep, disp_with_definite_timestep], axis=1)
    print("Vertex feature matrix shape:", concatenated_representations.shape)
    #print("First 5 rows:\n", concatenated_representations[:5])