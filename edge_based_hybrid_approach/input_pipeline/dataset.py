import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import ticker
import sys
sys.path.append("../research_project/edge_based_hybrid_approach")

from DDACSDataset import DDACSDataset
from utils.utils_DDACS import extract_point_cloud, extract_mesh, display_structure, extract_element_thickness, extract_point_springback

from torch.utils.data import DataLoader, random_split
import torch

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
    #   print(f"  > {key:4} - {desc[key]}")

    #print(f"\nSample simulations:")
    #print(df.head())

    component = "blank"
    timestep = 3

    with h5py.File(h5_path, "r") as f:
        # choose "blank"
        op10 = f["OP10"]
        comp = op10[component]

        # coordinates (num_elements, 3) 
        coords_all = comp["node_coordinates"][:] # [:] tells h5py to read the entire dataset from disk into a NumPy array  
        coords_at_definite_timestep = coords_all

        # displacements (timesteps, num_elements, 3) 
        disp_all = comp["node_displacement"][:] # tells h5py to read the entire dataset from disk into a NumPy array   
        disp_at_definite_timestep = disp_all[timestep]

        # Stress tensor (timesteps, num_elements, 3, 6)
        stress = comp["element_shell_stress"][:]  
        stress_at_definite_timestep = stress[timestep]
        

        # Strain tensor (timesteps, num_elements, 2, 6)
        strain = comp["element_shell_strain"][:] 
        strain_at_definite_timestep = strain[timestep]

        # Get springback information
        final_coords, displacement_vectors = extract_point_springback(h5_path, operation=10)


    print(f"the shape of final_coords at timestep {timestep}:", final_coords.shape)
    print(f"the shape of displacement_vectors at timestep {timestep}:", displacement_vectors.shape)
    print(f"the shape of coords at timestep {timestep}:", coords_at_definite_timestep.shape)
    print(f"the shape of disp at timestep {timestep}:", disp_at_definite_timestep.shape)

    # build features
    concatenated_representations_vertex = np.concatenate([coords_at_definite_timestep, disp_at_definite_timestep], axis=1)
    print("Vertex representation matrix shape:", concatenated_representations_vertex.shape)
    print("First 5 rows of vetex representation matrix:\n", concatenated_representations_vertex[:5])

    print("Stress shape:", stress.shape)
    print("Strain shape:", strain.shape)

    print(f"the shape of Stress at timestep {timestep}:", stress_at_definite_timestep.shape)
    print(f"the shape of strain at timestep {timestep}:", strain_at_definite_timestep.shape)

    # Example: get first element stress/strain
    print("first element of Stress:\n", stress_at_definite_timestep[0])
    print("first element of Strain:\n", strain_at_definite_timestep[0])

    #Example: get 2 elements final_coords/displacement_vectors
    print("elements of final_coords:\n", final_coords[:2])
    print("elements of displacement_vectors:\n", displacement_vectors[:2])




#ds = DDACSDataset(data_dir, "h5")

# Always set before getting random operation
torch.random.seed = 0
train_fraction, test_fraction, eval_fraction = 0.7, 0.2, 0.1

train_dataset, test_dataset, eval_dataset = random_split(dataset, [train_fraction, test_fraction, eval_fraction])
print(f"Length of the datasets:\n - train: {len(train_dataset)}\n - test: {len(test_dataset)}\n - eval: {len(eval_dataset)}")

train_dataloader = DataLoader(train_dataset, 16, shuffle=True)
test_dataloader = DataLoader(test_dataset, 4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, 4, shuffle=True)



    