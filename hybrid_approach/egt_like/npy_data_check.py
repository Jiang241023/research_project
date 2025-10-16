from pathlib import Path
import numpy as np

sample_to_check = "16039"  # change to any ID you saved

data_dir = Path("/mnt/data/jiang")
pred_dir = Path("results/ddacs-node-regression/preds_new")

X  = np.load(data_dir / f"{sample_to_check}_new_concatenated_features.npy")
EI = np.load(data_dir / f"{sample_to_check}_edge_index.npy")
EF = np.load(data_dir / f"{sample_to_check}_edge_features.npy")
node_coords = np.load(data_dir / f"{sample_to_check}_node_coords.npy")
Y = np.load(data_dir / f"{sample_to_check}_node_displacement.npy")
# PREDICTED displacement (this is the new file)
Y_pred = np.load(pred_dir / f"{sample_to_check}_pred_node_displacement.npy")

# Optional files (only print if present)
node_index_path = data_dir / f"{sample_to_check}_node_index.npy"
EI_2_path = data_dir / f"{sample_to_check}_edge_index_2.npy"
node_index = np.load(node_index_path) if node_index_path.exists() else None
EI_2 = np.load(EI_2_path) if EI_2_path.exists() else None

print("-----------------------------------")
print(f"For sample {sample_to_check} (PRED set):")
print("-----------------------------------")
# print(f"new_concatenated_features[0]:\n{X[0]}")
print(f"node_displacement:\n{Y}")
print(f"pred_node_displacement:\n{Y_pred}")
# print(f"edge_index[0]: {EI[0]}")
# print(f"edge_features[0]:\n{EF[0]}")
# print(f"node_coords[0]:\n{node_coords[0]}")
# print(f"node_index[0]:\n{node_index[0] if node_index is not None else '<not found>'}")
# print(f"edge_index_2[0]: {EI_2[0] if EI_2 is not None else '<not found>'}")
