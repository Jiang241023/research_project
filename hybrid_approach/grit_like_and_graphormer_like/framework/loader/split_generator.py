"""This file is from GRIT:https://github.com/LiamMa/GRIT/blob/main/grit"""
import numpy as np
import torch 
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import index2mask #, set_dataset_attr

def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object.

    Args:
        dataset: PyG dataset object (works for streaming or InMemory)
        splits:  [train_ids, val_ids, test_ids] as lists/arrays of graph indices
    """
    # 1) Validate: no overlap between splits
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            si, sj = set(map(int, splits[i])), set(map(int, splits[j]))
            n_intersect = len(si & sj)
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not intersect: split #{i} (n={len(si)}) "
                    f"and split #{j} (n={len(sj)}) intersect in {n_intersect} indices."
                )

    # 2) Decide task level for splitting
    task_level = cfg.dataset.task
    y_carrier = getattr(dataset.data, 'y', None)
    is_multi_graph = len(dataset) > 1

    # For multi-graph node regression datasets, split by graphs:
    if task_level == 'node' and (y_carrier is None or is_multi_graph):
        task_level = 'graph'

    # 3) Assign splits
    if task_level == 'graph':
        split_names = ['train_graph_index', 'val_graph_index', 'test_graph_index']
        for split_name, split_index in zip(split_names, splits):
            # write directly to dataset.data for GraphGym compatibility
            setattr(dataset.data, split_name, torch.as_tensor(split_index, dtype=torch.long))

    elif task_level == 'node':
        # Single-graph, node-level masks expected
        split_names = ['train_mask', 'val_mask', 'test_mask']
        if getattr(dataset.data, 'y', None) is None:
            raise ValueError("Node-level split requested but dataset.data.y is missing.")
        size = int(dataset.data.y.shape[0])
        for split_name, split_index in zip(split_names, splits):
            mask = index2mask(split_index, size=size)
            print(f"mask (from set_dataset_splits):{mask}")
            setattr(dataset.data, split_name, mask)

    else:
        raise ValueError(f"Unsupported dataset task level: {task_level}")
