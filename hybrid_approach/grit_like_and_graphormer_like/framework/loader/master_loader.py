import logging
import time
import warnings
from functools import partial

import torch
import torch_geometric.transforms as T
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg
from torch_geometric.graphgym.register import register_loader

from framework.loader.split_generator import (#prepare_splits,
                                         set_dataset_splits)

from framework.loader.dataset.ddacs_npy_stream import DDACSNPYStream

def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")

    # Show a peek at the first sample (safe for streaming datasets)
    try:
        d0 = dataset[0]
        def _shape(x):
            if x is not None and hasattr(x, 'shape'):
                x = tuple(x.shape)  
            else:
                x = None
            return x
        
        x_s   = _shape(getattr(d0, 'x', None))
        pos_s = _shape(getattr(d0, 'pos', None))
        ei_s  = _shape(getattr(d0, 'edge_index', None))
        y_s   = _shape(getattr(d0, 'y', None))
        ea_s  = _shape(getattr(d0, 'edge_attr', None))
            
        logging.info(
            "sample[0]: Data("
            f"new_concatenated_features={x_s}, "
            f"node_coords={pos_s}, "
            f"edge_index={ei_s}, "
            f"node_displacement={y_s}, "
            f"edge_features={ea_s})"
        )
    except Exception as e:
        logging.info(f"sample[0]: <unavailable due to: {e}>")

    logging.info(f"  num graphs: {len(dataset)}")

    # Estimate avg num_nodes/graph from a small sample to avoid scanning the whole set
    try:
        k = min(16, len(dataset))
        sizes = []
        for i in range(k):
            di = dataset[i]
            sizes.append(int(getattr(di, 'num_nodes', di.x.size(0) if getattr(di, 'x', None) is not None else 0)))
        avg_nodes = int(round(sum(sizes) / max(1, len(sizes))))
        logging.info(f"  approx avg num_nodes/graph: {avg_nodes}")
    except Exception as e:
        logging.info(f"  approx avg num_nodes/graph: n/a ({e})")

@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    # Create dataset
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        print(f"pyg_dataset_id (from load_dataset_master):{pyg_dataset_id}")
        if pyg_dataset_id == 'DDACSNPYStream':
            dataset = DDACSNPYStream(
                root=dataset_dir,
                max_samples=getattr(cfg.dataset, 'max_samples', None),
                debug=getattr(cfg, 'debug', False),
            )

            # Make and APPLY graph-wise splits (70/20/10 by default) for streaming
            if hasattr(dataset, 'get_idx_split'):
                s = dataset.get_idx_split(seed=cfg.seed) # {'train': [...], 'val': [...], 'test': [...]}
                splits = [s['train'], s['val'], s['test']]
                set_dataset_splits(dataset, splits)      # <-- apply directly (writes to dataset.data)

                # Check the splits
                print(len(dataset))
                train, val, test = splits
                print(f"[Splits] sizes: train={len(train)}  val={len(val)}  test={len(test)}")
                print(f"train[:10]:\n{dataset.data.train_graph_index[:10]}")
                print(f"val[:10]:\n{dataset.data.val_graph_index[:10]}")
                print(f"test[:10]:\n{dataset.data.test_graph_index[:10]}")

                # Check one sample from each split
                index_train = int(dataset.data.train_graph_index[0])
                index_val = int(dataset.data.val_graph_index[0])
                index_test = int(dataset.data.test_graph_index[0])

                sample_train = dataset[index_train]
                sample_val   = dataset[index_val]
                sample_test  = dataset[index_test]

                print("TRAIN → idx:", index_train, "sample_id:", getattr(sample_train, "sample_id", None))
                print("VAL   → idx:", index_val,   "sample_id:", getattr(sample_val, "sample_id", None))
                print("TEST  → idx:", index_test,  "sample_id:", getattr(sample_test, "sample_id", None))

                for name_dataset, data in [("train", sample_train), ("val", sample_val), ("test", sample_test)]:
                    nx = None if getattr(data, "x", None) is None else tuple(data.x.shape)
                    ny = None if getattr(data, "y", None) is None else tuple(data.y.shape)
                    ne = None if getattr(data, "edge_index", None) is None else data.edge_index.size(1)
                    print(f"{name_dataset}: new_concatenated_features={nx}, node_coords={ny}, edge_index={ne}, num_nodes={getattr(data, 'num_nodes', None)}")

            else:
                raise RuntimeError("Dataset has no get_idx_split() to generate splits.")

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")
    else:
        raise ValueError(f"Unknown data format: {format}")

    # Log a quick summary (safe for streaming datasets)
    log_loaded_dataset(dataset, format, name)

    return dataset

