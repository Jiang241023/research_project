import logging
import time
import warnings
from functools import partial

import torch
import torch_geometric.transforms as T
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg
from torch_geometric.graphgym.register import register_loader

from grit.loader.split_generator import (#prepare_splits,
                                         set_dataset_splits)
from grit.transform.posenc_stats import compute_posenc_stats, ComputePosencStat
from grit.transform.transforms import pre_transform_in_memory

from grit.loader.dataset.ddacs_npy_stream import DDACSNPYStream

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
            # Precreate graph-wise splits (70/20/10 by default) for streaming
            if not hasattr(dataset, 'split_idxs') and hasattr(dataset, 'get_idx_split'):
                s_dict = dataset.get_idx_split(seed=cfg.seed)
                dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
                train, val, test = dataset.split_idxs
                print(f"[Splits] sizes: train={len(train)}  val={len(val)}  test={len(test)}")                
                print(f"  train[:10]:\n{train[:10]}")
                print(f"  val[:10]:\n{val[:10]}")
                print(f"  test[:10]:\n{test[:10]}")
        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    elif format == 'PyG':
        # GraphGym default loader for Pytorch Geometric datasets
        dataset = load_pyg(name, dataset_dir)
    else:
        raise ValueError(f"Unknown data format: {format}")

    # Log a quick summary (safe for streaming datasets)
    log_loaded_dataset(dataset, format, name)

    # Positional encodings: precompute stats or attach on-the-fly
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and getattr(pecfg, 'enable', False):
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                if getattr(pecfg.kernel, 'times_func', None):
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: {pecfg.kernel.times}")

    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: {pe_enabled_list} for all graphs...")

        # Avoid dataset[:10] (not supported by all Datasets). Probe first k items.
        probe_n = min(10, len(dataset))
        is_undirected = all(dataset[i].is_undirected() for i in range(probe_n))
        logging.info(f"  ...estimated to be undirected: {is_undirected}")

        if not getattr(cfg.dataset, 'pe_transform_on_the_fly', False):
            pre_transform_in_memory(
                dataset,
                partial(
                    compute_posenc_stats,
                    pe_types=pe_enabled_list,
                    is_undirected=is_undirected,
                    cfg=cfg,
                ),
                show_progress=True,
                cfg=cfg,
                posenc_mode=True,
            )
            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")
        else:
            warnings.warn('PE transform on the fly to save memory consumption; experimental, please only use for RWSE/RWPSE')
            pe_transform = ComputePosencStat(
                pe_types=pe_enabled_list,
                is_undirected=is_undirected,
                cfg=cfg
            )
            if dataset.transform is None:
                dataset.transform = pe_transform
            else:
                dataset.transform = T.compose([pe_transform, dataset.transform])

    # If already prepared split indices, set them and skip generation.
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')
    else:
        raise RuntimeError("No split_idxs found and no fallback split generator configured.")

    return dataset

