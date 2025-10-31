import logging
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader

from framework.loader.split_generator import set_dataset_splits
from framework.loader.dataset.ddacs_npy_stream import DDACSNPYStream


def _shape_any(x):
    """Return a tuple shape for numpy/torch, or None."""
    if x is None:
        return None
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if hasattr(x, "size"):
        # torch.Tensor path
        try:
            return tuple(x.size())
        except Exception:
            return None
    return None


def log_loaded_dataset(dataset, fmt, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{fmt}':")

    # Peek sample[0] (safe for streaming datasets)
    try:
        d0 = dataset[0]
        x_s   = _shape_any(getattr(d0, "x", None))
        pos_s = _shape_any(getattr(d0, "pos", None))
        ei_s  = _shape_any(getattr(d0, "edge_index", None))
        y_s   = _shape_any(getattr(d0, "y", None))
        ea_s  = _shape_any(getattr(d0, "edge_attr", None))
        eei_t = getattr(d0, "edge_edge_index", getattr(d0, "eei", None))
        eei_s = _shape_any(eei_t)

        logging.info(
            "sample[0]: Data("
            f"new_concatenated_features={x_s}, "
            f"node_coords={pos_s}, "
            f"edge_index={ei_s}, "
            f"node_displacement={y_s}, "
            f"edge_features={ea_s}, "
            f"edge_edge_index={eei_s})"
        )
    except Exception as e:
        logging.info(f"sample[0]: <unavailable due to: {e}>")

    logging.info(f"  num graphs: {len(dataset)}")

    # Quick averages (nodes & edges) from a small prefix
    try:
        k = min(16, len(dataset))
        nn, ne = [], []
        for i in range(k):
            di = dataset[i]
            # nodes
            if hasattr(di, "num_nodes") and di.num_nodes is not None:
                nn.append(int(di.num_nodes))
            elif getattr(di, "x", None) is not None:
                nn.append(int(di.x.size(0)))
            else:
                nn.append(0)
            # edges
            if getattr(di, "edge_index", None) is not None:
                ne.append(int(di.edge_index.size(1)))
            else:
                ne.append(0)
        avg_nodes = int(round(sum(nn) / max(1, len(nn))))
        avg_edges = int(round(sum(ne) / max(1, len(ne))))
        logging.info(f"  approx avg num_nodes/graph: {avg_nodes}")
        logging.info(f"  approx avg num_edges/graph: {avg_edges}")
    except Exception as e:
        logging.info(f"  approx avg num_nodes/edges: n/a ({e})")


@register_loader("custom_master_loader")
def load_dataset_master(fmt, name, dataset_dir):
    """
    Single dataset entry point.
    Use in YAML: dataset.format: PyG-DDACSNPYStream
    """
    if not fmt.startswith("PyG-"):
        raise ValueError(f"Unknown data format: {fmt}")

    pyg_dataset_id = fmt.split("-", 1)[1]
    print(f"pyg_dataset_id (from load_dataset_master):{pyg_dataset_id}")

    if pyg_dataset_id != "DDACSNPYStream":
        raise ValueError(
            f"Unexpected PyG Dataset identifier: {pyg_dataset_id}. "
            "Use 'PyG-DDACSNPYStream'."
        )

    # Instantiate dataset (returns LineGraphData and sets num_edges/num_nodes)
    dataset = DDACSNPYStream(
        root=dataset_dir,
        max_samples=getattr(cfg.dataset, "max_samples", None),
        debug=getattr(cfg, "debug", False),
    )

    # Build & apply splits
    if not hasattr(dataset, "get_idx_split"):
        raise RuntimeError("Dataset has no get_idx_split() to generate splits.")

    s = dataset.get_idx_split(seed=cfg.seed)  # {'train','val','test'}
    splits = [s["train"], s["val"], s["test"]]
    set_dataset_splits(dataset, splits)

    # Print split summary and a sanity check sample from each
    print(len(dataset))
    train, val, test = splits
    print(f"[Splits] sizes: train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"train[:10]:\n{dataset.data.train_graph_index[:10]}")
    print(f"val[:10]:\n{dataset.data.val_graph_index[:10]}")
    print(f"test[:10]:\n{dataset.data.test_graph_index[:10]}")

    idx_train = int(dataset.data.train_graph_index[0])
    idx_val = int(dataset.data.val_graph_index[0])
    idx_test = int(dataset.data.test_graph_index[0])

    s_train, s_val, s_test = dataset[idx_train], dataset[idx_val], dataset[idx_test]

    print("TRAIN → idx:", idx_train, "sample_id:", getattr(s_train, "sample_id", None))
    print("VAL   → idx:", idx_val, "sample_id:", getattr(s_val, "sample_id", None))
    print("TEST  → idx:", idx_test, "sample_id:", getattr(s_test, "sample_id", None))

    for tag, d in [("train", s_train), ("val", s_val), ("test", s_test)]:
        x_shape   = None if getattr(d, "x", None) is None else tuple(d.x.shape)
        pos_shape = None if getattr(d, "pos", None) is None else tuple(d.pos.shape)
        y_shape   = None if getattr(d, "y", None) is None else tuple(d.y.shape)
        E         = None if getattr(d, "edge_index", None) is None else d.edge_index.size(1)
        M         = None
        eei_t     = getattr(d, "edge_edge_index", getattr(d, "eei", None))
        if eei_t is not None:
            M = eei_t.size(1) if eei_t.dim() == 2 else None

        print(
            f"{tag}: new_concatenated_features={x_shape}, "
            f"node_coords={pos_shape}, node_displacement={y_shape}, "
            f"edge_index(E)={E}, edge_edge_index(M)={M}, "
            f"num_nodes={getattr(d, 'num_nodes', None)}, "
            f"num_edges={getattr(d, 'num_edges', None)}"
        )

        # Light sanity check: num_edges should match E
        if getattr(d, "num_edges", None) is not None and E is not None:
            if int(d.num_edges) != int(E):
                logging.warning(
                    f"[{tag}] num_edges ({int(d.num_edges)}) != edge_index.size(1) ({int(E)}). "
                    "Batching of edge_edge_index may break."
                )

    # Log a quick summary (safe for streaming datasets)
    log_loaded_dataset(dataset, fmt, name)
    return dataset
