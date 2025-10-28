import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path

# ---------- Constants ----------
ID_SUFFIX = "_new_concatenated_features.npy"


# ---------- Utilities ----------
def _list_ids(root):
    """
    Return sorted numeric IDs from files named '<ID>_new_concatenated_features.npy'
    in `root`. Example: '16039_new_concatenated_features.npy' -> 16039.
    """
    ids = []
    for path in Path(root).glob(f"*{ID_SUFFIX}"):
        id_str = path.name[:-len(ID_SUFFIX)]
        if id_str.isdigit():
            ids.append(int(id_str))
    ids.sort()
    return ids


def _paths_for_id(root, sid):
    sid = str(sid)
    return {
        "x":                 os.path.join(root, f"{sid}_new_concatenated_features.npy"),
        "pos":               os.path.join(root, f"{sid}_node_coords.npy"),
        "y":                 os.path.join(root, f"{sid}_node_displacement.npy"),
        "edge_index":        os.path.join(root, f"{sid}_edge_index.npy"),
        "edge_attr":         os.path.join(root, f"{sid}_edge_features.npy"),
        "edge_index_2":      os.path.join(root, f"{sid}_edge_index_2.npy"),
        "node_index":        os.path.join(root, f"{sid}_node_index.npy"),
        "edge_edge_index":   os.path.join(root, f"{sid}_edge_edge_index.npy"),
    }


def _np_load(path, dtype=None, mmap=True):
    """Load .npy with optional dtype cast and memory map."""
    arr = np.load(path, mmap_mode="r") if mmap else np.load(path)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _to_long_2x(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure an index tensor is shape (2, M) and dtype long.
    Accepts (M, 2) or (2, M); returns (2, M).
    """
    if t.dtype != torch.long:
        t = t.long()
    if t.dim() != 2:
        raise ValueError(f"Index must be 2D, got shape {tuple(t.size())}")
    if t.size(0) == 2:
        return t.contiguous()
    if t.size(1) == 2:
        return t.t().contiguous()
    raise ValueError(f"Index must be (2, M) or (M, 2), got {tuple(t.size())}")


# ---------- Data subclass that knows how to batch line-graph indices ----------
class LineGraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('edge_index', 'edge_edge_index', 'eei'):
            return 1  # concatenate by columns
        # (optional) if you also carry an index like 'edge_index_2'
        if key == 'edge_index_2' and isinstance(value, torch.Tensor) and value.dim() == 2:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # Node graph edges → increment by num_nodes
        if key == 'edge_index':
            inc = int(self.num_nodes)
            if isinstance(value, torch.Tensor):
                if value.dim() == 2:
                    # column vector so it broadcasts across E
                    return value.new_full((value.size(0), 1), inc)
                else:  # unusual, but keep safe
                    return value.new_full((value.size(0),), inc)
            return inc

        # Line-graph edges (edge→edge) → increment by num_edges
        if key in ('edge_edge_index', 'eei'):
            inc = int(self.num_edges)
            if isinstance(value, torch.Tensor):
                if value.dim() == 2:
                    return value.new_full((value.size(0), 1), inc)
                else:
                    return value.new_full((value.size(0),), inc)
            return inc

        # (optional) if you also batch an index named 'edge_index_2'
        if key == 'edge_index_2':
            inc = int(self.num_edges)
            if isinstance(value, torch.Tensor):
                if value.dim() == 2:
                    return value.new_full((value.size(0), 1), inc)
                else:  # 1D placeholder like arange(E)
                    return inc
            return inc

        return super().__inc__(key, value, *args, **kwargs)


# ---------- Dataset ----------
class DDACSNPYStream(Dataset):
    """
    Streaming PyG dataset for DDACS NPY exports (no processed/ folder).
    Expects per-ID .npy files in `root` as finalized in your pipeline:
      • {id}_new_concatenated_features.npy  -> x: (N, 34) float32
      • {id}_node_coords.npy               -> pos: (N, 3) float32
      • {id}_node_displacement.npy         -> y: (N, 3) float32
      • {id}_edge_index.npy                -> edge_index: (E,2) or (2,E) int64
      • {id}_edge_features.npy             -> edge_attr: (E,31) float32
      • {id}_edge_edge_index.npy           -> eei/edge_edge_index: (M,2) or (2,M) int64
      Optional:
      • {id}_edge_index_2.npy              -> custom; kept if present
      • {id}_node_index.npy                -> (N,) int64
    """
    def __init__(self, root, max_samples=None, debug=False,
                 transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.debug = bool(debug)
        self.ids = _list_ids(root)
        if max_samples is not None:
            self.ids = self.ids[:int(max_samples)]
        if self.debug:
            print(f"[DDACSNPYStream] Streaming from {root}; samples: {len(self.ids)}")

        # Minimal attrs GraphGym sometimes probes
        self.data = Data()
        self._data = self.data
        self.name = "DDACSNPYStream"

    def len(self):
        return len(self.ids)

    def get(self, index):
        sid = self.ids[index]
        P = _paths_for_id(self.root, sid)

        # Load arrays (memory-mapped to be light on RAM)
        x   = torch.from_numpy(_np_load(P["x"],   np.float32))
        pos = torch.from_numpy(_np_load(P["pos"], np.float32))
        y   = torch.from_numpy(_np_load(P["y"],   np.float32))

        ei_np  = _np_load(P["edge_index"],      np.int64)
        ea_np  = _np_load(P["edge_attr"],       np.float32)
        eei_np = _np_load(P["edge_edge_index"], np.int64)

        # Normalize shapes/dtypes for indices
        edge_index = _to_long_2x(torch.as_tensor(ei_np))     # (2, E)
        edge_attr  = torch.as_tensor(ea_np, dtype=torch.float32)  # (E, 31)
        eei        = _to_long_2x(torch.as_tensor(eei_np))    # (2, M)

        # Build LineGraphData (not plain Data!)
        d = LineGraphData()
        d.x = x
        d.pos = pos
        d.y = y
        d.edge_index = edge_index
        d.edge_attr  = edge_attr
        d.edge_edge_index = eei
        d.eei = d.edge_edge_index  # alias for convenience

        # Optional extras
        if os.path.exists(P["edge_index_2"]):
            ei2_np = _np_load(P["edge_index_2"], np.int64)
            t = torch.as_tensor(ei2_np)
            if t.dim() == 2 and t.size(-1) == 2:
                t = t.t().contiguous()
            d.edge_index_2 = t.long()
        if os.path.exists(P["node_index"]):
            d.node_index = torch.from_numpy(_np_load(P["node_index"], np.int64)).long()

        # IMPORTANT: expose counts so __inc__ knows how to offset batched graphs
        d.num_nodes = int(x.size(0))
        d.num_edges = int(edge_index.size(1))  # number of edge tokens (E)
        d.sample_id = int(sid)

        # Pre-transform (if any)
        if self.pre_transform is not None:
            d = self.pre_transform(d)

        # Transform (if any)
        if self.transform is not None:
            d = self.transform(d)

        return d

    def get_idx_split(self, seed, ratios=(0.7, 0.2, 0.1)):
        """
        Deterministic split by contiguous ranges, with per-subset shuffles for reproducibility.
        """
        n = len(self.ids)
        n_train = int(ratios[0] * n)
        n_val   = int(ratios[1] * n)
        n_test  = n - n_train - n_val

        train = np.arange(0, n_train)
        val   = np.arange(n_train, n_train + n_val)
        test  = np.arange(n_train + n_val, n_train + n_val + n_test)

        rng_train = np.random.default_rng(seed)
        rng_val   = np.random.default_rng(seed + 1)

        rng_train.shuffle(train)
        rng_val.shuffle(val)

        return {'train': train, 'val': val, 'test': test}
