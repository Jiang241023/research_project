from torch_geometric.graphgym.config import cfg
from torch_geometric.transforms import Compose
from framework.transform.posenc_stats import ComputePosencStat
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path

#  Constants 
NPZ_SUFFIX = ".npz"


# - Utilities
def _list_ids(root: str | Path):
    """
    Return sorted numeric IDs from NPZ bundles named '<ID>.npz'.
    """
    root = Path(root)
    ids = []
    for path in root.glob(f"*{NPZ_SUFFIX}"):
        stem = path.stem  # filename without .npz
        if stem.isdigit():
            ids.append(int(stem))
    ids.sort()
    return ids


def npz_path(root: str | Path, sid: int | str) -> Path:
    return Path(root) / f"{sid}{NPZ_SUFFIX}"


def ensure_writable(arr: np.ndarray) -> np.ndarray:
    """Guarantee a writable ndarray (PyTorch warns on read-only)."""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if not arr.flags.writeable:
        arr = np.array(arr, copy=True)
    return arr


def to_long_2x(t: torch.Tensor) -> torch.Tensor:
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


#  Data subclass that knows how to batch line-graph indices 
class LineGraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('edge_index', 'edge_edge_index', 'eei'):
            return 1  # concatenate by columns
        if key == 'edge_index_2' and isinstance(value, torch.Tensor) and value.dim() == 2:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # Node graph edges → increment by num_nodes
        if key == 'edge_index':
            inc = int(self.num_nodes)
            if isinstance(value, torch.Tensor):
                if value.dim() == 2:
                    return value.new_full((value.size(0), 1), inc)
                else:
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

        # Optional: an index named 'edge_index_2'
        if key == 'edge_index_2':
            inc = int(self.num_edges)
            if isinstance(value, torch.Tensor):
                if value.dim() == 2:
                    return value.new_full((value.size(0), 1), inc)
                else:
                    return inc
            return inc

        return super().__inc__(key, value, *args, **kwargs)


# Dataset (NPZ only)
class DDACSNPYStream(Dataset):
    """
    Streaming PyG dataset for DDACS NPZ exports (one compressed bundle per ID).

    Expects per-ID files: '{id}.npz' with keys:
      - new_concatenated_features: (N,34) float32
      - node_displacement:        (N,3)  float32
      - edge_index:               (E,2) or (2,E) int64
      - edge_features:            (E,31) float32
      - node_coords:              (N,3)  float32
      - edge_edge_index:          (M,2) or (2,M) int64
      Optional:
      - node_index:               (N,)   int64
      - edge_index_2:             (E,) or (E,2)/(2,E)
    """
    def __init__(self, root, max_samples=None, debug=False,
                 transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.debug = bool(debug)
        self.ids = _list_ids(root)
        if max_samples is not None:
            self.ids = self.ids[:int(max_samples)]
        if self.debug:
            print(f"[DDACSNPYStream/NPZ] Streaming from {root}; samples: {len(self.ids)}")

        # Minimal attrs GraphGym sometimes probes
        self.data = Data()
        self._data = self.data
        self.name = "DDACSNPYStream"
        # Attach RRWP pre-transform if enabled in cfg and not already set
        if getattr(cfg, "posenc_RRWP", None) and getattr(cfg.posenc_RRWP, "enable", False):
            rrwp_t = ComputePosencStat(pe_types=['RRWP'], is_undirected=True, cfg=cfg)
            if self.pre_transform is None:
                self.pre_transform = rrwp_t
            else:
                # run existing pre_transform first, then RRWP
                self.pre_transform = Compose([self.pre_transform, rrwp_t])

    def len(self):
        return len(self.ids)

    def load_from_npz(self, sid: int) -> LineGraphData:
        npz = npz_path(self.root, sid)
        if self.debug:
            print(f"[DDACSNPYStream/NPZ] Loading: {npz}")
        if not npz.exists():
            raise FileNotFoundError(f"Expected NPZ bundle not found: {npz}")

        with np.load(npz) as z:
            # Required keys
            X   = ensure_writable(z["new_concatenated_features"]).astype(np.float32, copy=False)
            POS = ensure_writable(z["node_coords"]).astype(np.float32, copy=False)
            Y   = ensure_writable(z["node_displacement"]).astype(np.float32, copy=False)
            EI  = ensure_writable(z["edge_index"]).astype(np.int64,   copy=False)
            EF  = ensure_writable(z["edge_features"]).astype(np.float32, copy=False)
            EEI = ensure_writable(z["edge_edge_index"]).astype(np.int64, copy=False)

            # Optional keys
            NODE_IDX = z["node_index"] if "node_index" in z else None
            EI_2     = z["edge_index_2"] if "edge_index_2" in z else None
            if NODE_IDX is not None:
                NODE_IDX = ensure_writable(NODE_IDX).astype(np.int64, copy=False)
            if EI_2 is not None:
                EI_2 = ensure_writable(EI_2)  # dtype/shape handled below

        # Tensor conversions
        x   = torch.from_numpy(X)           # (N,34)
        pos = torch.from_numpy(POS)         # (N,3)
        y   = torch.from_numpy(Y)           # (N,3)
        edge_attr  = torch.from_numpy(EF).to(torch.float32)   # (E,31)

        edge_index = to_long_2x(torch.from_numpy(EI))        # (2,E)
        eei        = to_long_2x(torch.from_numpy(EEI))       # (2,M)

        d = LineGraphData()
        d.x = x
        d.pos = pos
        d.y = y
        d.edge_index = edge_index
        d.edge_attr  = edge_attr
        d.edge_edge_index = eei
        d.eei = d.edge_edge_index  # alias

        # Optional extras
        if EI_2 is not None:
            t = torch.from_numpy(EI_2)
            # accept (E,), (E,2) or (2,E)
            if t.dim() == 2 and t.size(-1) == 2:
                t = t.t().contiguous()
            d.edge_index_2 = t.long()
        if NODE_IDX is not None:
            d.node_index = torch.from_numpy(NODE_IDX).long()

        # Counts for batching offsets
        d.num_nodes = int(x.size(0))
        d.num_edges = int(edge_index.size(1))
        d.sample_id = int(sid)
        return d

    def get(self, index):
        sid = self.ids[index]
        d = self.load_from_npz(sid)

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
