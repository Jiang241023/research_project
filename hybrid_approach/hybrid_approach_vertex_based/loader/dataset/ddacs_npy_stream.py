import os, glob
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from pathlib import Path

ID_SUFFIX = "_new_concatenated_features.npy"

def _list_ids(root):
    """
    Return sorted numeric IDs from files named '<ID>_new_concatenated_features.npy'
    in `root`. Example: '16039_new_concatenated_features.npy' -> 16039.
    """
    ids = [] # list[int]
    for path in Path(root).glob(f"*{ID_SUFFIX}"): # "*_new_concatenated_features.npy" means:"give me all files whose names end with _new_concatenated_features.npy (with anything—or nothing—before it)."
        id_str = path.name[:-len(ID_SUFFIX)]  # everything before the fixed suffix
        if id_str.isdigit():                  # only accept pure numeric IDs
            ids.append(int(id_str))        
    ids.sort()
    return ids

def _paths_for_id(root, sid):
    sid = str(sid)
    return {
        "x":            os.path.join(root, f"{sid}_new_concatenated_features.npy"),
        "pos":          os.path.join(root, f"{sid}_node_coords.npy"),
        "y":            os.path.join(root, f"{sid}_node_displacement.npy"),
        "edge_index":   os.path.join(root, f"{sid}_edge_index.npy"),
        "edge_attr":    os.path.join(root, f"{sid}_edge_features.npy"),
        "edge_index_2": os.path.join(root, f"{sid}_edge_index_2.npy"),
        "node_index":   os.path.join(root, f"{sid}_node_index.npy"),
    }

class DDACSNPYStream(Dataset):
    """
    DDACSNPYStream is for getting the dataset in the path /mnt/data/jiang
    """
    def __init__(self, root, max_samples=None, debug=False,
                 transform=None, pre_transform=None):                       
        super().__init__(root, transform, pre_transform)
        self.debug = debug                         
        self.ids = _list_ids(root)
        if max_samples is not None:
            self.ids = self.ids[:max_samples]
        if self.debug:
            print(f"[DDACSNPYStream] Streaming from {root}; samples: {len(self.ids)}")

        # minimal attrs GraphGym sometimes probes
        self.data = Data()
        self._data = self.data
        # self.slices = None
        self.name = "DDACSNPYStream"

    def len(self):
        return len(self.ids)

    def get(self, index):
        sid = self.ids[index]
        P = _paths_for_id(self.root, sid)

        x   = torch.from_numpy(np.load(P["x"]).astype(np.float32))
        pos = torch.from_numpy(np.load(P["pos"]).astype(np.float32))
        
        y = torch.from_numpy(np.load(P["y"]).astype(np.float32))

        ei  = torch.from_numpy(np.load(P["edge_index"]).astype(np.int64))
        ea  = torch.from_numpy(np.load(P["edge_attr"]).astype(np.float32))

        d = Data(
            x=x,
            pos=pos,
            y=y,
            edge_index=ei.t().contiguous(),
            edge_attr=ea,
        )
        if os.path.exists(P["edge_index_2"]):
            ei2 = torch.from_numpy(np.load(P["edge_index_2"]).astype(np.int64))
            d.edge_index_2 = ei2.t().contiguous()
        if os.path.exists(P["node_index"]):
            d.node_index = torch.from_numpy(np.load(P["node_index"]).astype(np.int64))

        d.num_nodes = x.shape[0]
        d.sample_id = int(sid)

        # (optional) default masks so node head can index without errors
        full = torch.ones(d.num_nodes, dtype=torch.bool)
        d.train_mask = full
        d.val_mask = full.clone()
        d.test_mask = full.clone()

        if self.pre_transform is not None:
            d = self.pre_transform(d)
        return d

    def get_idx_split(self, seed: int = 42, ratios=(0.7, 0.2, 0.1)):
        idx = np.arange(len(self.ids))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(ratios[0] * n))
        n_val   = int(round(ratios[1] * n))
        train = idx[:n_train].tolist()
        val   = idx[n_train:n_train + n_val].tolist()
        test  = idx[n_train + n_val:].tolist()
        return {'train': train, 'val': val, 'test': test}