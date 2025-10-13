import os, glob, os.path as osp, time, signal, traceback 
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

# Get nice memory stats
try:
    import psutil
except Exception:
    psutil = None

# Return resident set size (RSS) in MB
def _read_rss_mb():
    """Return current process RSS in MB without extra deps."""
    if psutil is not None:
        return int(psutil.Process().memory_info().rss / (1024*1024))
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # e.g., "VmRSS:\t  123456 kB"
                    kb = int(line.split()[1])
                    return kb // 1024
    except Exception:
        pass
    return -1  # unknown


class DDACSNPYInMem(InMemoryDataset):
    # Extends InMemoryDataset: keeps all graphs in RAM and in a single tensor store on disk.
    """
    In-memory PyG dataset that reads per-sample .npy files from a directory
    and collates them into a single `data/slices` storage so that
    `dataset.data` exists (required by the existing logger).

    Expects files per sample id like:
      {id}_new_concatenated_features.npy   -> (N, F_x) float32  -> data.x
      {id}_edge_index.npy                  -> (E, 2)   int64    -> data.edge_index
      {id}_edge_features.npy               -> (E, F_e) float32  -> data.edge_attr
      {id}_node_coords.npy                 -> (N, 3)   float32  -> data.pos  (optional but used)
      {id}_node_displacement.npy           -> (N, 3)   float32  -> data.y    (node-level regression)
    """

    def __init__(self, root, transform=None, pre_transform=None,
                 new_concatenated_features_suffix="new_concatenated_features",
                 node_displacement_suffix="node_displacement",
                 edge_index_suffix="edge_index_2",
                 edge_feat_suffix="edge_features",
                 node_coords_suffix="node_coords",
                 manifest_name="ddacs_manifest.txt",
                 max_samples=5000,   #  limit how many samples to load
                 debug=False):
        self.data_dir = root                     

        self.new_concatenated_features_suffix = new_concatenated_features_suffix
        self.node_displacement_suffix = node_displacement_suffix
        self.edge_index_suffix = edge_index_suffix
        self.edge_feat_suffix = edge_feat_suffix
        self.node_coords_suffix = node_coords_suffix
        self._manifest = manifest_name
        self.max_samples = max_samples
        self.debug = bool(debug or os.getenv("DEBUG_DDACS", ""))

        # Install a SIGTERM handler so OOM killer / scheduler kill prints a last line
        def _sigterm_handler(signum, frame):
            print(f"[DDACS][KILLED] Received signal {signum}. Last known Resident set memories{_read_rss_mb()} MB")
            # flush stdio quickly
            try: import sys; sys.stdout.flush(); sys.stderr.flush()
            except Exception: pass
        try:
            signal.signal(signal.SIGTERM, _sigterm_handler)
        except Exception:
            pass

        t0 = time.time()
        if self.debug:
            print(f"[DDACS] __init__ start | root={root} | debug=1")   

        super().__init__(root, transform, pre_transform)

        if self.debug:
            print(f"[DDACS] Loading processed tensors from: {self.processed_paths[0]}")
        
        # After process(), load processed tensor stores:
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("self.data: {self.data.shape}")
        print("self.slices {self.slices}")
        if self.debug:
            for k, v in self.slices.items():
                print(f"[DDACS] slices[{k}].shape = {tuple(v.shape)}  first={v[:min(5, v.numel())]}")



        # GOOD debug (self.data has no .shape):
        if self.debug:
            print(f"[DDACS] processed_path: {self.processed_paths[0]}")
            print(f"[DDACS] graphs={self.len()} | Resident set memories ={_read_rss_mb()} MB")

        if self.debug:
            n_graphs = self.len()
            nf = int(self.data.x.size(-1)) if self.data.x is not None else 0
            ef = int(self.data.edge_attr.size(-1)) if self.data.edge_attr is not None else 0
            print(f"[DDACS] Loaded processed data: graphs={n_graphs}, "
                  f"node_feat_dim={nf}, edge_feat_dim={ef}, "
                  f"resident set size = {_read_rss_mb()} MB, took {time.time()-t0:.2f}s")

    # Tiny helper: convenience to keep logs gated on the flag 
    def _dbg(self, msg):
        if self.debug:
            print(f"[DDACS] {msg}")

    @property
    def raw_file_names(self):
        # Use a small "manifest" file to satisfy PyG's raw/processed checks.
        return self._manifest

    @property
    def processed_file_names(self):
        # different caches per cap
        return f"data_max{self.max_samples}.pt"

    # Scan root dir for files ending with _{feature_suffix}.npy and extracts {id}.
    def _discover_ids(self):
        patt = osp.join(self.data_dir, f"*_{self.new_concatenated_features_suffix}.npy")
        ids = sorted(osp.basename(p).split("_")[0] for p in glob.glob(patt))
        if len(ids) == 0:
            raise FileNotFoundError(
                f"No '*_{self.new_concatenated_features_suffix}.npy' files found under {self.data_dir}"
            )
        if self.max_samples is not None:
            ids = ids[:int(self.max_samples)]
        return ids

    #  create a manifest so raw_file_names "exists" 
    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        ids = self._discover_ids()
        with open(osp.join(self.raw_dir, self._manifest), "w") as f:
            f.write("\n".join(ids))
        self._dbg(f"Manifest written with {len(ids)} ids at {osp.join(self.raw_dir, self._manifest)}")

    # Load everything and collate
    def process(self):
        self._dbg(f"process() begin | raw_dir={self.raw_dir} | processed_dir={self.processed_dir}")
        man_path = osp.join(self.raw_dir, self._manifest)

        # Ensure the manifest exists
        if not osp.exists(man_path):
            # If user deleted raw/, rebuild manifest on the fly.
            self._dbg("Manifest missing; calling download() to rebuild")
            self.download()

        with open(man_path) as fh:
            ids = [line.strip() for line in fh if line.strip()]
        self._dbg(f"Found {len(ids)} sample ids from manifest")
        if self.max_samples is not None:
            ids = ids[:int(self.max_samples)]
        data_list = []
        t_all0 = time.time()
        for idx, sid in enumerate(ids, 1):
            t0 = time.time()

            def npy(suffix): return osp.join(self.data_dir, f"{sid}_{suffix}.npy")

            try:
                #load arrays (mmap to reduce peak RSS)
                new_concatenated_features_suffix_path = npy(self.new_concatenated_features_suffix)
                edge_index_path = npy(self.edge_index_suffix)
                edge_feat_path = npy(self.edge_feat_suffix)
                node_coords_path = npy(self.node_coords_suffix)
                node_displacement_path = npy(self.node_displacement_suffix)
                
                # mmap_mode="r" keeps NumPy arrays read-only and avoids loading entire arrays eagerly, reducing peak RAM during reading.
                new_concatenated_features = np.load(new_concatenated_features_suffix_path, mmap_mode="r")
                edge_index = np.load(edge_index_path, mmap_mode="r")
                node_coords = np.load(node_coords_path, mmap_mode="r")
                node_displacement = np.load(node_displacement_path, mmap_mode="r")
                edge_feat = np.load(edge_feat_path, mmap_mode="r")

                self._dbg(
                    f"#{idx}/{len(ids)} sid={sid} "
                    f"| new_concatenated_features{tuple(new_concatenated_features.shape)} {new_concatenated_features.dtype} "
                    f"| edge_index{tuple(edge_index.shape)} {edge_index.dtype} "
                    f"| edge_feat{tuple(edge_feat.shape) if edge_feat is not None else '(missing)'} "
                    f"| node_coords{tuple(node_coords.shape)} {node_coords.dtype} "
                    f"| node_displacement{tuple(node_displacement.shape)} {node_displacement.dtype} "
                    f"| Resident set memories={_read_rss_mb()} MB"
                )

                # # Convert to tensors
                # new_concatenated_features_t = torch.from_numpy(new_concatenated_features)
                # edge_index_t = torch.from_numpy(edge_index).t().contiguous()  # (2, E)
                # edge_feat_t  = torch.from_numpy(edge_feat)
                # node_coords_t     = torch.from_numpy(node_coords)
                # node_displacement_t     = torch.from_numpy(node_displacement)

                # # Build Data
                # d = Data(new_concatenated_features=new_concatenated_features_t, 
                #          edge_index=edge_index_t, 
                #          edge_feat=edge_feat_t, 
                #          node_coords=node_coords_t, 
                #          node_displacement=node_displacement_t)
                
                # # Many files use (x, edge_attr, pos, y), so I add these lines
                # d.x = d.new_concatenated_features
                # d.edge_attr = d.edge_feat
                # d.pos = d.node_coords
                # d.y = d.node_displacement
                # d.num_nodes = new_concatenated_features_t.size(0)
                # d.sample_id = sid

                # --- convert numpy -> torch (types consistent) ---
                x_t   = torch.from_numpy(new_concatenated_features.astype(np.float32, copy=False))
                ei_t  = torch.from_numpy(edge_index.astype(np.int64, copy=False)).t().contiguous()
                ea_np = edge_feat if edge_feat is not None else np.zeros((edge_index.shape[0], 1), np.float32)
                ea_t  = torch.from_numpy(ea_np.astype(np.float32, copy=False))
                pos_t = torch.from_numpy(node_coords.astype(np.float32, copy=False))
                y_t   = torch.from_numpy(node_displacement.astype(np.float32, copy=False))

                # --- build Data with ONLY canonical keys ---
                d = Data(x=x_t, edge_index=ei_t, edge_attr=ea_t, pos=pos_t, y=y_t)
                d.num_nodes = x_t.size(0)

                # (Optional) If you want to keep an id, use an INT TENSOR (not a string):
                try:
                    sid_int = int(sid)
                except Exception:
                    sid_int = idx
                d.sid = torch.tensor([sid_int], dtype=torch.long)

                if self.pre_transform is not None:
                    d = self.pre_transform(d)

                data_list.append(d)

                self._dbg(f"sid={sid} appended | nodes={d.num_nodes} | edges={edge_index.shape[0]} "
                          f"| elapsed={time.time()-t0:.2f}s | Resident set memories={_read_rss_mb()} MB")

            except Exception as e:
                self._dbg(f"[ERROR] sid={sid} failed: {e}\n{traceback.format_exc()}")
                raise  # re-raise so you see it in the main log

            # Optional: print every k samples to avoid spam if debug via env
            if not self.debug and (idx % 100 == 0):
                print(f"[DDACS] processed {idx}/{len(ids)} | Resident set memories={_read_rss_mb()} MB")

        # collate converts the Python list into the efficient big tensor store + index slices that InMemoryDataset uses.
        self._dbg(f"Collating {len(data_list)} graphs … | Resident set memories={_read_rss_mb()} MB")
        t_coll0 = time.time()
        data, slices = self.collate(data_list)
        self._dbg(f"Collate done in {time.time()-t_coll0:.2f}s | Resident set memories={_read_rss_mb()} MB")

        os.makedirs(self.processed_dir, exist_ok=True)
        t_save0 = time.time()
        torch.save((data, slices), self.processed_paths[0])
        self._dbg(f"Saved processed to {self.processed_paths[0]} in {time.time()-t_save0:.2f}s")

        self._dbg(f"process() end | total {time.time()-t_all0:.2f}s | Resident set memories={_read_rss_mb()} MB")

    # Return indices for splitting graphs, not nodes.
    def get_idx_split(self, split_seed=42, train_fraction=0.7, val_fraction=0.2):
        rng = np.random.default_rng(split_seed)
        n = self.len()
        perm = rng.permutation(n)
        n_train = int(n * train_fraction)
        n_val = int(n * val_fraction)
        idx_train = torch.as_tensor(perm[:n_train], dtype=torch.long)
        idx_val   = torch.as_tensor(perm[n_train:n_train+n_val], dtype=torch.long)
        idx_test  = torch.as_tensor(perm[n_train+n_val:], dtype=torch.long)
        return {'train': idx_train, 'val': idx_val, 'test': idx_test}

    # Read from the collated store to report feature dims.
    @property
    def num_node_features(self):
        # new_concatenated_features = self.data.new_concatenated_features
        # return int(new_concatenated_features.size(-1)) if new_concatenated_features is not None else 0
        x = getattr(self.data, 'x', None)
        return int(x.size(-1)) if x is not None else 0
    @property
    def num_edge_features(self):
        # edge_feat = self.data.edge_feat
        # return int(edge_feat.size(-1)) if edge_feat is not None else 0
        ea = getattr(self.data, 'edge_attr', None)
        return int(ea.size(-1)) if ea is not None else 0

if __name__ == '__main__':
    ds = DDACSNPYInMem(root="/mnt/data/jiang",max_samples=5000,debug=False)
    print(len(ds), ds.num_node_features, ds.num_edge_features)
    g = ds[0]  # a PyG Data with x, edge_index, edge_attr, pos, y

