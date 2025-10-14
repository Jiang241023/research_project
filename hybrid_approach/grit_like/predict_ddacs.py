import os, argparse, json
import torch
from torch_geometric.loader import DataLoader
from argparse import Namespace
import grit_like_framework  # noqa
from grit_like_framework.loader.dataset.ddacs_npy_stream import DDACSNPYStream
from torch_geometric.graphgym.config import set_cfg, load_cfg, cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.cmd_args import parse_args

def load_checkpoint(model, ckpt_path, device, strict=True):
    state = torch.load(ckpt_path, map_location=device)

    # Extract the model weights from common containers
    sd = None
    if isinstance(state, dict):
        for k in ("model_state", "model_state_dict", "state_dict", "model"):
            if k in state and isinstance(state[k], dict):
                sd = state[k]
                break
        # raw state-dict: {param_name: tensor}
        if sd is None and all(isinstance(v, torch.Tensor) for v in state.values()):
            sd = state
    if sd is None:
        raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}. "
                         f"Top-level keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")

    # Decide which module to load into:
    # - If keys already start with "model.", load into the outer GraphGymModule
    # - Else (keys like "encoder.*", "layers.*"), load into the inner model (model.model)
    keys = list(sd.keys())
    load_into_outer = any(k.startswith("model.") for k in keys)

    target = model if load_into_outer else model.model
    try:
        target.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        # Fall back to non-strict if minor mismatches (e.g., bn buffers) exist
        print(f"[warn] strict load failed ({e}); retrying with strict=False")
        target.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()
    return model

def infer_one_dir(data_dir,
                ckpt_path,
                cfg_path,
                out_dir,
                device='cuda:0'):
    # Load cfg and lock key settings for inference
    args = parse_args()
    set_cfg(cfg) 
    cfg.set_new_allowed(True)
    load_cfg(cfg, args)
    cfg.device = device
    cfg.dataset.dir = data_dir
    cfg.dataset.format = 'PyG-DDACSNPYStream'
    cfg.gnn.head = 'inductive_node'
    cfg.model.graph_pooling = 'none'
    cfg.share.dim_in = getattr(cfg.share, 'dim_in', 34)
    cfg.share.dim_out = 3  # force 3D node displacement

    # Build model and load weights
    model = create_model()
    model = load_checkpoint(model, ckpt_path, torch.device(device))

    # Build a simple PyG DataLoader over the folder
    ds = DDACSNPYStream(root=data_dir, debug=False) 
    loader = DataLoader(ds,  shuffle=False, num_workers=0)

    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            batch = batch.to(device)
            batch.split = 'test'
            # if not hasattr(batch, 'train_mask'):
            #     full = torch.ones(batch.x.size(0), dtype=torch.bool, device=batch.x.device)
            #     batch.train_mask = full
            #     batch.val_mask = full.clone()
            #     batch.test_mask = full.clone()
            pred, _true = model(batch)
            # slice per-graph and save using sample_id
            # batch.sample_id: [num_graphs_in_batch]
            # batch.batch: [sum_nodes] → graph index per node
            for g in range(int(batch.num_graphs)):
                sid = int(batch.sample_id[g].item() if batch.sample_id.dim() > 0 else batch.sample_id.item())
                node_mask = (batch.batch == g)
                pred_g = pred[node_mask].detach().cpu().numpy()

                out_np = os.path.join(out_dir, f"{sid}_pred_node_displacement.npy")
                import numpy as np
                np.save(out_np, pred_g)
                
                # optional sidecar JSON
                meta = {
                    "sample_id": sid,
                    "num_nodes": int(pred_g.shape[0]),
                    "pred_file": os.path.basename(out_np),
                }
                with open(os.path.join(out_dir, f"{sid}_meta.json"), "w") as f:
                    json.dump(meta, f)

    print(f"[✓] Saved predictions to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to training YAML")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint, e.g. results/ddacs-node-regression/41/ckpt/9.ckpt")
    ap.add_argument("--data", required=True, help="Folder of new graphs (same naming scheme).")
    ap.add_argument("--out", required=True, help="Where to save predictions.")
    ap.add_argument("--device", default="cuda:0")
    #ap.add_argument("--allow_missing_y", action="store_true")
    args = ap.parse_args()

    infer_one_dir(args.data, args.ckpt, args.cfg, args.out,
                #allow_missing_y=args.allow_missing_y, 
                device=args.device)


#python predict_ddacs.py --cfg /home/RUS_CIP/st186731/research_project/hybrid_approach/config_yaml/ddacs-node-regression.yaml --ckpt results/ddacs-node-regression/41/ckpt/9.ckpt --data /mnt/data/jiang --out results/ddacs-node-regression/preds_new