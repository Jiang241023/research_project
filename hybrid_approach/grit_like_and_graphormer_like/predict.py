import os, argparse, json
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from argparse import Namespace

import framework  
from framework.loader.dataset.ddacs_npy_stream import DDACSNPYStream
from torch_geometric.graphgym.config import set_cfg, load_cfg, cfg
from torch_geometric.graphgym.model_builder import create_model
import time

# Load checkpoint
def load_checkpoint(model, ckpt_path, device, strict=True):

    # Load the checkpoint file into the memory
    state = torch.load(ckpt_path, map_location=device, weights_only=True) # map_location=device ensures tensors inside the checkpoint are remapped directly onto the device
    #print(f"state: {state}")    

    # Extract the state dictionary
    sd = None
    if isinstance(state, dict):
        for k in ("model_state", "model_state_dict", "state_dict", "model"):
            if k in state and isinstance(state[k], dict):
                sd = state[k]
                break
        if sd is None and all(isinstance(v, torch.Tensor) for v in state.values()):
            sd = state
    if sd is None:
        raise ValueError(f"Unrecognized checkpoint: {ckpt_path}")

    # Decide where to load the checkpoints weights
    keys = list(sd.keys())
    #print(f"keys:{keys}")
    load_into_outer = any(k.startswith("model.") for k in keys)
    target = model if load_into_outer else model.model

    try:
        # Load these tensors (sd) into this module (target) by matching names of parameters/buffers
        target.load_state_dict(sd, strict=strict)
    except RuntimeError as e:
        print(f"[warn] strict load failed ({e}); retrying with strict=False")
        target.load_state_dict(sd, strict=False)

    # Moves the whole module (all parameters and buffers) onto the target device
    model.to(device).eval()
    return model

# Inference
def infer(data_dir, ckpt_path, cfg_path, out_dir, device='cuda:0', batch_size=8):
    # Build model from YAML 
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg,  Namespace(cfg_file=cfg_path, opts=[])) # load_cfg is written to accept something that looks like args = argparse.Namespace(...)
    model = create_model()
    model = load_checkpoint(model, ckpt_path, torch.device(device))

    # Load dataset
    ds = DDACSNPYStream(root=data_dir, debug=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Create the output directory
    os.makedirs(out_dir, exist_ok=True)

    # Inference loop
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch) # forward pass through encoders → GRIT layers → head.

            # slice per-graph and save
            for g in range(int(batch.num_graphs)):
                sid = int(batch.sample_id[g].item())
                print(f"Now it is processing sample-ID: {sid}")
                node_mask = (batch.batch == g)
                pred_g = pred[node_mask].detach().cpu().numpy()

                out_np = os.path.join(out_dir, f"{sid}_pred_node_displacement.npy")
                np.save(out_np, pred_g)
                with open(os.path.join(out_dir, f"{sid}_meta.json"), "w") as f:
                    json.dump({"sample_id": sid, "num_nodes": int(pred_g.shape[0]), "pred_file": os.path.basename(out_np)}, f)
           

    print(f" Saved predictions to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    # start time
    t0 = time.perf_counter()
    infer(args.data, args.ckpt, args.cfg, args.out, device=args.device, batch_size=args.batch_size)

    # end time
    total = time.perf_counter()-t0
    print(f"(total time:  {total:.2f}s)")

#python predict.py --cfg /home/RUS_CIP/st186731/research_project/hybrid_approach/config_yaml/ddacs-node-regression.yaml --ckpt /home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/results/ddacs-node-regression/41/ckpt/4.ckpt --data /mnt/data/jiang --out /home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/grit_like --batch_size 16
#python predict.py --cfg /home/RUS_CIP/st186731/research_project/hybrid_approach/config_yaml/ddacs-node-regression-graphormerlike.yaml --ckpt /home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/results/ddacs-node-regression-graphormerlike/41/ckpt/9.ckpt --data /mnt/data/jiang --out /home/RUS_CIP/st186731/research_project/hybrid_approach/grit_like_and_graphormer_like/prediction/ddacs-node-regression/graphormer_like --batch_size 16 