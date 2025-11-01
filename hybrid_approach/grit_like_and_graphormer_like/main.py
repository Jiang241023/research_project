import datetime
import os, sys
import torch
import torch.nn as nn
import logging
from yacs.config import CfgNode as CN

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import framework  # noqa, register custom modules
from framework.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg, dump_cfg, set_cfg, load_cfg, makedirs_rm_exist
)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from framework.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from framework.logger import create_logger
from torch_geometric.graphgym.register import act_dict

# diff & pruning
def cfg_diff(cur, base):
    """Return only keys whose values differ from defaults (recursively)."""
    out = CN()
    for k in cur.keys():
        v = cur[k]
        if k not in base:
            out[k] = v
            continue
        dv = base[k]
        if isinstance(v, CN) and isinstance(dv, CN):
            child = cfg_diff(v, dv)
            if len(child) > 0:
                out[k] = child
        else:
            if v != dv:
                out[k] = v
    return out

def prune_irrelevant(effective, full_cfg):
    """Hide knobs that are inert given higher-level choices (sampling/ckpt)."""
    eff = effective.clone()
    # Sampling knobs are irrelevant in full-batch mode
    sampler = getattr(full_cfg.train, 'sampler', 'full_batch')
    if sampler == 'full_batch' and 'train' in eff:
        for k in ['neighbor_sizes', 'walk_length', 'node_per_graph',
                  'radius', 'sample_node', 'iter_per_epoch']:
            if k in eff.train:
                del eff.train[k]
    # Checkpoint knobs irrelevant if disabled
    enable_ckpt = getattr(full_cfg.train, 'enable_ckpt', True)
    if not enable_ckpt and 'train' in eff:
        for k in ['ckpt_best', 'ckpt_clean', 'ckpt_period']:
            if k in eff.train:
                del eff.train[k]
    return eff

def save_yaml(text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

# Optim/Scheduler wrappers
def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum
    )

def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
        num_cycles=cfg.optim.num_cycles,
        min_lr_mode=cfg.optim.min_lr_mode
    )


# Output dir helpers

def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg."""
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    if name_tag:
        run_name += f"-{name_tag}" 
    else:
        run_name += ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)

def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run."""
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)

def run_loop_settings():
    """Create main loop execution settings based on the current cfg."""
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()

    # Build default cfg and snapshot it BEFORE any merges
    set_cfg(cfg)
    defaults_snapshot = cfg.clone()  # <-- defaults snapshot

    # Allow adding new keys and set work_dir
    cfg.set_new_allowed(True)
    cfg.work_dir = os.getcwd()

    # Merge YAML/CLI into cfg
    load_cfg(cfg, args)
    cfg.cfg_file = args.cfg_file

    # Set out_dir (based on cfg file name + optional tag)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)

    # Dump the full (merged) cfg to disk as usual
    dump_cfg(cfg)

    # Print a minimal diff vs defaults + a pruned version
    effective = cfg_diff(cfg, defaults_snapshot)
    pruned = prune_irrelevant(effective, cfg)
    logging.info("=== Effective (non-default) cfg ===\n%s", effective.dump())
    logging.info("=== Effective cfg (pruned) ===\n%s", pruned.dump())

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)

        # Optional device auto-select
        if getattr(cfg, "auto_select_device", False):
            auto_select_device()
        else:
            cfg.device = cfg.accelerator  # e.g., "cuda:0"

        # Pretrained settings
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)

        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        t0 = datetime.datetime.now()
        logging.info(f"    Starting now: {t0}")

        # Data / Logger / Model
        loaders = create_loader()
        loggers = create_logger()
        act_dict['Gelu'] = nn.GELU
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head
            )

        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

        # Print model + minimal cfg (per-run)
        logging.info(model)
        cfg.params = params_count(model) 
        logging.info('Num parameters: %s', cfg.params)

        effective_run = cfg_diff(cfg, defaults_snapshot)
        pruned_run = prune_irrelevant(effective_run, cfg)
        logging.info("=== Run %s: non-default cfg (pruned) ===\n%s",
                     run_id, pruned_run.dump())

        # Save the pruned effective cfg alongside the run
        save_yaml(pruned_run.dump(), os.path.join(cfg.run_dir, "effective_cfg.yaml"))

        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            if cfg.mlflow.use:
                logging.warning("[ML] MLflow logging is not supported with the "
                                "default train.mode, set it to `custom`")
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")

    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')

    t1 = datetime.datetime.now()
    logging.info(f"[*] All done: {t1} (total: {t1 - t0})")

# Example runs:
# python main.py --cfg /home/RUS_CIP/st186731/research_project/hybrid_approach/config_yaml/ddacs-node-regression.yaml  wandb.use False accelerator "cuda:0" optim.max_epoch 10 seed 41 dataset.dir '/mnt/data/jiang'
# python main.py --cfg /home/RUS_CIP/st186731/research_project/hybrid_approach/config_yaml/ddacs-node-regression-graphormerlike.yaml  wandb.use False accelerator "cuda:0" optim.max_epoch 15 seed 41 dataset.dir '/mnt/data/jiang'
