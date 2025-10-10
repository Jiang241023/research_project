import logging
import time
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch

def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        #print(f"the shape of pred (from train_epoch): {pred.shape}")
        #print(f"the shape of true (from train_epoch): {true.shape}")
        loss, pred_score = compute_loss(pred, true)
        #print(f"loss (from train_epoch): {loss}")
        #print(f"the shape of pred_score (from train_epoch): {pred_score.shape}")
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()

def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for step, batch in enumerate(loader):
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)

        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()

@register_train('example')
def train_example(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
