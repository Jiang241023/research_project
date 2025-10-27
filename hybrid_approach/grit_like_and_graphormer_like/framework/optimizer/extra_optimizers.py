import logging
import math
from typing import Iterator
from dataclasses import dataclass

import torch.optim as optim
from torch.nn import Parameter
from torch.optim import Adagrad, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.graphgym.optim import SchedulerConfig
import torch_geometric.graphgym.register as register

@register.register_optimizer('adamW')
def adamW_optimizer(params: Iterator[Parameter], base_lr: float,
                   weight_decay: float) -> AdamW:
    return AdamW(params, lr=base_lr, weight_decay=weight_decay)

@dataclass
class ExtendedSchedulerConfig(SchedulerConfig):
    reduce_factor: float = 0.5
    schedule_patience: int = 15
    min_lr: float = 1e-6
    num_warmup_epochs: int = 10
    train_mode: str = 'custom'
    eval_period: int = 1
    num_cycles: float = 0.5
    min_lr_mode: str = "threshold" # ["rescale", "threshold"]

@register.register_scheduler('linear_with_warmup')
def linear_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler

def get_linear_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        last_epoch: int = -1,
        min_lr:float=0.,
        min_lr_mode="rescale"
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    base_lr = optimizer.param_groups[0]["lr"]
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        lr = max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
        if min_lr > 0.:
            if  min_lr_mode == "clamp":
                lr = max(min_lr/base_lr, lr)
            elif min_lr_mode == "rescale": # "rescale lr"
                lr = (1 - min_lr / base_lr) * lr + min_lr / base_lr
        return  lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

