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
def adamW_optimizer(params, base_lr,
                   weight_decay) :
    return AdamW(params, lr=base_lr, weight_decay=weight_decay)

@dataclass
class ExtendedSchedulerConfig(SchedulerConfig):
    reduce_factor = 0.5
    schedule_patience = 15
    min_lr= 1e-6
    num_warmup_epochs = 10
    train_mode = 'custom'
    eval_period = 1
    num_cycles = 0.5
    min_lr_mode = "threshold" # ["rescale", "threshold"]


@register.register_scheduler('cosine_with_warmup')
def cosine_with_warmup_scheduler(optimizer,
                                 num_warmup_epochs, max_epoch,
                                 min_lr=0.,
                                 min_lr_mode="rescale", # ["clamp", "rescale"]
                                 ):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch,
        min_lr=min_lr,
        min_lr_mode=min_lr_mode
    )
    return scheduler

def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps,
        num_cycles = 0.5, last_epoch= -1,
        min_lr= 0.,
        min_lr_mode ="rescale"
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        if min_lr > 0.:
            if  min_lr_mode == "clamp":
                lr = max(min_lr/base_lr, lr)
            elif min_lr_mode == "rescale": # "rescale lr"
                lr = (1 - min_lr / base_lr) * lr + min_lr / base_lr

        return lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)



