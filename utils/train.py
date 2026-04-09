import copy
import warnings

from utils.muon import build_muon_optimizer

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from utils.warmup import GradualWarmupScheduler


# customize exp lr scheduler with min lr
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    elif cfg.type == 'muon':
        # Muon for hidden 2-D weight matrices + AdamW for all other params.
        # Config keys (all optional, defaults shown):
        #   muon_lr            (default 0.02)   – Muon spectral learning rate
        #   muon_momentum      (default 0.95)
        #   muon_weight_decay  (default 0.0)
        #   lr                 (default 3e-4)   – AdamW lr for non-Muon params
        #   weight_decay       (default 0.0)    – AdamW weight decay
        #   beta1, beta2       (default 0.9, 0.95)
        #   eps                (default 1e-8)
        #   ns_steps           (default 5)      – Newton-Schulz iterations
        #   muon_exclude_keywords (default ['embed']) – param-name substrings
        #                           whose tensors are always sent to AdamW
        exclude_kws = tuple(getattr(cfg, 'muon_exclude_keywords', ['embed']))
        return build_muon_optimizer(
            model,
            muon_lr=getattr(cfg, 'muon_lr', 0.02),
            muon_momentum=getattr(cfg, 'muon_momentum', 0.95),
            muon_weight_decay=getattr(cfg, 'muon_weight_decay', 0.0),
            adam_lr=getattr(cfg, 'lr', 3e-4),
            adam_betas=(getattr(cfg, 'beta1', 0.9), getattr(cfg, 'beta2', 0.95)),
            adam_eps=getattr(cfg, 'eps', 1e-8),
            adam_weight_decay=getattr(cfg, 'weight_decay', 0.0),
            muon_exclude_keywords=exclude_kws,
            ns_steps=getattr(cfg, 'ns_steps', 5),
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )
    elif cfg.type == 'warmup_plateau':
        return GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.multiplier,
            total_epoch=cfg.total_epoch,
            after_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr
            )
        )
    elif cfg.type == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=cfg.factor,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'expmin_milestone':
        gamma = np.exp(np.log(cfg.factor) / cfg.milestone)
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=gamma,
            min_lr=cfg.min_lr,
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)
