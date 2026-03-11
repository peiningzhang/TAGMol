import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    logging.shutdown()

def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %% DFM (Exact Discrete Flow Matching) non-linear time scheduler
class DFMTimeScheduler:
    """
    Non-linear time scheduler for Exact Discrete Flow Matching.
    kappa = sigma / (sigma + sigma_data), maps sigma in [0, +inf) to kappa in [0, 1).
    kappa_0=0, kappa -> 1 as sigma -> +inf.
    """

    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    # def kappa(self, sigma):
    #     """Interpolation coefficient kappa = sigma / (sigma + sigma_data)."""
    #     denom = sigma + self.sigma_data
    #     if isinstance(sigma, torch.Tensor):
    #         return sigma / denom
    #     return sigma / denom

    # def d_kappa_dt(self, sigma):
    #     """Derivative d(kappa)/d(sigma) = sigma_data / (sigma + sigma_data)^2."""
    #     denom = (sigma + self.sigma_data) ** 2
    #     if isinstance(sigma, torch.Tensor):
    #         return (torch.full_like(sigma, self.sigma_data, dtype=sigma.dtype, device=sigma.device) / denom)
    #     return self.sigma_data / denom
    def mask_rate(self, sigma):
        """Mask rate = sigma / (sigma + sigma_data)."""
        return sigma / (sigma + self.sigma_data)
    def kappa(self, sigma):
        """Interpolation coefficient mask_rate = sigma / (sigma + sigma_data)."""

        mask_rate = sigma / (sigma + self.sigma_data)
        if isinstance(sigma, torch.Tensor):
            return (1 - mask_rate)**2
        return (1 - mask_rate)**2

    def d_kappa_dt(self, sigma):
        """Derivative d(kappa)/d(sigma) = 2 * (1 - mask_rate) * mask_rate."""
        mask_rate = sigma / (sigma + self.sigma_data)
        if isinstance(sigma, torch.Tensor):
            return 2 * (1 - mask_rate)
        return 2 * (1 - mask_rate)
