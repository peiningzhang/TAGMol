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
    kappa_t in [0, 1] with kappa_0=0, kappa_1=1.
    Uses cosine schedule: kappa_t = (1 - cos(pi*t)) / 2.
    """

    def kappa(self, t):
        """Interpolation coefficient kappa_t. t in [0, 1]."""
        if isinstance(t, torch.Tensor):
            return (1 - torch.cos(np.pi * t)) / 2
        return (1 - np.cos(np.pi * t)) / 2

    def d_kappa_dt(self, t):
        """Derivative d(kappa)/dt."""
        if isinstance(t, torch.Tensor):
            return (np.pi / 2) * torch.sin(np.pi * t.clamp(0, 1))
        return (np.pi / 2) * np.sin(np.pi * np.clip(t, 0, 1))
