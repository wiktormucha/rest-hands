import numpy as np
import cv2 as cv2
import torch
import torch.optim as optim
import random


def define_optimizer(model, optimizer_cfg):
    """
    Define and return the optimizer based on the provided configuration.

    Parameters:
    model (torch.nn.Module): The model whose parameters will be optimized.
    optimizer_cfg (object): Configuration object containing optimizer settings.

    Returns:
    torch.optim.Optimizer: The configured optimizer.
    """
    if optimizer_cfg.optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(
        ), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay, momentum=optimizer_cfg.momentum)
    elif optimizer_cfg.optimizer_type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
    return optimizer


def freeze_seeds(seed_num=42):
    """
    Freeze the random seeds for reproducibility.

    Parameters:
    seed_num (int): The seed number to use for random number generators. Default is 42.
    """
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)


def set_max_cpu_threads(max_threads=16):
    """
    Set the maximum number of CPU threads for PyTorch operations.

    Parameters:
    max_threads (int): The maximum number of CPU threads to use. Default is 16.
    """
    torch.set_num_threads(max_threads)
