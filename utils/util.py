import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import random

def generate_mask(
        bz: int, ch_num: int, patch_num: int,
        mask_ratio: float, device: torch.device):
    """
    Generates a binary mask.

    Parameters
    ----------
    bz : int
        Batch size.
    ch_num : int
        Number of channels.
    patch_num : int
        Number of patches.
    mask_ratio : float
        Ratio of masked patches.
    device : torch.device
        Device on which the mask will be created.

    Returns
    -------
    torch.Tensor
        A binary mask of shape (bz, ch_num, patch_num) where each element is 0 or 1.
        The mask is generated such that approximately `mask_ratio`
        fraction of the patches are set to 1.
    """
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    # filling by Bernoulli distribution
    mask = mask.bernoulli_(mask_ratio)
    return mask

def to_tensor(array):
    return torch.from_numpy(array).float()


if __name__ == '__main__':
    a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    print(a)