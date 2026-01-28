import random
import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = False) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return g


def normalize_slots(slots, mean, std):
    return (slots - mean) / std


def denormalize_slots(slots, mean, std):
    return slots * std + mean


def convert_image_format(arr, in_fmt: str, out_fmt: str):
    """
    Handles conversion between 'HWC' and 'CHW' formats for both
    single frames and sequences.
    """
    if in_fmt == out_fmt:
        return arr

    if in_fmt == "HWC" and out_fmt == "CHW":
        if arr.ndim == 3:
            return np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 4:
            return np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim == 5:
            return np.transpose(arr, (0, 1, 4, 2, 3))
    elif in_fmt == "CHW" and out_fmt == "HWC":
        if arr.ndim == 3:
            return np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 4:
            return np.transpose(arr, (0, 2, 3, 1))
        elif arr.ndim == 5:
            return np.transpose(arr, (0, 1, 2, 3, 4))

    raise ValueError(f"Unsupported transpose for shape {arr.shape} ({in_fmt}→{out_fmt})")