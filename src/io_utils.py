import os
import h5py
import torch
from pathlib import Path
import tomli
import tomli_w
from os.path import exists


def save_dict_h5py(array_dict, fname, mode="w"):
    """Save list of dictionaries containing numpy arrays and strings to h5py file.
    If mode="a", append new groups with consecutive indices.
    """
    # Ensure directory exists
    directory = os.path.dirname(fname)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, mode) as hf:
        for key, value in array_dict.items():
            hf.create_dataset(key, data=value, compression="gzip", compression_opts=9)


def load_norm_stats(path):    
    if exists(path):
        print("Loading normalization stats from", path)
        stats = torch.load(path, weights_only=True)
        mean = stats["mean"]
        std = stats["std"]
    else:
        raise ValueError(f"Normalization stats file not found at {path}.")

    return mean, std


def save_norm_stats(mean, std, path):
    if exists(path):
        raise ValueError(f"Normalization stats file already exists at {path}. Please provide a new path to save the stats.")
    else:
        print("Saving normalization stats to", path)
        torch.save({"mean": mean, "std": std}, path)


def load_config_by_name(name):
    try:
        path = Path(__file__).parent.parent / "configs" / (name + ".toml")

        with path.open("rb") as f:
            config = tomli.load(f)

        return config
    except FileNotFoundError:
        print(f"Config file '{name}' does not exist!")
        raise
    except Exception as e:
        print(f"Error occured while loading config: {e}")
        raise


def load_config(path):
    with open(path, "rb") as f:
        config = tomli.load(f)

    return config


def save_config(config, path):
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def make_unique_dir(parent_dir, dirname):
    """ Get a unique save path by appending an index if the file already exists. """
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    if os.path.exists(os.path.join(parent_dir, dirname)):
        run_index = 0
        while os.path.exists(os.path.join(parent_dir, f"{dirname}_{run_index}")):
            run_index += 1
        run_name = f"{dirname}_{run_index}"
    else:
        run_name = dirname
    
    dir_path = os.path.join(parent_dir, run_name)
    os.mkdir(dir_path)
    return dir_path