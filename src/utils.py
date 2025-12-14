"""Utility functions."""

import argparse
from datetime import datetime
from datetime import timedelta
import os
from os.path import exists
import random
from typing import List
import h5py
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import tomli
import tomli_w
from torch.utils import data

import torch
from torch import nn

from datasets import ImageDataset, PerturbedImageSequenceDataset, PerturbedSlotSequenceDataset
from explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from implicit_latents.relational_latent_dynamics import RelationalLatentDynamics
from slot_attention.autoencoder import SlotAttentionAutoEncoder

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the mappings between property names and numerical codes
# Default: attributes are not ordered by explicit / implicit distinction
# Sorted: first explicit attributes (pos, hue), then implicit attributes (vel)
implicit_properties = ['vel_x', 'vel_y']
explicit_properties = ['pos_x', 'pos_y', 'hue']

STRING_TO_CODE_DEFAULT = {
    "pos_x": 0,
    "pos_y": 1,
    "vel_x": 2,
    "vel_y": 3,
    "hue": 4,
}

STRING_TO_CODE_SORTED = {
    "pos_x": 0,
    "pos_y": 1,
    "hue": 2,
    "vel_x": 3,
    "vel_y": 4,
}

CODE_TO_STRING_DEFAULT = {v: k for k, v in STRING_TO_CODE_DEFAULT.items()}
CODE_TO_STRING_SORTED = {v: k for k, v in STRING_TO_CODE_SORTED.items()}



def get_config_argument():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str, help='name of the configuration file to use')
    args = parser.parse_args()
    args = vars(args)

    if args['config'] is None:
        raise ValueError("Please provide a configuration file.")
    if len(args.keys()) != 1:
        raise ValueError("Invalid arguments provided.")

    return args['config']


def recursive_update(base_dict, override_dict):
    """
    Recursively updates base_dict with values from override_dict.
    If a key doesn't exist in base_dict, it is added.
    """
    for key, value in override_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def reorder_perturbation_indices(indices, shift=0):
    """Reorder perturbation indices to match the new STRING_TO_CODE mapping."""
    reordered_indices = []
    for idx in indices:
        prop_name = CODE_TO_STRING_DEFAULT[idx.item()]
        new_idx = STRING_TO_CODE_SORTED[prop_name]
        reordered_indices.append(new_idx + shift)
    return torch.tensor(reordered_indices, device=indices.device)


def get_implicit_codes() -> List[int]:
    return [STRING_TO_CODE_DEFAULT[prop] for prop in implicit_properties]


def get_explicit_codes() -> List[int]:
    return [STRING_TO_CODE_DEFAULT[prop] for prop in explicit_properties]


def get_lr_schedule(optimizer, warmup_steps, decay_steps, decay_rate):
    """ Creates a learning rate scheduler with warmup and exponential decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Exponential decay after warmup
            decay_factor = (current_step - warmup_steps) / decay_steps
            return decay_rate ** decay_factor
    
    return LambdaLR(optimizer, lr_lambda)


def huber_loss_(targets, predictions, batch):
    """
    Returns the list of indexes of the predicitons corersponding to the target
    Args:
    - targets: shape (num_masks, height, width, channels)
    - predictions: shape as targets
    Returns the matching index for each predicted mask 
    """
    loss_fn = torch.nn.HuberLoss()
    order = []
    offset = batch * predictions.shape[0]
    for pred in predictions:
        pred_loss = []
        for target in targets:
            target_loss = loss_fn(pred, target)
            pred_loss.append(target_loss.item())
        order.append(np.argmin(pred_loss) + offset)

    return order
    

def huber_loss(targets, predictions):
    return [huber_loss_(targets[i], predictions[i], i) for i in range(targets.shape[0])]


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


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
            

def load_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, "r") as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                if hf[grp][key].shape == ():
                    array_dict[i][key] = hf[grp][key][()]
                else:
                    array_dict[i][key] = hf[grp][key][:]

    return array_dict


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                if hf[grp][key].shape == ():
                    array_dict[i][key] = hf[grp][key][()]
                else:
                    array_dict[i][key] = hf[grp][key][:]
                
    return array_dict


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y):
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(indices.size()[0], max_index, dtype=torch.float32, device=indices.device)
    
    if max_index == 0:
        return zeros
 
    one_hot = zeros.scatter_(1, indices.unsqueeze(1), 1)
    return one_hot


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


def load_model(model, optimizer, path):
    """
    loads a model from a given path
    :param path: (str) file path (.pt file)
    """
    checkpoint = torch.load(path)
    iteration = checkpoint['iteration']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    model.train()
    return iteration


def convert_image_format(obs, hdf5_format, output_format):
    if hdf5_format == 'CHW' and output_format == 'HWC':
        obs = np.transpose(obs, (1, 2, 0))  # Convert CHW to HWC
    elif hdf5_format == 'HWC' and output_format == 'CHW':
        obs = np.transpose(obs, (2, 0, 1))  # Convert HWC to CHW
    elif hdf5_format != output_format:
        raise ValueError(f'Expected \'CHW\' or \'HWC\' as hdf5-format and output-format, received \'{hdf5_format}\ and \'{output_format}\'.')
    return obs


def log_progress(current_step, total_steps, start_time, additional_msg=''):
    elapsed_time = (datetime.now() - start_time).total_seconds()
    progress_fraction = current_step / total_steps
    estimated_total_time = elapsed_time / progress_fraction if progress_fraction > 0 else 0
    remaining_time = estimated_total_time - elapsed_time
    remaining_timedelta = timedelta(seconds=int(remaining_time))
    remaining_time_str = str(remaining_timedelta)

    msg = (
        f"{current_step}/{total_steps} - "
        f"Remaining: {remaining_time_str}"
    )

    if additional_msg: msg += f" - {additional_msg}"
    print(msg)


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


def plot_images(images, save_path, labels=None, title=None):
    """
    Displays all images in a single row and saves the resulting plot.

    Args:
        images (iterable): An iterable of images. Each image should be of shape [3, H, W].
        save_path (str): File path to save the plotted image.
        labels (iterable): An iterable of labels for each image.
    """
    num_images = len(images)
    images = list(images)
    for i in range(num_images):
        if hasattr(images[i], 'detach'):
            images[i] = images[i].detach().cpu().numpy()
        
        images[i] = np.clip(images[i], 0, 1)
        cmap = None

        # check if it is a grayscale image
        if images[i].shape[0] == 1:
            images[i] = images[i].squeeze(0)  # shape: [H, W]
            cmap = 'gray'
        elif images[i].ndim == 2:
            cmap = 'gray'
        elif images[i].shape[0] == 3:
            images[i] = images[i].transpose(1, 2, 0)  # shape: [H, W, 3]
    
    # Create a figure with one row and as many columns as there are images.
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    
    # Ensure that axes is always iterable (if only one image, axes is not a list).
    if num_images == 1:
        axes = [axes]
    
    # Loop over images and display each one.
    for idx, img in enumerate(images):
        axes[idx].imshow(img, cmap=cmap)
        axes[idx].axis('off')
    
    if labels is not None:
        for ax, label in zip(axes, labels):
            ax.set_title(label)

    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to add padding at the top
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_grid(rows, row_titles, column_titles, save_path="output.png"):
    num_columns = len(column_titles)
    
    if not num_columns == len(rows[0]):
        raise ValueError(f"Number of column titles ({num_columns}) must match number of columns in rows ({len(rows[0])}).")
    if not len(row_titles) == len(rows):
        raise ValueError(f"Number of row titles ({len(row_titles)}) must match number of rows ({len(rows)}).")
    
    fig, axes = plt.subplots(len(rows), num_columns, figsize=(4 * num_columns, 4 * len(rows)))

    if num_columns == 1:
        axes = axes.reshape(len(rows), 1)

    for row_idx, images in enumerate(rows):
        for col_idx in range(num_columns):
            img = images[col_idx]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()

            if img.shape[0] == 1:
                img = img.squeeze(0)  # shape: [H, W]
                cmap = "gray"
            elif img.ndim == 2:
                cmap = "gray"
            else:
                img = img.transpose(1, 2, 0)  # shape: [H, W, 3]
                cmap = None

            axes[row_idx, col_idx].imshow(img.clip(0, 1), cmap=cmap)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(column_titles[col_idx], fontsize=12)

    # Manually add row titles on the left
    for row_idx, title in enumerate(row_titles):
        fig.text(0.1, 0.8 - row_idx * 0.76 / len(rows), title, va='center', ha='right', fontsize=14, weight='bold')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def create_trail(img_seq, highlight="last"):
    """
    Creates a trailing effect by blending the highlighted image with
    a series of trailing images.

    Args:
        img_seq: numpy array of shape [T, H, W, C]
        highlight: 'last' or 'first' - which frame to highlight
    Returns:
        final_img: numpy array of shape [H, W, C]
    """
    if highlight == "last":
        highlighted_img = img_seq[-1]
    elif highlight == "first":
        highlighted_img = img_seq[0]
    else:
        raise ValueError(f"Unknown highlight mode: {highlight}")

    trail_length = img_seq.shape[1] - 1
    final_img = np.copy(highlighted_img) / 2.0

    for trail_img in img_seq[:-1]:
        final_img += (
            np.minimum(final_img, trail_img) / trail_length / 2.0
        )

    return final_img

from PIL import Image

def save_gif_from_array(frames, output_path="rollout.gif", fps=30, scale=4.0, loop=0):
    """
    Create a GIF directly from a list or NumPy array of RGB frames.
    
    Args:
        frames: list or np.ndarray of shape [T, H, W, 3] (values in [0,1] or [0,255])
        output_path: where to save the gif
        fps: frames per second
        scale: how much to enlarge frames (e.g. 2.0 = 2× bigger)
        loop: 0 = infinite, N = loop N times
    """
    # Convert to uint8 if needed
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    
    if frames.ndim != 4:
        raise ValueError(f"Expected frames to have 4 dimensions [T, H, W, C], got {frames.shape}")
    
    if frames.shape[-1] > 10 and frames.shape[1] == 3:
        frames = frames.transpose(0, 2, 3, 1)  # Convert from [T, C, H, W] to [T, H, W, C]

    if scale != 1.0:
        new_frames = []
        for f in frames:
            img = Image.fromarray(f)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.NEAREST)
            new_frames.append(np.array(img))
        frames = np.stack(new_frames)

    imageio.mimsave(output_path, frames, duration=1/fps, loop=loop)
    print(f"GIF saved to {output_path}")


from pathlib import Path

def load_norm_stats(path):    
    if exists(path):
        print("Loading normalization stats from", path)
        stats = torch.load(path, weights_only=True)
        mean = stats["mean"]
        std = stats["std"]
        print(f"mean: {mean}, std: {std}")
    else:
        raise ValueError(f"Normalization stats file not found at {path}.")

    return mean, std

def save_norm_stats(mean, std, path):
    if exists(path):
        raise ValueError(f"Normalization stats file already exists at {path}. Please provide a new path to save the stats.")
    else:
        print(f"mean: {mean}, std: {std}")
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


def get_dataloader(config: dict, save_mode=False) -> data.DataLoader:
    print("Loading training data...")

    if config["type"] == "slot_attention":
        # In save mode, load sequences
        if save_mode:
            dataset = PerturbedImageSequenceDataset(
                npz_path=config["data"]["path"],
                in_format=config["data"]["obs_format"],
                T=config["data"]["seq_length"],
                only_original=False
            )
        # Otherwise, load individual images
        else:
            dataset = ImageDataset(
                npz_path=config["data"]["path"],
                in_format=config["data"]["obs_format"],
                seq_length=config["data"]["seq_length"],
                only_original=not config["data"]["include_perturbed"]
            )

    elif config["type"] == "explicit_latents":
        prop_skip_codes = [] if save_mode else get_implicit_codes()
        normalize = config["data"]["normalize"] if "normalize" in config["data"] else False
        mean = None
        std = None
        norm_stats_path = config["data"]["path"].replace(".h5", "_norm_stats.pt")

        if normalize and os.path.exists(norm_stats_path):
            print("Loading normalization stats from", norm_stats_path)
            stats = torch.load(norm_stats_path, weights_only=True)
            mean = stats["mean"]
            std = stats["std"]

        dataset = PerturbedSlotSequenceDataset(
            hdf5_file=config["data"]["path"],
            feature_mean=mean,
            feature_std=std,
            normalize=normalize,
            timesteps=config["data"]["seq_length"],
            prop_skip_codes=prop_skip_codes
        )            

        if normalize and not os.path.exists(norm_stats_path):
            print("Saving normalization stats to", norm_stats_path)
            torch.save({"mean": dataset.feature_mean, "std": dataset.feature_std}, norm_stats_path)

    elif config["type"] == "implicit_dynamics":
        t_past = config["model"]["t_past"]
        t_future = config["train"]["t_future"]
        
        dataset = PerturbedSlotSequenceDataset(
            hdf5_file=config["data"]["path"], 
            normalize=False, 
            timesteps=t_past + t_future,
            prop_skip_codes=get_explicit_codes()
        ) 
    
    train_dataloader = data.DataLoader(
        dataset=dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=config["num_workers"]
    )
    print(f"Finished loading all {config['train']['batch_size'] * len(train_dataloader)} training samples.")
    return train_dataloader


def initialize_model(dataloader: data.DataLoader, config: dict, eval_mode: bool) -> torch.nn.Module:
    if config["type"] == "slot_attention":
        model = SlotAttentionAutoEncoder(
            resolution=(config["model"]["obs_height"], config["model"]["obs_width"]),
            num_channels=config["model"]["obs_channels"],
            num_slots=config["slot"]["num_slots"],
            num_iterations=config["slot"]["sa_iterations"],
            slots_dim=config["model"]["slot_size"],
            encdec_dim=config["model"]["mlp_size"]
        )
    elif config["type"] == "explicit_latents":
        slots_dim = next(iter(dataloader))[0].shape[-1]
        model = ExplicitLatentAutoEncoder(
            config["model"]["explicit_dim"],
            slots_dim
        )
    elif config["type"] == "implicit_dynamics":
        explicit_dim = next(iter(dataloader))[0].shape[-1]
        model = RelationalLatentDynamics(
            explicit_dim=explicit_dim,
            implicit_dim=config["model"]["latent_dim"] - explicit_dim,
            seq_len=config["model"]["t_past"],
            edge_dim=config["model"]["edge_dim"],
            latent_edge_dim=config["model"]["latent_edge_dim"]
        )

    model = model.to(DEVICE)
    if eval_mode:
        model.eval()
    else:
        model.train()

    ckpt = config['base_ckpt']
    if ckpt != "":
        print(f"Loading model weights from {ckpt}")
        checkpoint = torch.load(ckpt, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Finished loading model. Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model