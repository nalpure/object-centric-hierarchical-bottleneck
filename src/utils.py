"""Utility functions."""

import argparse
from datetime import datetime
from datetime import timedelta
import json
import os
from os.path import exists
import random
import h5py
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torch import nn

EPS = 1e-17
CONFIG_DIR = "configs/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_CHANNELS = 3

# Define the mappings between property names and numerical codes
STRING_TO_CODE = {
    "pos_x": 0,
    "pos_y": 1,
    "vel_x": 2,
    "vel_y": 3,
    "hue": 4,
}

CODE_TO_STRING = {v: k for k, v in STRING_TO_CODE.items()}

DEFAULT_CONFIG = {
    "slot_attention": {
        # General parameters
        "train_path": "data/observations/training_data.h5",
        "init_ckpt": None,
        "ckpt_path": "checkpoints/slot_attention/checkpoint.ckpt",
        "slot_save_path": "data/slots/slots.h5",
        "seed": 0,
        "num_workers": 8,
        # Image parameters
        "hdf5_format": "HWC",
        "resolution": [64, 64],
        # Slot Attention parameters
        "num_slots": 4,
        "num_iterations": 3,
        "slots_dim": 64,
        "encdec_dim": 32,
        # Training parameters
        "batch_size": 64,
        "optimizer": "adam",
        "mixed_precision": False,
        "num_epochs": 100,
        "learning_rate": 0.0003,
        "warmup_epochs": 20,
        "decay_epochs": 80,
        "decay_rate": 0.5
    },
    "explicit_latents": {
        # General parameters
        "train_path": "data/slots/slots.h5",
        "init_ckpt": None,
        "ckpt_path": "checkpoints/explicit_latents/checkpoint.ckpt",
        "save_path": "data/explicit_latents/expl_latents.h5",
        "seed": 0,
        "num_workers": 4,
        # Training parameters
        "batch_size": 128,
        "optimizer": "adam",
        "mixed_precision": False,
        "num_epochs": 500,
        "learning_rate": 0.0004,
        "warmup_epochs": 0,
        "decay_epochs": 500,
        "decay_rate": 0.5,
        # Further model parameters
        "latent_dim": 3,
        "normalize": True,
        "noise": 0.0
    },
    "implicit_latents": {
        # General parameters
        "train_path": "data/explicit_latents/expl_latents.h5",
        "init_ckpt": None,
        "ckpt_path": "checkpoints/implicit_latents/checkpoint.ckpt",
        "seed": 0,
        "num_workers": 0,
        # Training parameters
        "batch_size": 128,
        "optimizer": "adam",
        "mixed_precision": False,
        "num_epochs": 500,
        "learning_rate": 0.0004,
        "warmup_epochs": 0,
        "decay_epochs": 500,
        "decay_rate": 0.5,
        # Further model parameters
        "latent_dim": 5,
        "hidden_dim": 128,
        "normalize": True
    }
}

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


def load_config(config_name):
    """
    Load configuration from a nested JSON file.
    """
    user_config_path = CONFIG_DIR + config_name + '.json'
    if not exists(user_config_path):
        raise ValueError("Configuration file not found.")

    with open(user_config_path, 'r') as f:
        user_config = json.load(f)
    
    combined_config = recursive_update(DEFAULT_CONFIG.copy(), user_config)

    return combined_config


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


def encode(string):
    """Encode a string into its corresponding numerical code."""
    if string not in STRING_TO_CODE:
        raise ValueError(f"String '{string}' is not in the predefined set.")
    return STRING_TO_CODE[string]


def decode(code):
    """Decode a numerical code back into its corresponding string."""
    if code not in CODE_TO_STRING:
        raise ValueError(f"Code '{code}' is not valid.")
    return CODE_TO_STRING[code]


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
    estimated_end_time = datetime.now() + timedelta(seconds=remaining_time)

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

        # check if it is a grayscale image
        if images[i].shape[0] == 1:
            images[i] = images[i].squeeze(0)  # shape: [H, W]
            cmap = 'gray'
        elif images[i].ndim == 2:
            cmap = 'gray'
        else:
            images[i] = images[i].transpose(1, 2, 0)  # shape: [H, W, 3]
            cmap = None
    
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


class ImageDataset(data.Dataset):
    def __init__(self, hdf5_file, hdf5_format='HWC', output_format='CHW', only_first=False, only_original=False):
        """
        Streaming dataset over a single .h5 file.
        Assumes structure:
              /<episode_idx>/
                obs            shape: (P, T, H, W, C) or (P, T, C, H, W)
                perturbed      same shape
        Returns individual images of shape (C,H,W) or (H,W,C). 
        If only_first is True, returns only the first image of each episode.
        If only_original is True, returns only the original images.
        """
        self.h5_path = hdf5_file
        self.hdf5_format = hdf5_format
        self.output_format = output_format
        self.only_first = only_first
        self.only_original = only_original

        with h5py.File(self.h5_path, "r") as f:
            self.episodes = sorted(int(k) for k in f.keys())
            # assume every episode has same T
            first = f[str(self.episodes[0])]["obs"]
            self.num_seq = first.shape[0]
            self.seq_len = first.shape[1]
            
        self.idx2ep = []
        for ep in self.episodes:
            for seq in range(self.num_seq):
                if self.only_first:
                    self.idx2ep.append((ep, seq, 0, False))
                    if not self.only_original:
                        self.idx2ep.append((ep, seq, 0, True))
                else:
                    for t in range(self.seq_len):
                        self.idx2ep.append((ep, seq, t, False))
                        if not self.only_original:
                            self.idx2ep.append((ep, seq, t, True))
    
    def __len__(self):
        return len(self.idx2ep)
    
    def __getitem__(self, idx):
        # open per worker (lazy)
        if not hasattr(self, "_h5"):
            self._h5 = h5py.File(self.h5_path, "r")

        ep, seq, t, is_perturbation = self.idx2ep[idx]
        grp = self._h5[str(ep)]

        if is_perturbation:
            img = grp["perturbed"][seq][t]
        else:
            img = grp["obs"][seq][t]

        # optional transpose
        if self.hdf5_format == "HWC" and self.output_format == "CHW":
            img = np.transpose(img, (2, 0, 1))
        elif self.hdf5_format == "CHW" and self.output_format == "HWC":
            img = np.transpose(img, (1, 2, 0))

        # convert to torch
        img = torch.from_numpy(img).float()

        return img
    

class PerturbedImageSequenceDataset(data.Dataset):
    def __init__(self, h5_path, hdf5_format="HWC", output_format="CHW"):
        """
        Streaming dataset over a single .h5 file.
        Assumes structure:
           /<episode_idx>/
               obs            shape: (P, T, H, W, C) or (P, T, C, H, W)
               perturbed      same shape
               magnitudes     shape: (P,)
               indices        shape: (P,)
               properties     shape: (P,)
        Returns whole time series for each observation/perturbation-pair.
        """
        self.h5_path = h5_path
        self.hdf5_format = hdf5_format
        self.output_format = output_format

        # Read once to build idx mapping
        with h5py.File(self.h5_path, "r") as f:
            self.episodes = sorted(int(k) for k in f.keys())
            # assume every episode has same T
            first = f[str(self.episodes[0])]["obs"]
            num_pairs = first.shape[0]

        # Build (episode, step) lookup
        self.idx2ep = []
        for ep in self.episodes:
            for p in range(num_pairs):
                self.idx2ep.append((ep, p))

    def __len__(self):
        return len(self.idx2ep)

    def __getitem__(self, idx):
        # open per worker (lazy)
        if not hasattr(self, "_h5"):
            self._h5 = h5py.File(self.h5_path, "r")

        ep, p = self.idx2ep[idx]
        grp = self._h5[str(ep)]

        # Load on‐demand
        orig = grp["obs"][p]         # (T,H,W,C) or (T,C,H,W)
        pert = grp["perturbed"][p]
        mags = grp["magnitudes"][p]
        inds = grp["indices"][p]
        props = grp["properties"][p]

        # optional transpose
        if self.hdf5_format == "HWC" and self.output_format == "CHW":
            # orig: (T,H,W,C) -> (T,C,H,W)
            orig = np.transpose(orig, (0,3,1,2))
            pert = np.transpose(pert, (0,3,1,2))
        elif self.hdf5_format == "CHW" and self.output_format == "HWC":
            orig = np.transpose(orig, (0,2,3,1))
            pert = np.transpose(pert, (0,2,3,1))

        # convert to torch
        orig = torch.from_numpy(orig).float()
        pert = torch.from_numpy(pert).float()
        mags = torch.tensor(mags, dtype=torch.float32)
        inds = torch.tensor(inds, dtype=torch.int8)
        props = torch.tensor(props, dtype=torch.int8)

        return orig, pert, mags, inds, props

    def __del__(self):
        # ensure file closes on worker shutdown
        if hasattr(self, "_h5"):
            try:
                self._h5.close()
            except:
                pass


class ImageSequencePairDataset(data.Dataset):
    """
    Streaming dataset over a single .h5 file that returns *pairs* of consecutive
    observations from each sequence.

    For a sequence of length T=4, this yields pairs (1,2), (2,3), (3,4).
    Each pair corresponds to the original (non-perturbed) observations.

    Structure assumed in HDF5:
       /<episode_idx>/
           obs            shape: (P, T, H, W, C) or (P, T, C, H, W)
           perturbed      same shape (ignored here)
           magnitudes     shape: (P,)
           indices        shape: (P,)
           properties     shape: (P,)
    """

    def __init__(self, h5_path, hdf5_format="HWC", output_format="CHW"):
        self.h5_path = h5_path
        self.hdf5_format = hdf5_format
        self.output_format = output_format

        # Read once to determine sequence length (T) and counts
        with h5py.File(self.h5_path, "r") as f:
            self.episodes = sorted(int(k) for k in f.keys())
            first_obs = f[str(self.episodes[0])]["obs"]
            num_pairs = first_obs.shape[0]
            self.seq_len = first_obs.shape[1]  # T

        # Build index map: (episode, pair_index, t)
        # For each sequence of T timesteps, there are (T-1) pairs
        self.idx2pair = []
        for ep in self.episodes:
            for p in range(num_pairs):
                for t in range(self.seq_len - 1):
                    self.idx2pair.append((ep, p, t))

    def __len__(self):
        return len(self.idx2pair)

    def __getitem__(self, idx):
        if not hasattr(self, "_h5"):
            self._h5 = h5py.File(self.h5_path, "r")

        ep, p, t = self.idx2pair[idx]
        grp = self._h5[str(ep)]

        # Load only the two consecutive frames
        orig = grp["obs"][p, t:t+2]  # shape: (2, H, W, C) or (2, C, H, W)

        # optional transpose
        if self.hdf5_format == "HWC" and self.output_format == "CHW":
            orig = np.transpose(orig, (0, 3, 1, 2))
        elif self.hdf5_format == "CHW" and self.output_format == "HWC":
            orig = np.transpose(orig, (0, 2, 3, 1))

        orig = torch.from_numpy(orig).float()  # shape [2, C, H, W]

        return orig  # (two-frame pair)



class SlotDataset(data.Dataset):
    """
    Loads slots from orig_seq and pert_seq (shape [B, T, O, S]) in a HDF5 file.
    Flattens all slots across time and objects, returning one randomly chosen slot
    (either original or perturbed) per __getitem__ call.
    """
    def __init__(self, hdf5_file, feature_mean=None, feature_std=None, normalize=True):
        self.hdf5_file = hdf5_file
        self.normalize = normalize
        self.slots = self._load_slots_data()

        if normalize:
            if feature_mean is None or feature_std is None:
                self.feature_mean = self.slots.mean(dim=0)  # Shape: [S]
                self.feature_std = self.slots.std(dim=0)    # Shape: [S]
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
            self.slots = (self.slots - self.feature_mean) / self.feature_std


    def _load_slots_data(self):
        data = {
            'orig_seq': [],
            'pert_seq': []
        }

        with h5py.File(self.hdf5_file, 'r') as f:
            for key in f.keys():
                for k in data:
                    if k in key:
                        data[k].append(f[key][...])
                        break

        # Stack all data into single tensors
        orig_seq = torch.tensor(np.concatenate(data['orig_seq']), dtype=torch.float32)
        pert_seq = torch.tensor(np.concatenate(data['pert_seq']), dtype=torch.float32)
        sequences = torch.cat((orig_seq, pert_seq), dim=0)
        # Flatten the sequences across time and objects
        slots = sequences.reshape(-1, orig_seq.shape[-1])  # Shape: [B*T*O, S]
        return slots
    
    def _compute_mean_std(self):
        all_data = torch.tensor(np.concatenate([self.original, self.perturbed], axis=0), dtype=torch.float32)
        flat = all_data.reshape(-1, all_data.shape[-1])  # Shape: [B*T*O, E]
        mean = flat.mean(dim=0)  # Shape: [E]
        std = flat.std(dim=0)    # Shape: [E]
        return mean, std
        

    def __len__(self):
        return self.slots.shape[0]

    def __getitem__(self, idx):
        # Select a random slot regardless of whether it was original or perturbed
        return self.slots[idx]


class PerturbedSlotSequenceDataset(data.Dataset):
    """
    Dataset for loading perturbed slot sequences from a HDF5 file.
    Assumes the HDF5 file contains the following datasets:
        - 'orig_seq': Original slot sequences
        - 'pert_seq': Perturbed slot sequences
        - 'magnitude': Magnitude of perturbations
        - 'obj_index': Object indices
        - 'prop_index': Property indices

    The dataset can be normalized using the provided feature mean and standard deviation.
    If normalization is enabled but feature mean and standard deviation are not provided,
    the mean and standard deviation will be computed from the data.
    """
    def __init__(self, hdf5_file, feature_mean=None, feature_std=None, normalize=True):
        self.hdf5_file = hdf5_file
        self.normalize = normalize

        # Load all data into memory
        self.original, self.perturbed, self.magnitude, self.obj_index, self.prop_index = self._load_slots_data()
        self.num_samples = len(self.original)

        if self.normalize:
            if feature_mean is None or feature_std is None:
                self.feature_mean, self.feature_std = self._compute_mean_std()
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
            self.original = self._normalize_tensor(self.original)
            self.perturbed = self._normalize_tensor(self.perturbed)


    def _load_slots_data(self):
        data = {
            'orig_seq': [],
            'pert_seq': [],
            'magnitude': [],
            'obj_index': [],
            'prop_index': []
        }

        with h5py.File(self.hdf5_file, 'r') as f:
            for key in f.keys():
                for k in data:
                    if k in key:
                        data[k].append(f[key][...])
                        break

        # Stack all data into single tensors
        orig_seq = torch.tensor(np.concatenate(data['orig_seq']), dtype=torch.float32)
        pert_seq = torch.tensor(np.concatenate(data['pert_seq']), dtype=torch.float32)
        magnitude = torch.tensor(np.concatenate(data['magnitude']), dtype=torch.float32)
        obj_index = torch.tensor(np.concatenate(data['obj_index']), dtype=torch.int64)
        prop_index = torch.tensor(np.concatenate(data['prop_index']), dtype=torch.int64)

        return orig_seq, pert_seq, magnitude, obj_index, prop_index

    def _compute_mean_std(self):
        with torch.no_grad():
            all_data = torch.tensor(np.concatenate([self.original, self.perturbed], axis=0), dtype=torch.float32)
            flat = all_data.reshape(-1, all_data.shape[-1])  # Shape: [B*T*O, E]
            mean = flat.mean(dim=0)  # Shape: [E]
            std = flat.std(dim=0)    # Shape: [E]
            return mean, std

    def _normalize_tensor(self, x):
        return (x - self.feature_mean) / self.feature_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.original[idx],
                self.perturbed[idx],
                self.magnitude[idx],
                self.obj_index[idx],
                self.prop_index[idx])