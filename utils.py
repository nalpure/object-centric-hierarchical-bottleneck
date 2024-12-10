"""Utility functions."""

from datetime import datetime
from datetime import timedelta
import os
import random
import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17

# Define the mappings between property names and numerical codes
STRING_TO_CODE = {
    "pos_x": 0,
    "pos_y": 1,
    "vel_x": 2,
    "vel_y": 3,
    "hue": 4,
}

CODE_TO_STRING = {v: k for k, v in STRING_TO_CODE.items()}


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


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


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
        f"Elapsed: {timedelta(seconds=int(elapsed_time))} - "
        f"Remaining: {remaining_time_str} - "
        f"Estimated End Time: {estimated_end_time.strftime('%H:%M:%S')}"
    )

    if additional_msg: msg += f" - {additional_msg}"
    print(msg)


def set_seed(seed):
    random.seed()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, hdf5_format='HWC', output_format='CHW'):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.hdf5_format = hdf5_format
        self.output_format = output_format

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]
        obs = to_float(self.experience_buffer[ep]['obs'][step])
        obs = convert_image_format(obs, self.hdf5_format, self.output_format)
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        return obs, action, next_obs



class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, hdf5_format='CHW', output_format='CHW', path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.experience_buffer = [el for el in self.experience_buffer if el['next_obs'].size > 0]
        self.hdf5_format = hdf5_format
        self.output_format = output_format
        self.path_length = path_length

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        l = len(self.experience_buffer[idx]['obs'])
        path_length = self.path_length if l >= self.path_length else l
        for i in range(path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            obs = convert_image_format(obs, self.hdf5_format, self.output_format)
            action = self.experience_buffer[idx]['action'][i]
            observations.append(obs)
            actions.append(action)
        obs = to_float(self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        obs = convert_image_format(obs, self.hdf5_format, self.output_format)
        observations.append(obs)
        
        return observations, actions
