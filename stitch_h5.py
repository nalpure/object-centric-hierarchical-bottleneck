import os
import glob
from utils import StateTransitionsDataset, save_list_dict_h5py, load_list_dict_h5py
from torch.utils import data
import numpy as np


def visualize_list_dict_structure(list_dict):
    for idx, data_dict in enumerate(list_dict):
        print(f"Entry {idx}:")

        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} (numpy array)")
            else:
                print(f"  {key}: {type(value)}")

        print()


def ensure_chw_format(batch):
    if batch.shape[-1] == 3:  # If it's in SHWC (Samples, Height, Width, Channels) format
        batch = batch.permute(0, 3, 1, 2)  # Convert to SCHW format
    return batch


def get_replay_dict(dataloader):
    replay_buffer = []

    for batch in dataloader: 
        obs, action, next_obs = batch
        sample_dict = {
            'obs' : ensure_chw_format(obs),
            'next_obs': ensure_chw_format(next_obs),
            'action': action
            }
        replay_buffer.append(sample_dict)

    return replay_buffer


def stitch_h5_files(input_dir, output_file):
    """
    Stitch together multiple .h5 files containing replay buffer data into one file.
    
    Args:
        input_dir (str): Path to the directory containing the .h5 files.
        output_file (str): Path where the stitched output .h5 file will be saved.
    """

    # Get list of all .h5 files in the directory
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))

    # Iterate over each .h5 file, load data and save it in combined replay buffer
    combined_replay_buffer = []
    for h5_file in h5_files:
        dataset = StateTransitionsDataset(hdf5_file=h5_file)
        dataloader = data.DataLoader(dataset, batch_size=12, shuffle=True, num_workers=0)   # hard-coded batch size of 12
        combined_replay_buffer += get_replay_dict(dataloader)[:-1]  # discarding the last, as it is unlikely to have full batch size of 12

    save_list_dict_h5py(combined_replay_buffer, output_file)


stitch_h5_files(input_dir='data/spriteworld_data/', output_file='data/spriteworld.h5')

#list_dict = load_list_dict_h5py('data/spriteworld.h5')
#visualize_list_dict_structure(list_dict[:10])