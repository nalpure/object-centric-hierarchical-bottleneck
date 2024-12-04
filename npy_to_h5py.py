import numpy as np
import argparse
import os
from utils import save_list_dict_h5py


parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, default='data/train.h5', help='Input file name / path.')
parser.add_argument('--stacked-frames', default=1, type=int, help='number of frames stacked in each sample')

args = parser.parse_args()
args = vars(args)

output_path = os.path.splitext(args["fname"])[0] + ".h5"


sequences = np.load(args["fname"])
replay_buffer = []

# create arrays with shifted training data
shifted_data = [sequences[:, i : i + sequences.shape[1] - args['stacked_frames']] for i in range(args['stacked_frames'])]

# create frame stack with consecutive frames:
# concatenate the differently shifted data arrays in the channel dimension
# the length of the sequence reduces by (stacked_frames - 1)
train_x = np.concatenate(shifted_data, axis=-1)

# normalize and rearrange: (num_episodes, num_steps, num_channels, x_shape, y_shape)
train_x = np.transpose(train_x, (0, 1, 4, 2, 3)) / 255.

for idx in range(sequences.shape[0]):
    # create sample dictionary
    # 'obs' is an array of the stitched together frames
    # 'next_obs' is also an array of the stitched together frames, however shifted by one
    # 'action' is left as zeros
    sample = {
        'obs': train_x[idx, :-1],
        'next_obs': train_x[idx, 1:],
        'action': np.zeros((train_x.shape[1] - 1, args['stacked_frames']), dtype=np.int64)
    }

    replay_buffer.append(sample)

print()
print(f'Created {len(replay_buffer)} episodes, each with the following data:')
print(f'--- Observations {replay_buffer[0]["obs"].shape}')
print(f'--- Next observations {replay_buffer[0]["next_obs"].shape}')
print(f'--- Actions {replay_buffer[0]["action"].shape}')
print()

save_list_dict_h5py(replay_buffer, output_path)
print("Saved to file", output_path)