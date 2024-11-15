import os
from envs import physics_sim
import numpy as np
import argparse
from utils import save_list_dict_h5py


parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str,
                    default='data',
                    help='File name / path.')
parser.add_argument('--num-episodes', type=int, default=1000,
                    help='Number of episodes to generate.')
parser.add_argument('--samples-per-episode', default=10, type=int, 
                    help='number of samples with should be saved for each episode')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
parser.add_argument('--stacked-frames', default=1, type=int, 
                    help='number of frames stacked in each sample')

args = parser.parse_args()
args = vars(args)

np.random.seed(args['seed'])

# Ensure directory exists
directory = os.path.dirname(args['fname'] + '.h5')
if not os.path.exists(directory):
    os.makedirs(directory)

seq_len = args['stacked_frames'] + args['samples_per_episode'] + 1
print(f'Generating {args["num_episodes"]} 3-body dataset sequences, each consisting of {seq_len} single-frames...')

physics_sim.generate_3_body_problem_dataset(
    dest=args['fname'] + '.npz',
    train_set_size=args['num_episodes'],
    valid_set_size=2,
    test_set_size=2,
    seq_len = seq_len,
    img_size=[50, 50],
    dt=2.0,
    vx0_max=0.5,
    vy0_max=0.5,
    color=True,
    seed=args['seed']
)

# data dictionary, keys: train_x, valid_x, test_x
# e.g. data['train_x] shape: (num_episodes, num_steps, x_shape, y_shape, num_channels)
data = np.load(args['fname'] + '.npz')

replay_buffer = []

# create arrays with shifted training data
shifted_data = [data['train_x'][:, i : i + data['train_x'].shape[1] - args['stacked_frames']] for i in range(args['stacked_frames'])]

# create frame stack with consecutive frames:
# concatenate the differently shifted data arrays in the channel dimension
# the length of the sequence reduces by (stacked_frames - 1)
train_x = np.concatenate(shifted_data, axis=-1)

# normalize and rearrange: (num_episodes, num_steps, num_channels, x_shape, y_shape)
train_x = np.transpose(train_x, (0, 1, 4, 2, 3)) / 255.

for idx in range(data['train_x'].shape[0]):
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

save_list_dict_h5py(replay_buffer, args['fname'] + '.h5')
print("Saved to file", args['fname'] + '.h5')

os.remove(args['fname'] + '.npz')
print('Removed file', args['fname'] + '.npz')
