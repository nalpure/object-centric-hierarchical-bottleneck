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
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')
parser.add_argument('--eval', action='store_true', default=False,
                    help='Create evaluation set.')

args = parser.parse_args()

np.random.seed(args.seed)

physics_sim.generate_3_body_problem_dataset(
    dest=args.fname + '.npz',
    train_set_size=args.num_episodes,
    valid_set_size=2,
    test_set_size=2,
    seq_len=12,
    img_size=[50, 50],
    dt=2.0,
    vx0_max=0.5,
    vy0_max=0.5,
    color=True,
    seed=args.seed
)

# data dictionary, keys: train_x, valid_x, test_x
# e.g. data['train_x] shape: (num_episodes, num_steps, x_shape, y_shape, num_channels)
data = np.load(args.fname + '.npz')

# combine consecutive frames in the channel dimension (hence also loss of one step)
train_x = np.concatenate(
    (data['train_x'][:, :-1], data['train_x'][:, 1:]), axis=-1)

# normalize and rearrange: (num_episodes, num_steps, num_channels, x_shape, y_shape)
train_x = np.transpose(train_x, (0, 1, 4, 2, 3)) / 255.

replay_buffer = []

for idx in range(data['train_x'].shape[0]):
    # create sample dictionary
    # 'obs' is an array of the stitched together frames
    # 'next_obs' is also an array of the stitched together frames, however shifted by one
    # 'action' is left as zeros
    sample = {
        'obs': train_x[idx, :-1],
        'next_obs': train_x[idx, 1:],
        'action': np.zeros((train_x.shape[1] - 1), dtype=np.int64)
    }

    replay_buffer.append(sample)

save_list_dict_h5py(replay_buffer, args.fname)
