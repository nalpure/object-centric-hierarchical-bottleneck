import argparse
from utils import ObservationDataset, set_seed
from torch.utils import data
import numpy as np
import torch
from slot_attention.slot_attention import DisentangledSlotAttentionAutoEncoder, SlotAttentionAutoEncoder
import os
import matplotlib.pyplot as plt
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
criterion = torch.nn.MSELoss()

parser.add_argument('--config', default=None, type=str, help='name of the configuration to use' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--hdf5_format', default='CHW', type=str, help='format of train, val and test data frames')
parser.add_argument('--ckpt_path', default='checkpoints/3-body/', type=str, help='where the models were saved' )
parser.add_argument('--output_dir', default='data/figures/')
parser.add_argument('--num_output_figs', default=3, type=int, help='desired number of output figures')
parser.add_argument('--randomize_frame_order', default=False, type=bool, help='If true, reorders the frames in each frame stack randomly.')
parser.add_argument('--disentangle', default=False, action='store_true', help='If true, adds disentanglement loss. Expects training data to include perturbations.')

args = parser.parse_args()
args = vars(args)

if args["config"] is not None:
    args["ckpt_name"] = args["config"]
    with open("configs.json", "r") as config_file:
        configs = json.load(config_file)[args["config"]]
    for key, value in configs.items():
        try:
            args[key] = value
        except KeyError:
            Warning(f"{key} is not a valid parameter")

if args['num_output_figs'] > args['batch_size']:
    raise ValueError("Number of output figures exceeds batch size.")


def load_model(checkpoint_path):
    print("Loading model:", checkpoint_path)
    if args["disentangle"]:
        model = DisentangledSlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["num_slots"],
            args["stacked_frames"] * args["channels_per_frame"],
            args["num_iterations"],
            args["slots_dim"],
            32 if args["small_arch"] else 64,
            args["small_arch"],
            args["latent_dim"]
        ).to(device)
    else:
        model = SlotAttentionAutoEncoder(resolution=args["resolution"],
                                        num_slots=args["num_slots"], 
                                        num_channels=args["stacked_frames"] * args["channels_per_frame"],
                                        num_iterations=args["num_iterations"], 
                                        slots_dim=args["slots_dim"], 
                                        encdec_dim=32 if args["small_arch"] else 64, 
                                        small_arch=args["small_arch"])
    
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.to(device)
    model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(device)  # model.to(device) do not move
    model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(device)  # these tensors automatically
    model.eval()
    return model


def get_reconstructions(model, test_path, max_samples=10000):
    print("Loading test dataset:", test_path)
    test_dataset = ObservationDataset(hdf5_file=test_path, hdf5_format=args["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    all_obs = []
    all_recons = []
    all_masks = []
    all_recon_combined = []
    
    print('Calculating reconstructions for {} batches of size {} (total: {} samples)'.format(
        len(test_dataloader), args['batch_size'], len(test_dataloader) * args['batch_size']))
    
    if len(test_dataloader) * args['batch_size'] > max_samples:
        print(f'Limiting test datasize to {max_samples} samples') 

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):

            if batch_idx * args['batch_size'] > max_samples:
                break

            # samples consists of 3 lists for observations, actions, next observations.
            # obs and next_obs have shape (num_steps, num_channels, frame_height, frame_width)
            obs, action, next_obs = batch
            obs = obs[:,:args['stacked_frames']*args['channels_per_frame'],:,:]

            if args['randomize_frame_order']:
                # Reshape to separate the frames in each stack
                obs = obs.view(args['batch_size'], args['stacked_frames'], args['channels_per_frame'], args['resolution'][0], args['resolution'][1])

                # Randomize the order of frames for each sample in the batch independently
                randomized_obs = []
                for i in range(args['batch_size']):
                    indices = torch.randperm(args['stacked_frames'])  # Generate a random permutation for each sample
                    randomized_obs.append(obs[i, indices, :, :, :])  # Apply the random order to each sample individually

                # Stack the randomized samples back into a single tensor
                obs = torch.stack(randomized_obs)

                # Reshape back to original shape
                obs = obs.view(args['batch_size'], args['stacked_frames'] * args['channels_per_frame'], args['resolution'][0], args['resolution'][1])

            obs = obs.to(device)
            recon_combined, recons, masks, slots = model(obs)

            all_obs.append(obs)
            all_recons.append(recons)
            all_masks.append(masks)
            all_recon_combined.append(recon_combined)
        
    return all_obs, all_recons, all_masks, all_recon_combined


# Ensure obs and recon_combined are numpy arrays
# obs.shape and recon_combined.shape = (num_samples, 6, height, width)

def plot_obs_with_combined_recons(all_obs, all_recons, criterion):
    """
    Display images for comparison between original observations, reconstructions, and their differences.
    
    Args:
    - obs: Tensor of original observations with shape (num_samples, num_channels, height, width).
    - recons: Tensor of reconstructed images with shape (num_samples, num_channels, height, width).
    - criterion: A loss function (e.g., MSE) to compute the loss between the observation and reconstruction.
    """

    # Iterate over each sample
    for sample_idx in range(all_obs.shape[0]):
        # Create a figure with 3 rows and 'imgs_per_sample' columns
        fig, axes = plt.subplots(3, args['stacked_frames'], figsize=(4 * args['stacked_frames'], 12))

        # If there's only one image per sample, axes will be a 1D array
        if args['stacked_frames'] == 1:
            axes = np.expand_dims(axes, axis=1)  # Make it 2D for consistent indexing

        for img_idx in range(args['stacked_frames']):
            # Get the start and end indices for the current RGB image
            start_idx = img_idx * 3
            end_idx = (img_idx + 1) * 3

            # Extract the RGB image from obs and recon for the current sample
            obs_img = all_obs[sample_idx, start_idx:end_idx, :, :].cpu().numpy()
            recon_img = all_recons[sample_idx, start_idx:end_idx, :, :].cpu().numpy()

            # Transpose the images to make them (H, W, C) for display
            obs_img = np.transpose(obs_img, (1, 2, 0))
            recon_img = np.transpose(recon_img, (1, 2, 0))

            diff_img = np.abs(obs_img - recon_img)

            # Clip image values to valid range [0,1]
            obs_img = np.clip(obs_img, 0.0, 1.0)
            recon_img = np.clip(recon_img, 0.0, 1.0)
            diff_img = np.clip(diff_img, 0.0, 1.0)

            # Display the original observation
            axes[0, img_idx].imshow(obs_img)
            axes[0, img_idx].set_title(f't={img_idx}', fontsize=12, fontweight='bold')
            axes[0, img_idx].axis('off')

            # Display the reconstruction
            axes[1, img_idx].imshow(recon_img)
            axes[1, img_idx].axis('off')

            # Display the difference
            axes[2, img_idx].imshow(diff_img)
            axes[2, img_idx].axis('off')

        # Add row titles
        fig.text(0.015, 0.8, 'Truth', va='center', rotation='vertical', fontsize=12, fontweight='bold')
        fig.text(0.015, 0.5, 'Reconstructed', va='center', rotation='vertical', fontsize=12, fontweight='bold')
        fig.text(0.015, 0.2, 'Difference', va='center', rotation='vertical', fontsize=12, fontweight='bold')

        # Compute and add the title (loss between obs and recon)
        loss = criterion(all_obs[sample_idx], all_recons[sample_idx]).item()  # Calculate loss for the current sample
        fig.suptitle(f'Loss: {loss:.8f}', fontsize=16, fontweight='bold')

        # Create the output directory if it doesn't exist
        os.makedirs(args['output_dir'], exist_ok=True)

        # Save the figure
        plt.savefig(os.path.join(args['output_dir'], f'reconstruction_{sample_idx}.png'))
        plt.close()


def plot_observations_with_masks(all_obs, all_masks):
    """
    Plot each observation in `all_obs` and its masks in `all_masks` with frames on the first row 
    and corresponding masks on the second row per plot.
    
    Args:
        all_obs (torch.Tensor): Tensor of shape (num_images, num_channels, height, width) with observations.
        all_masks (torch.Tensor): Tensor of shape (num_images, num_slots, height, width, 1) with masks.
        save_dir (str): Directory path to save the generated plots.
    """   
    num_samples = all_obs.shape[0]
    
    for sample_idx in range(num_samples):
        # Select the current observation and corresponding masks
        observation = all_obs[sample_idx].cpu().numpy()  # Shape: (num_channels, height, width)
        masks = all_masks[sample_idx].cpu().numpy()      # Shape: (num_slots, height, width, 1)
        
        num_slots = masks.shape[0]
        
        # Calculate the number of frames and reshape observation to split frames
        frames = observation.reshape(args['stacked_frames'], args['channels_per_frame'], *observation.shape[1:])  # Shape: (STACKED_FRAMES, CHANNELS_PER_FRAME, height, width)
        
        # Create a figure with stacked frames in the first row and masks in the second
        fig, axes = plt.subplots(2, max(args['stacked_frames'], num_slots), figsize=(4 * max(args['stacked_frames'], num_slots), 8))
        
        # Plot each frame in the first row
        for i in range(args['stacked_frames']):
            frame = frames[i].transpose(1, 2, 0)  # Move channels to the last dimension for RGB plotting
            frame = np.clip(frame, 0.0, 1.0)
            axes[0, i].imshow(frame)
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Frame {i + 1}")
        
        # Fill any remaining empty spaces in the first row if STACKED_FRAMES < num_slots
        for i in range(args['stacked_frames'], max(args['stacked_frames'], num_slots)):
            axes[0, i].axis('off')
        
        # Plot each mask in the second row
        for i in range(num_slots):
            mask = masks[i, :, :, 0]  # Extract 2D mask from shape (height, width, 1)
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Mask {i + 1}")
        
        # Fill any remaining empty spaces in the second row if num_slots < STACKED_FRAMES
        for i in range(num_slots, max(args['stacked_frames'], num_slots)):
            axes[1, i].axis('off')
        
        # Save the plot with a unique name for each observation
        plt.tight_layout()
        plt.savefig(os.path.join(args['output_dir'], f"slot_masks_{sample_idx}.png"))
        plt.close(fig)


def plot_observations_and_reconstructions(all_obs, recons):
    """
    Visualizes each observation and corresponding reconstruction. Observed frames are displayed in the first row.
    Each slot's reconstruction is displayed in subsequent rows.
    
    Args:
    - all_obs (torch.Tensor): The observed frames, shape (num_samples, STACKED_FRAMES * CHANNELS_PER_FRAME, height, width).
    - recons (torch.Tensor): The reconstructed frames, shape (num_samples, num_slots, width, height, STACKED_FRAMES * CHANNELS_PER_FRAME).
    - save_path (str): Directory path to save the plots.
    """

    num_samples = all_obs.shape[0]
    num_slots = recons.shape[1]
    height, width = all_obs.shape[2], all_obs.shape[3]
    recons = recons.permute(0, 1, 4, 2, 3)

    # Loop through each observation and corresponding reconstruction
    for i in range(num_samples):
        fig, axes = plt.subplots(num_slots + 1, args['stacked_frames'], figsize=(4 * args['stacked_frames'], 4 * (num_slots + 1)))

        # If there's only one image per sample, axes will be a 1D array
        if args['stacked_frames'] == 1:
            axes = np.expand_dims(axes, axis=1)  # Make it 2D for consistent indexing

        # Plot observed frames in the first row
        obs_frames = all_obs[i].view(args['stacked_frames'], args['channels_per_frame'], height, width)
        for j in range(args['stacked_frames']):
            frame = obs_frames[j].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) format for plotting
            frame = np.clip(frame, 0.0, 1.0)
            axes[0, j].imshow(frame)
            axes[0, j].set_title(f't={j}', fontsize=12, fontweight='bold')
            axes[0, j].axis('off')
            if j == 0:
                axes[0, j].set_ylabel("Observed", fontsize=12)

        # Plot reconstructed frames for each slot in subsequent rows
        for slot in range(num_slots):
            recon_frames = recons[i, slot].view(args['stacked_frames'], args['channels_per_frame'], height, width)
            for j in range(args['stacked_frames']):
                recon_frame = recon_frames[j].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) format
                recon_frame = np.clip(recon_frame, 0.0, 1.0)
                axes[slot + 1, j].imshow(recon_frame)
                axes[slot + 1, j].axis('off')
                if j == 0:
                    axes[slot + 1, j].set_ylabel(f"Slot {slot + 1}", fontsize=12)

        # Add shared labels for x and y axes
        fig.text(0.5, 0.04, 'Frames', ha='center', fontsize=14)
        fig.text(0.04, 0.5, 'Reconstructions (Slots)', va='center', rotation='vertical', fontsize=14)
        fig.subplots_adjust(left=0.4, right=0.9, bottom=0.2, top=0.95)  # Add space for labels

        # Save the figure for the current observation and reconstruction
        plt.tight_layout(pad=4)
        plt.savefig(os.path.join(args['output_dir'], f"slot_recons_{i}.png"))
        plt.close(fig)


set_seed(args["seed"])

model = load_model(f'{args["ckpt_path"]}{args["ckpt_name"]}.ckpt')
all_obs, all_recons, all_masks, all_recon_combined = get_reconstructions(model, args['test_path'])

loss_list = np.array([criterion(obs, rec).item() for obs, rec in zip(all_obs, all_recon_combined)])

for i, loss in enumerate(loss_list):
    print(f'Loss batch #{i}: {loss:6f}')

print()
print(f"Mean batch loss: {np.mean(loss_list):.6f}")

plot_obs_with_combined_recons(all_obs[0][:args['num_output_figs']], all_recon_combined[0][:args['num_output_figs']], criterion)
plot_observations_with_masks(all_obs[0][:args['num_output_figs']], all_masks[0][:args['num_output_figs']])
plot_observations_and_reconstructions(all_obs[0][:args['num_output_figs']], all_recons[0][:args['num_output_figs']])