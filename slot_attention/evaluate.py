import argparse
from utils import StateTransitionsDataset
from torch.utils import data
from tqdm import tqdm
import numpy as np
import torch
import json
from slot_attention.slot_attention import SlotAttentionAutoEncoder
import os
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--config', default=None, type=str, help='name of the configuration to use' )
parser.add_argument('--ckpt_path', default='checkpoints/spriteworld/', type=str, help='where the models were saved' )
parser.add_argument('--ckpt_name', default='model', type=str, help='where the models were saved' )

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
            exit(f"{key} is not a valid parameter")

# TODO remove this in the future or make it hyperparameter
stacked_frames = 1
channels_per_frame = 3

num_output_figs = 3 # must be <= number samples per batch
output_dir = 'data/slot_evaluation'
criterion = torch.nn.MSELoss()

epoch = 849 # TODO automize for multiple epochs
full_ckpt_path = args["ckpt_path"]+args["ckpt_name"]+"_"+str(epoch)+"ep.ckpt"


def load_model(checkpoint_path):
    print("Loading model:", checkpoint_path)
    model = SlotAttentionAutoEncoder(resolution=args["resolution"],
                                     num_slots=args["num_slots"], 
                                     num_iterations=args["num_iterations"], 
                                     slots_dim=args["slots_dim"], 
                                     encdec_dim=32 if args["small_arch"] else 64, 
                                     small_arch=args["small_arch"])  # TODO num_frames as input parameter
    
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.to(device)
    model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(device)  # model.to(device) do not move
    model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(device)  # these tensors automatically
    model.eval()
    return model


def get_reconstructions(model, validation_path):
    print("Loading validation dataset:", validation_path)
    validation_dataset = StateTransitionsDataset(hdf5_file=validation_path)
    validate_dataloader = data.DataLoader(validation_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=2)
    all_obss = []
    all_recss = []

    with torch.no_grad():
        for batch in tqdm(validate_dataloader, desc="Processing validation data"):
            # samples consists of 3 lists for observations, actions, next observations.
            # obs and next_obs have shape (num_steps, num_channels, frame_height, frame_width)
            obss, actions, next_obss = batch
            obss = obss[:,:stacked_frames*channels_per_frame,:,:]
            obss = obss.to(device)
            recon_combined, recons, masks, slots = model(obss)

            all_obss.append(obss)
            all_recss.append(recon_combined)
        
    return all_obss, all_recss


# Ensure obs and recon_combined are numpy arrays
# obs.shape and recon_combined.shape = (num_samples, 6, height, width)

def display_images(obss, recss, criterion):
    """
    Display images for comparison between original observations, reconstructions, and their differences.
    
    Args:
    - obss: Tensor of original observations with shape (num_samples, num_channels, height, width).
    - recss: Tensor of reconstructed images with shape (num_samples, num_channels, height, width).
    - criterion: A loss function (e.g., MSE) to compute the loss between the observation and reconstruction.
    """

    # Check how many images each sample contains based on the number of channels
    imgs_per_sample = obss.shape[1] // channels_per_frame

    # Iterate over each sample
    for sample_idx in range(obss.shape[0]):
        # Create a figure with 3 rows and 'imgs_per_sample' columns
        fig, axes = plt.subplots(3, imgs_per_sample, figsize=(4 * imgs_per_sample, 12))

        # If there's only one image per sample, axes will be a 1D array
        if imgs_per_sample == 1:
            axes = np.expand_dims(axes, axis=1)  # Make it 2D for consistent indexing

        for img_idx in range(imgs_per_sample):
            # Get the start and end indices for the current RGB image
            start_idx = img_idx * 3
            end_idx = (img_idx + 1) * 3

            # Extract the RGB image from obs and recon for the current sample
            obs_img = obss[sample_idx, start_idx:end_idx, :, :].cpu().numpy()
            recon_img = recss[sample_idx, start_idx:end_idx, :, :].cpu().numpy()

            # Transpose the images to make them (H, W, C) for display
            obs_img = np.transpose(obs_img, (1, 2, 0))
            recon_img = np.transpose(recon_img, (1, 2, 0))

            diff_img = np.abs(obs_img - recon_img)

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
        loss = criterion(obss[sample_idx], recss[sample_idx]).item()  # Calculate loss for the current sample
        fig.suptitle(f'Loss: {loss:.4f}', fontsize=16, fontweight='bold')

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'comparison_sample_{sample_idx}.png'))
        plt.close()


model = load_model(full_ckpt_path)
validation_path=args['data_path']  # TODO change to actual validation data
#'data/balls_2frame_eval.h5'
all_obss, all_recss = get_reconstructions(model, validation_path)

all_losses = []
for obss, recss in zip(all_obss, all_recss):
    for obs, rec in zip(obss, recss):
        all_losses.append(criterion(obs, rec))

avg_loss = sum(all_losses) / len(all_losses)

print(f"Epoch #{epoch}:")
print(f"Average loss over all samples: {avg_loss:.4f}")
print(f"Min loss: {min(all_losses):.4f}")
print(f"Max loss: {max(all_losses):.4f}")
print()

display_images(all_obss[0][:num_output_figs], all_recss[0][:num_output_figs], criterion)