import argparse
from utils import ObservationDataset, PerturbationDataset, set_seed
from torch.utils import data
import numpy as np
import torch
from torch.amp import autocast
import matplotlib.pyplot as plt
import json
from slot_attention.AE import SlotAttentionAutoEncoder
from disentangle.latent_AE import LatentAutoEncoder


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PERTURBATION_MAGNITUDE = 0.1

def parse_arguments():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--config', default=None, type=str, help='name of the configuration to use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--test_path', default='data/slipscape/test_data', type=str, help='Path to the test data')
    parser.add_argument('--ckpt_path_SA', default='checkpoints/slot_attention/', type=str, help='directory where the slot attention autoencoder model is saved')
    parser.add_argument('--ckpt_path_disentangle', default='checkpoints/disentangle/', type=str, help='directory where the disentangle model is saved')
    parser.add_argument('--output_dir', default='data/figures/')
    parser.add_argument('--num_output_figs', default=3, type=int, help='desired number of output figures')
    parser.add_argument('--batch_size', default=64, type=int)

    # Image parameters
    parser.add_argument('--hdf5_format', default='CHW', type=str, help='format of train, val and test data frames')
    parser.add_argument('--resolution', default=[64, 64], type=list)
    parser.add_argument('--stacked_frames', default=1, type=int, help='number of frames stacked in each sample')
    parser.add_argument('--channels_per_frame', default=3, type=int, help='number of channels for a single frame')

    # Further Slot Attention parameters
    parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
    parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--encdec_dim', default=32, type=int, help='encoder/decoder dimension size') 

    # Disentanglement parameters
    parser.add_argument('--latent_dim', default=3, type=int, help='Size of the latent space.')

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

    return args


def main():
    args = parse_arguments()
    set_seed(args["seed"])
    ckpt_path_SA = f"{args['ckpt_path_SA']}{args['ckpt_name']}.ckpt"
    ckpt_path_disentangle = f"{args['ckpt_path_disentangle']}{args['ckpt_name']}.ckpt"

    print("Loading model:", ckpt_path_SA)
    model_SA = SlotAttentionAutoEncoder(
        resolution=args["resolution"],
        num_slots=args["num_slots"],
        num_frames = args["stacked_frames"],
        num_iterations=args["num_iterations"], 
        num_channels=args["channels_per_frame"],
        slots_dim=args["slots_dim"], 
        encdec_dim=args["encdec_dim"]).to(DEVICE)
    model_SA.eval() 
    missing_keys, unexpected_keys = model_SA.load_state_dict(torch.load(ckpt_path_SA, weights_only=True)['model_state_dict'], strict=False)
    
    # these keys can be generated again by the model
    generatable_keys = ['encoder_cnn.encoder_pos.grid', 'decoder_cnn.decoder_pos.grid']
    for key in generatable_keys:
        if key in missing_keys:
            missing_keys.remove(key)
    
    if missing_keys:
        raise KeyError(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        raise KeyError(f"Unexpected keys: {unexpected_keys}")

    print("Loading model:", ckpt_path_disentangle)
    model_disentangle = LatentAutoEncoder(
        latent_dim=args["latent_dim"], 
        slots_dim=args["slots_dim"]
    ).to(DEVICE)
    model_disentangle.eval()
    model_disentangle.load_state_dict(torch.load(ckpt_path_disentangle, weights_only=True)['model_state_dict'], strict=True)

    print(f"Loading observation test dataset: {args['test_path']}")
    test_dataset = ObservationDataset(hdf5_file=args["test_path"], hdf5_format=args["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    batch = next(iter(test_dataloader))
    obs = batch[0].to(DEVICE)

    with torch.no_grad():
        recon_SA, _, _, slots_orig, slot_background = model_SA(obs)
        active_slots_recon, z = model_disentangle(slots_orig)
        all_slots_recon = torch.cat((active_slots_recon, slot_background), dim=1)
        print("slots recon", all_slots_recon.shape)
        recon_latent, _, _ = model_SA.decode(all_slots_recon)
        
        object_index = 0
        recon_perturbed_list = []
        
        print("z")
        print(z[0])
        for l in range(args['latent_dim']):
            z_perturbed = z.clone()
            z_perturbed[:, object_index, l] += PERTURBATION_MAGNITUDE
            print(f"z_pert#{l}")
            print(z_perturbed[0])
            active_slots_perturbed = model_disentangle.decode(z_perturbed)
            all_slots_perturbed = torch.concat((active_slots_perturbed, slot_background), dim=1)
            recon_perturbed, _, _ = model_SA.decode(all_slots_perturbed)
            recon_perturbed_list.append(recon_perturbed)
        
        for i in range(args['num_output_figs']):
            # plot image row: original, reconstructed (SA), reconstructed (from latent dim), reconstructions with latent perturbations
            imgs = [obs[i], recon_SA[i], recon_latent[i]]
            labels = ["Original", "Reconstructed (SA)", "Reconstructed (from latent dim)"]
            for j, recon_perturbed in enumerate(recon_perturbed_list):
                imgs.append(recon_perturbed[i])
                labels.append(f"Pert #{j}")
            plot_images(imgs, save_path=f"{args['output_dir']}output_{i}.png", labels=labels)


def plot_frames(orig, masks, combined_recons, save_path="output.png"):
    """
    Plots original frames, combined reconstructions, and masks for each slot.

    Args:
    - orig (numpy.ndarray or torch.Tensor): Original frames of shape [num_frames, 3, height, width].
    - masks (numpy.ndarray or torch.Tensor): Masks of shape [num_frames, slots, height, width].
    - combined_recons (numpy.ndarray or torch.Tensor): Combined reconstructions of shape [num_frames, 3, height, width].
    - save_path (str): File path to save the plot.
    """

    # Convert torch tensors to numpy if necessary
    if hasattr(orig, 'detach'):
        orig = orig.detach().cpu().numpy()
    if hasattr(masks, 'detach'):
        masks = masks.detach().cpu().numpy()
    if hasattr(combined_recons, 'detach'):
        combined_recons = combined_recons.detach().cpu().numpy()

    if len(masks.shape) == 3:
        masks = masks.reshape(1, *masks.shape)

    if len(orig.shape) == 3:
        orig = orig.reshape(1, *orig.shape)

    if len(combined_recons.shape) == 3:
        combined_recons = combined_recons.reshape(1, *combined_recons.shape)

    num_frames, num_slots, height, width = masks.shape
    total_rows = num_slots + 2  # Original, reconstructions, and masks per slot

    fig, axes = plt.subplots(total_rows, num_frames, figsize=(num_frames * 2, total_rows * 2))
    
    if num_frames == 1:
        axes = axes.reshape(total_rows, 1) 

    for f in range(num_frames):
        # Plot original frames (First row)
        axes[0, f].imshow(np.clip(orig[f].transpose(1, 2, 0), 0, 1))
        axes[0, f].axis('off')
        if f == 0:
            axes[0, f].set_ylabel("Original", fontsize=12, fontweight="bold")

        # Plot combined reconstructions (Second row)
        axes[1, f].imshow(np.clip(combined_recons[f].transpose(1, 2, 0), 0, 1))
        axes[1, f].axis('off')
        if f == 0:
            axes[1, f].set_ylabel("Reconstructed", fontsize=12, fontweight="bold")

        # Plot masks for each slot (Remaining rows)
        for s in range(num_slots):
            axes[s + 2, f].imshow(masks[f, s], cmap="gray")
            axes[s + 2, f].axis('off')
            if f == 0:
                axes[s + 2, f].set_ylabel(f"Slot {s+1}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


import matplotlib.pyplot as plt

def plot_images(images, save_path, labels=None):
    """
    Displays all images in a single row and saves the resulting plot.

    Args:
        images (iterable): An iterable of images. Each image should be of shape [3, H, W].
        save_path (str): File path to save the plotted image.
    """
    num_images = len(images)
    images = list(images)

    for i in range(num_images):
        if hasattr(images[i], 'detach'):
            images[i] = images[i].detach().cpu().numpy()
        images[i] = images[i].transpose(1, 2, 0)
    
    # Create a figure with one row and as many columns as there are images.
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    
    # Ensure that axes is always iterable (if only one image, axes is not a list).
    if num_images == 1:
        axes = [axes]
    
    # Loop over images and display each one.
    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].axis('off')
    
    if labels is not None:
        for ax, label in zip(axes, labels):
            ax.set_title(label)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to add padding at the top
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    main()