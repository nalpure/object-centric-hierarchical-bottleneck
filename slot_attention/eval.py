import argparse
from utils import ObservationDataset, set_seed
from torch.utils import data
import numpy as np
import torch
from slot_attention.AE import SlotAttentionAutoEncoder
import matplotlib.pyplot as plt
import json


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--config', default=None, type=str, help='name of the configuration to use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--test_path', default='data/slipscape/test_data', type=str, help='Path to the test data')
    parser.add_argument('--ckpt_path', default='checkpoints/slipscape/', type=str, help='where the model is saved')
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
    criterion = torch.nn.MSELoss()
    set_seed(args["seed"])
    ckpt_path = f"{args['ckpt_path']}{args['ckpt_name']}.ckpt"
    
    print("Loading model:", ckpt_path)
    model = SlotAttentionAutoEncoder(
        resolution=args["resolution"],
        num_slots=args["num_slots"],
        num_frames = args["stacked_frames"],
        num_iterations=args["num_iterations"], 
        num_channels=args["channels_per_frame"],
        slots_dim=args["slots_dim"], 
        encdec_dim=args["encdec_dim"]).to(DEVICE)
    
    model.eval()    
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path, weights_only=True)['model_state_dict'], strict=False)

    # these keys can be generated again by the model
    generatable_keys = ['encoder_cnn.encoder_pos.grid', 'decoder_cnn.decoder_pos.grid']
    for key in generatable_keys:
        if key in missing_keys:
            missing_keys.remove(key)
    
    if missing_keys:
        raise KeyError(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        raise KeyError(f"Unexpected keys: {unexpected_keys}")
    
    print(f"Loading observation test dataset: {args['test_path']}")
    test_dataset = ObservationDataset(hdf5_file=args["test_path"], hdf5_format=args["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    batch = next(iter(test_dataloader))
    obs = batch[0].to(DEVICE)

    with torch.no_grad():
        recon_combined, _, masks, _, _ = model(obs)            

    # split into frames
    obs_frames = obs.view(obs.shape[0], args['stacked_frames'], args['channels_per_frame'], *args['resolution'])
    recon_combined_frames = recon_combined.view(recon_combined.shape[0], args['stacked_frames'], args['channels_per_frame'], *args['resolution'])
    masks_frames = masks.view(masks.shape[0], args['stacked_frames'], args['num_slots'], *args['resolution'])

    # shape of recon_combined is [B, stacked_frames, 3, H, W] 
    for i in range(args['num_output_figs']):
        plot_frames(obs_frames[i], masks_frames[i], recon_combined_frames[i], save_path=f"data/figures/sample{i}.png")
        print(f"Loss {i}:", criterion(obs_frames[i], recon_combined_frames[i]).item())

    print("Overall loss: ", criterion(obs_frames, recon_combined_frames).item())
    
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


if __name__ == '__main__':
    main()