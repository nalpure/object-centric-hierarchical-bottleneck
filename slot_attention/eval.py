import argparse
from utils import ObservationDataset, PerturbationDataset, set_seed
from torch.utils import data
import numpy as np
import torch
from torch.amp import autocast
from slot_attention.AE import SlotAttentionAutoEncoder
from slot_attention.disentangled_AE import DisentangledSlotAttentionAutoEncoder
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


def main():
    set_seed(args["seed"])
    disentangled = args["train_PH"] or args["train_SA_disentangled"]
    model = load_model(f'{args["ckpt_path"]}{args["ckpt_name"]}_SA_disentangled.ckpt', disentangled) #TODO filename

    print(f"Loading {'perturbation' if disentangled else 'observation'} test dataset: {args['test_path']}")
    TestDataclass = PerturbationDataset if disentangled else ObservationDataset 
    test_dataset = TestDataclass(hdf5_file=args["test_path"], hdf5_format=args["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    batch = next(iter(test_dataloader))
    obs = batch[0].to(device)

    with torch.no_grad():
        obs_forwarded = model(obs)    
        recon_combined = obs_forwarded[0]
        masks = obs_forwarded[2]

        if disentangled:
            slots = obs_forwarded[3]
            z_obs = obs_forwarded[4]

            _, obs_perturbed, magnitudes, _, properties = batch
            obs_perturbed = obs_perturbed.to(device)
            magnitudes = magnitudes.to(device)
            
            recon_combined_perturbed, _, masks_perturbed, _, z_perturbed = model(obs_perturbed, init_slots = slots)
            obs_perturbed_frames = obs_perturbed.view(obs_perturbed.shape[0], args['stacked_frames'], args['channels_per_frame'], *args['resolution'])
            recon_combined_perturbed_frames = recon_combined_perturbed.view(recon_combined_perturbed.shape[0], args['stacked_frames'], args['channels_per_frame'], *args['resolution'])
            masks_perturbed_frames = masks_perturbed.view(masks_perturbed.shape[0], args['stacked_frames'], args['num_slots'], *args['resolution'])
            
            for i in range(args['num_output_figs']):
                plot_frames(obs_perturbed_frames[i], masks_perturbed_frames[i], recon_combined_perturbed_frames[i], save_path=f"data/figures/perturbed{i}.png")
            
            disentanglement_score(z_obs, z_perturbed, magnitudes, properties)
            

    # split into frames
    obs_frames = obs.view(obs.shape[0], args['stacked_frames'], args['channels_per_frame'], *args['resolution'])
    recon_combined_frames = recon_combined.view(recon_combined.shape[0], args['stacked_frames'], args['channels_per_frame'], *args['resolution'])
    masks_frames = masks.view(masks.shape[0], args['stacked_frames'], args['num_slots'], *args['resolution'])

    # shape of recon_combined is [B, stacked_frames, 3, H, W] 
    for i in range(args['num_output_figs']):
        plot_frames(obs_frames[i], masks_frames[i], recon_combined_frames[i], save_path=f"data/figures/original{i}.png")
        print(f"Loss {i}:", criterion(obs_frames[i], recon_combined_frames[i]).item())

    print("Overall loss: ", criterion(obs_frames, recon_combined_frames).item())

def load_model(checkpoint_path, disentangled=False):
    print("Loading model:", checkpoint_path)
    if disentangled:
        model = DisentangledSlotAttentionAutoEncoder(
            tuple(args["resolution"]),
            args["stacked_frames"],
            args["channels_per_frame"],
            args["num_slots"],
            args["num_iterations"],
            args["slots_dim"],
            args["encdec_dim"],
            args["latent_dim"]
        )
    else:
        model = SlotAttentionAutoEncoder(
            resolution=args["resolution"],
            num_slots=args["num_slots"],
            num_frames = args["stacked_frames"],
            num_iterations=args["num_iterations"], 
            num_channels=args["channels_per_frame"],
            slots_dim=args["slots_dim"], 
            encdec_dim=args["encdec_dim"])
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def disentanglement_score(z_obs, z_perturbed, magnitudes, properties):
    """
    @param z_obs: torch.Tensor, [B, S, D]
        Latent representation of the original observation.
    @param z_perturbed: torch.Tensor, [B, S, D]
        Latent representation of the perturbed observation.
    @param magnitudes: torch.Tensor, [B]
        Magnitude of the perturbation.
    @param properties: torch.Tensor, [B]
        Number code of the property that was actually perturbed.
    """
    latent_dim = z_obs.shape[2]

    eye = torch.eye(latent_dim, device=device)
    deltas = eye.unsqueeze(0) * magnitudes[:, None, None]               # [B, D, D]

    z_obs_expanded = z_obs.unsqueeze(2).unsqueeze(2)                    # [B, S, 1, 1, D]
    z_perturbed_expanded = z_perturbed.unsqueeze(1).unsqueeze(3)        # [B, 1, S, 1, D]
    deltas_expanded = deltas.unsqueeze(1).unsqueeze(1)                  # [B, 1, 1, D, D]

    diff_delta = z_perturbed_expanded - (z_obs_expanded + deltas_expanded)      # [B, S, S, D, D]
    diff_norm = torch.linalg.vector_norm(diff_delta, dim=-1)                   # [B, S, S, D]
    
    losses = diff_norm.min(dim=-1).values.min(dim=-1).values.min(dim=-1).values # [B]

    helper = (z_perturbed_expanded - z_obs_expanded).squeeze()

    print("deltas:\n", deltas[0])
    print("z_pert - z_obs:\n", helper[0])
    #print("diff_delta:\n", diff_delta[0][0])
    print("loss: ", losses[0])
    print("z_obs_expanded:\n", z_obs_expanded.squeeze()[0])
    print("z_perturbed_expanded:\n", z_perturbed_expanded.squeeze()[0])
    #print("mean batch disentanglement loss: ", losses.mean().item())

    print("properties: ", properties)
    #TODO remove
    # replace all 4s with 2s
    properties = np.where(properties == 4, 2, properties)
    z_obs_reduced = z_obs_expanded.squeeze(3)               # [B, S, 1, D]
    z_perturbed_reduced = z_perturbed_expanded.squeeze(3)   # [B, 1, S, D]

    # for each property perturbation, count the number of times each latent dimension is perturbed
    perturbed_index_count = np.zeros((latent_dim, latent_dim), dtype=np.int32)

    for sample_idx in range(z_obs.shape[0]):
        # find index of the minimum loss
        min_loss_idx = torch.argmin(diff_norm[sample_idx])
        min_loss_idx = torch.unravel_index(min_loss_idx, diff_norm[sample_idx].shape)                                     # tuple of length 3

        diff = torch.abs(z_perturbed_reduced[sample_idx] - z_obs_reduced[sample_idx]) # [S, S, D]
        #print("diff", diff)
        # find indices of maximum difference along the latent dimension
        _, latent_indices = torch.max(diff, dim=-1)                                          # [S, S]

        # of lowest loss slot combination, find the latent index with the highest difference
        latent_index = latent_indices[min_loss_idx[0], min_loss_idx[1]]

        perturbed_index_count[properties[sample_idx], latent_index] += 1

        #print(f"latent index: {latent_index}, property: {properties[sample_idx]}, loss: {losses[sample_idx]}")
        #print(f"latent indices: {latent_indices}")

    print(perturbed_index_count)

    
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